"""CTNet backbone, adapted for the SEED emotion dataset.

Attribution
-----------
The architecture implemented here is CTNet, introduced by

    Zhao W, Jiang X, Zhang B, Xiao S, Weng S.
    "CTNet: a convolutional transformer network for EEG-based motor imagery
    classification." Scientific Reports 14, 20237 (2024).
    https://doi.org/10.1038/s41598-024-71118-7

released under the MIT License, Copyright (c) 2024 snailpt, at
https://github.com/snailpt/CTNet .  This file is a derivative of
``CTNet-main/main_subject_specific.py`` from that repository.

Changes relative to the original (nothing else is modified)
----------------------------------------------------------
1.  Class/channel counts are supplied for SEED (3 classes, 62 channels) instead
    of the hardcoded BCI IV-2a / IV-2b registry.
2.  ``flatten_eeg1`` (the classifier input width) is *derived* from the window
    length instead of being a hardcoded constant, and the token count is
    computed as ``W // pooling_size1 // pooling_size2``.
3.  A ``fusion`` switch selects how the CNN and Transformer branches are
    combined: ``'add'`` reproduces CTNet exactly (``features = cnn + trans``),
    ``'cat'`` is the Emo-CT variant (``torch.cat([cnn, trans], dim=-1)``).
4.  ``PositioinalEncoding`` no longer calls ``.cuda()`` inside ``forward``
    (which broke CPU use and was redundant with ``module.cuda()``), and its
    positional table length is set to the actual token count.
5.  ``ClassificationHead`` dropout is a parameter rather than a hardcoded 0.5.
6.  ``TransformerEncoderBlock`` dropout / expansion are reachable from the
    top-level model instead of always taking their defaults.
7.  An optional ``use_transformer=False`` path supports the no-Transformer
    ablation without touching the block definitions.

The CNN and Transformer blocks themselves are byte-for-byte the original
computation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor


def number_class_channel(database_type: str) -> Tuple[int, int]:
    """(n_classes, n_channels) for the supported datasets.

    Kept alongside the model so the SEED branch is explicit rather than a
    silent edit to the upstream helper.
    """
    if database_type == "A":          # BCI Competition IV-2a
        return 4, 22
    if database_type == "B":          # BCI Competition IV-2b
        return 2, 3
    if database_type == "SEED":       # SEED emotion recognition
        return 3, 62
    raise ValueError("unknown dataset type %r" % (database_type,))


# ---------------------------------------------------------------------------
# CNN branch (original: main_subject_specific.py:52)
# ---------------------------------------------------------------------------


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            # temporal convolution; kernel_size = sampling_rate / 4
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            # channel depth-wise convolution
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding="valid", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            # spatial convolution
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(Rearrange("b e (h) (w) -> b (h w) e"))

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn_module(x)
        return self.projection(x)


def conv_token_count(window_len: int, pooling_size1: int, pooling_size2: int) -> int:
    """Number of tokens the CNN branch emits for a window of ``window_len``."""
    after_pool1 = window_len // pooling_size1
    return max(after_pool1 // pooling_size2, 1)


# ---------------------------------------------------------------------------
# Transformer encoder (original: lines 94-184)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Module):
    """Original head, with the dropout made configurable (change 5)."""

    def __init__(self, flatten_number, n_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(flatten_number, n_classes))

    def forward(self, x):
        return self.fc(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        return self.layernorm(self.drop(res) + x_input)


class TransformerEncoderBlock(nn.Module):
    """Original block, with dropout/expansion made reachable (change 6)."""

    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__()
        self.attention = ResidualAdd(
            MultiHeadAttention(emb_size, num_heads, drop_p), emb_size, drop_p
        )
        self.feed_forward = ResidualAdd(
            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            emb_size, drop_p,
        )
        self.attn = self.attention
        self.feed = self.feed_forward

    def forward(self, x):
        x = self.attn(x)
        x = self.feed(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size, drop_p=0.5, forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(*[
            TransformerEncoderBlock(emb_size, heads, drop_p, forward_expansion, forward_drop_p)
            for _ in range(depth)
        ])


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=2, depth=6, emb_size=16, number_channel=22, f1=8,
                 kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, **kwargs):
        super().__init__(
            PatchEmbeddingCNN(
                f1=f1, kernel_size=kernel_size, D=D,
                pooling_size1=pooling_size1, pooling_size2=pooling_size2,
                dropout_rate=dropout_rate, number_channel=number_channel,
                emb_size=emb_size,
            )
        )


class PositionalEncoding(nn.Module):
    """Learnable positional table (original: line 214), with two fixes.

    * the hardcoded ``.cuda()`` in ``forward`` is gone - the parameter already
      moves with ``module.cuda()``, and the call made the module CPU-hostile;
    * ``length`` is set to the actual token count instead of a fixed 100.
    """

    def __init__(self, embedding, length, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, : x.shape[1], :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Full model (original: line 226)
# ---------------------------------------------------------------------------


class EEGTransformer(nn.Module):
    def __init__(self, heads=2, emb_size=16, depth=6, database_type="SEED",
                 eeg1_f1=8, eeg1_kernel_size=50, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.5, window_len=2000,
                 fusion="cat", use_transformer=True,
                 head_dropout=0.5, transformer_dropout=0.5,
                 forward_expansion=4, **kwargs):
        super().__init__()
        self.number_class, self.number_channel = number_class_channel(database_type)

        assert eeg1_D * eeg1_f1 == emb_size, (
            "CNN output channels (D*f1 = %d) must equal the transformer width "
            "(emb_size = %d), otherwise the two branches cannot be combined"
            % (eeg1_D * eeg1_f1, emb_size)
        )
        if fusion not in ("add", "cat"):
            raise ValueError("fusion must be 'add' or 'cat', got %r" % (fusion,))

        self.fusion = fusion
        self.use_transformer = bool(use_transformer)
        self.emb_size = emb_size
        self.window_len = int(window_len)
        self.depth = int(depth)

        n_tokens = conv_token_count(self.window_len, eeg1_pooling_size1, eeg1_pooling_size2)
        self.n_tokens = n_tokens
        # change 2: derived, not hardcoded.  'cat' doubles the flat width.
        self.flatten_eeg1 = n_tokens * emb_size * (2 if fusion == "cat" else 1)

        self.cnn = BranchEEGNetTransformer(
            heads=heads, depth=depth, emb_size=emb_size,
            number_channel=self.number_channel, f1=eeg1_f1,
            kernel_size=eeg1_kernel_size, D=eeg1_D,
            pooling_size1=eeg1_pooling_size1, pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate,
        )
        self.position = PositionalEncoding(emb_size, length=n_tokens, dropout=0.1)
        # change 7: the ablation variant must not carry dead Transformer weights,
        # otherwise its parameter count misreports what was actually run.  The
        # wider flat dimension is kept (zero-padded) so that the ablation differs
        # from the full model only in the Transformer's contribution.
        self.trans = (
            TransformerEncoder(heads, depth, emb_size,
                               drop_p=transformer_dropout,
                               forward_expansion=forward_expansion)
            if self.use_transformer else None
        )
        self.classification = ClassificationHead(self.flatten_eeg1, self.number_class,
                                                 dropout=head_dropout)
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)

        if self.use_transformer:
            trans = self.trans(cnn)
        else:
            trans = torch.zeros_like(cnn)

        if self.fusion == "cat":
            features = torch.cat([cnn, trans], dim=-1)
        else:
            features = cnn + trans

        out = self.classification(self.flatten(features))
        return features, out


def build_ctnet(window_len: int, fusion: str = "cat", n_channels: int = 62,
                n_classes: int = 3, heads: int = 2, depth: int = 6, emb_size: int = 16,
                f1: int = 8, D: int = 2, kernel_size: Optional[int] = None,
                pooling_size1: int = 8, pooling_size2: int = 8,
                dropout_rate: float = 0.5, head_dropout: float = 0.5,
                transformer_dropout: float = 0.5, use_transformer: bool = True,
                database_type: str = "SEED") -> EEGTransformer:
    """Instantiate the adapted CTNet.

    ``kernel_size`` defaults to ``sampling_rate / 4`` for the 200 Hz SEED data
    (i.e. 50), matching CTNet's stated rule that the temporal kernel is
    one-fourth of the sampling rate.
    """
    if kernel_size is None:
        kernel_size = 200 // 4
    model = EEGTransformer(
        heads=heads, emb_size=emb_size, depth=depth, database_type=database_type,
        eeg1_f1=f1, eeg1_kernel_size=kernel_size, eeg1_D=D,
        eeg1_pooling_size1=pooling_size1, eeg1_pooling_size2=pooling_size2,
        eeg1_dropout_rate=dropout_rate, window_len=window_len,
        fusion=fusion, use_transformer=use_transformer,
        head_dropout=head_dropout, transformer_dropout=transformer_dropout,
    )
    # sanity: the channel/class registry must agree with what was asked for
    assert model.number_channel == n_channels, (model.number_channel, n_channels)
    assert model.number_class == n_classes, (model.number_class, n_classes)
    return model


def selfcheck() -> None:
    """Shape and width checks; runnable without a GPU."""
    print("%-6s %-5s %-6s %-9s %-8s %s" % ("W", "fus", "tokens", "flat_dim", "params", "out"))
    for window_len in (400, 800, 1000, 2000, 6000):
        for fusion in ("add", "cat"):
            m = build_ctnet(window_len=window_len, fusion=fusion)
            x = torch.zeros(2, 1, 62, window_len)
            feats, out = m(x)
            n_params = sum(p.numel() for p in m.parameters())
            print("%-6d %-5s %-6d %-9d %-8d %s" % (
                window_len, fusion, m.n_tokens, m.flatten_eeg1, n_params, tuple(out.shape)))
            assert out.shape == (2, 3), out.shape
            expect_tokens = window_len // 8 // 8
            assert m.n_tokens == max(expect_tokens, 1), (m.n_tokens, expect_tokens)
    # the original configuration must reproduce the original width
    m = build_ctnet(window_len=1000, fusion="add", n_channels=22, n_classes=4,
                    database_type="A", kernel_size=64)
    assert m.flatten_eeg1 == 240, m.flatten_eeg1
    print("original-config check: window_len=1000 -> flatten 240  (matches CTNet)\n")
    print("selfcheck: PASS")


if __name__ == "__main__":
    selfcheck()
