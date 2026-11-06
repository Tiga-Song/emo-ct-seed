"""Baseline architectures for the SEED experiments.

Sources (all from the CTNet repository, ``CTNet-main/other models.py`` and
``CTNet-main/Conformer_fine_tune_2a_77.66_2b_85.87.ipynb``):

* ``EEGNet``         - Lawhern et al., J Neural Eng 15, 056013 (2018)
* ``ShallowConvNet`` - Schirrmeister et al., Hum Brain Mapp 38, 5391 (2017)
* ``DeepConvNet``    - Schirrmeister et al., Hum Brain Mapp 38, 5391 (2017)
* ``Conformer``      - Song et al., IEEE TNSRE 31, 710 (2022)

What was changed, and why
-------------------------
Only the classifier input width.  The upstream definitions hardcode
``nn.Linear(2440, ...)`` and ``nn.Linear(1400, ...)``, which are the flat
widths for 22 channels x 1000 samples.  SEED is 62 channels x a window that the
window-length pilot has not fixed yet, so the width must be derived.

Every baseline's internal pooling, kernels, normalisation and residual layout
are copied verbatim - a reviewer will check that the baselines are faithful, and
quietly retuning their receptive fields would be exactly the kind of thing this
project exists to stop doing.  The head width is obtained by running a single
zero tensor through the feature stack.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


def _flat_width(module: nn.Module, feature_fn: Callable[[Tensor], Tensor],
                number_channel: int, signal_length: int) -> int:
    """Flat width produced by ``feature_fn`` for a single dummy input.

    Run in eval mode: in train mode a ``BatchNorm2d`` whose spatial extent has
    collapsed raises a confusing batch-statistics error instead of telling us
    the window is too short.  Any failure is reported as a clear statement that
    this baseline's canonical pooling does not apply at that length - we never
    silently retune the pooling to make it fit.
    """
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            out = feature_fn(torch.zeros(1, 1, number_channel, signal_length))
        if out.shape[-1] <= 0:
            raise ValueError("time axis collapses to %d steps" % out.shape[-1])
        return int(out.flatten(1).shape[1])
    except Exception as exc:
        raise ValueError(
            "this baseline's canonical pooling cannot be applied at "
            "signal_length=%d samples: %s" % (signal_length, exc)
        ) from exc
    finally:
        module.train(was_training)


# ---------------------------------------------------------------------------
# EEGNet
# ---------------------------------------------------------------------------


class EEGNet(nn.Module):
    """Verbatim from ``other models.py``; already parametric in signal_length."""

    def __init__(self, nb_classes=3, kernel_size=125, number_channel=62,
                 signal_length=2000, dropoutRate=0.5, pooling_size1=8,
                 pooling_size2=8, f1=8, D=2, f2=16, norm_rate=0.25,
                 dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding="same", bias=False)
        self.batch_norm1 = nn.BatchNorm2d(f1)

        self.depthwise_conv = nn.Conv2d(f1, f1 * D, (number_channel, 1), (1, 1),
                                        groups=f1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(f1 * D)
        self.elu = nn.ELU()

        self.avg_pool1 = nn.AvgPool2d((1, pooling_size1))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.separable_conv = nn.Conv2d(f1 * D, f2, (1, 16), padding="same",
                                        groups=f1 * D, bias=False)
        self.seperable_conv_1x1 = nn.Conv2d(f2, f2, (1, 1), padding="same", bias=False)
        self.batch_norm3 = nn.BatchNorm2d(f2)

        self.avg_pool2 = nn.AvgPool2d((1, pooling_size2))
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(
            f2 * (signal_length // pooling_size1 // pooling_size2), nb_classes)
        self.norm_rate = norm_rate

        # upstream registers these to clamp weights after every step
        self.depthwise_conv.register_forward_pre_hook(self.apply_max_norm_depthwise)
        self.classifier.register_forward_pre_hook(self.apply_max_norm_classifier)

    def _features(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        x = self.separable_conv(x)
        x = self.seperable_conv_1x1(x)
        x = self.batch_norm3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def apply_max_norm_depthwise(self, module, inputs):
        with torch.no_grad():
            norm = self.depthwise_conv.weight.data.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=1.0)
            self.depthwise_conv.weight.data *= desired / (norm + 1e-8)

    def apply_max_norm_classifier(self, module, inputs):
        with torch.no_grad():
            norm = self.classifier.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.norm_rate)
            self.classifier.weight.data *= desired / (norm + 1e-8)


# ---------------------------------------------------------------------------
# ShallowConvNet
# ---------------------------------------------------------------------------


class ShallowConvNet(nn.Module):
    def __init__(self, number_channel=62, nb_classes=3, dropout_rate=0.5,
                 signal_length=2000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # pooling acts as slicing to obtain 'patch' along the time axis, as in ViT
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout_rate),
        )
        self.flatten = nn.Flatten()
        # upstream hardcodes 2440 (= 40 x 61 for 1000 samples)
        width = _flat_width(self, self.features, number_channel, signal_length)
        self.classifier = nn.Linear(width, nb_classes)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


# ---------------------------------------------------------------------------
# DeepConvNet
# ---------------------------------------------------------------------------


class DeepConvNet(nn.Module):
    def __init__(self, number_channel=62, nb_classes=3, dropout_rate=0.5,
                 signal_length=2000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 3)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 3)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 3)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 3)),
            nn.Dropout(dropout_rate),
        )
        self.flatten = nn.Flatten()
        # upstream hardcodes 1400 (= 200 x 7 for 1000 samples).
        # Its four stride-3 max-pools need signal_length >= 441 (2.2 s at 200 Hz);
        # _flat_width raises a clear error below that instead of silently
        # changing the canonical pooling.
        width = _flat_width(self, self.features, number_channel, signal_length)
        self.classifier = nn.Linear(width, nb_classes)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


# ---------------------------------------------------------------------------
# Conformer
# ---------------------------------------------------------------------------


class _MultiHeadAttention(nn.Module):
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
            energy.mask_fill(~mask, torch.finfo(torch.float32).min)
        att = F.softmax(energy / (self.emb_size ** (1 / 2)), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        return self.projection(rearrange(out, "b h n d -> b n (h d)"))


class _ResidualAdd(nn.Module):
    """Conformer's residual: normalisation sits *inside* the wrapped block."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _ConformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class _ConformerEncoder(nn.Sequential):
    def __init__(self, num_heads, depth, emb_size):
        super().__init__(*[_ConformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])


class Conformer(nn.Module):
    def __init__(self, number_channel=62, nb_classes=3, dropout_rate=0.5,
                 signal_length=2000, num_heads=10, depth=6, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.trans = _ConformerEncoder(num_heads, depth, emb_size)
        self.flatten = nn.Flatten()
        # upstream hardcodes 2440
        width = _flat_width(self, self._features, number_channel, signal_length)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(width, nb_classes))

    def _features(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        x = self.trans(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.flatten(self._features(x)))


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------


BASELINES = {
    "eegnet": EEGNet,
    "shallowconvnet": ShallowConvNet,
    "deepconvnet": DeepConvNet,
    "conformer": Conformer,
}


def selfcheck() -> None:
    print("%-16s %-6s %-10s %s" % ("model", "W", "flat_dim", "out"))
    for name, cls in BASELINES.items():
        for window_len in (400, 1000, 2000, 6000):
            try:
                model = cls(number_channel=62, nb_classes=3, signal_length=window_len)
                out = model(torch.zeros(2, 1, 62, window_len))
            except ValueError as exc:
                print("%-16s %-6d  skipped: %s" % (name, window_len, exc))
                continue
            width = model.classifier.in_features if hasattr(model, "classifier") \
                else model.fc[-1].in_features
            print("%-16s %-6d %-10d %s" % (name, window_len, width, tuple(out.shape)))
            assert out.shape == (2, 3), out.shape
    print("\nselfcheck: PASS")


if __name__ == "__main__":
    selfcheck()
