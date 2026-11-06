"""Single entry point for constructing any model used in the experiments.

Every model is built from ``(window_len, n_channels, n_classes, dropout_rate)``
so the experiment matrix can treat them interchangeably, and every model returns
raw logits of shape ``(B, n_classes)``.
"""
from __future__ import annotations

from typing import Dict

import torch.nn as nn

from baseline_models import Conformer, DeepConvNet, EEGNet, ShallowConvNet
from model_ctnet import build_ctnet

# The six models compared in the main tables.
#
#   ctnet_add  = CTNet as published (CNN and Transformer outputs added)
#   emo_ct_cat = the same backbone with the two branches concatenated instead,
#                which is the only architectural change this project introduces
#
MAIN_MODELS = ("ctnet_add", "emo_ct_cat", "eegnet", "shallowconvnet",
               "deepconvnet", "conformer")

# Extra variants used only in the ablation table.
ABLATION_MODELS = (
    "emo_ct_cat_cnn_only",      # concat fusion but the Transformer is removed
    "ctnet_add_cnn_only",       # CTNet without its Transformer
)

ALL_MODELS = MAIN_MODELS + ABLATION_MODELS

_CTNET_VARIANTS: Dict[str, Dict] = {
    "ctnet_add": {"fusion": "add", "use_transformer": True},
    "emo_ct_cat": {"fusion": "cat", "use_transformer": True},
    "ctnet_add_cnn_only": {"fusion": "add", "use_transformer": False},
    "emo_ct_cat_cnn_only": {"fusion": "cat", "use_transformer": False},
}

_BASELINES = {
    "eegnet": EEGNet,
    "shallowconvnet": ShallowConvNet,
    "deepconvnet": DeepConvNet,
    "conformer": Conformer,
}


def build_model(name: str, window_len: int, n_channels: int = 62,
                n_classes: int = 3, dropout_rate: float = 0.5,
                head_dropout: float = 0.5, **overrides) -> nn.Module:
    """Instantiate ``name`` for a given window length.

    ``dropout_rate`` is CTNet's ``eeg1_dropout_rate`` / the baselines' dropout
    and is set identically for every model so the comparison is like-for-like.
    """
    if name in _CTNET_VARIANTS:
        kwargs = dict(_CTNET_VARIANTS[name])
        kwargs.update(overrides)
        return build_ctnet(
            window_len=window_len, n_channels=n_channels, n_classes=n_classes,
            dropout_rate=dropout_rate, head_dropout=head_dropout, **kwargs
        )

    if name in _BASELINES:
        cls = _BASELINES[name]
        kwargs = dict(overrides)
        if name == "conformer":
            # Conformer's own head dropout is hardcoded at 0.5 upstream
            return cls(number_channel=n_channels, nb_classes=n_classes,
                       dropout_rate=dropout_rate, signal_length=window_len, **kwargs)
        return cls(number_channel=n_channels, nb_classes=n_classes,
                   dropout_rate=dropout_rate, signal_length=window_len, **kwargs)

    raise ValueError("unknown model %r; known: %s" % (name, ", ".join(ALL_MODELS)))


def classifier_width(model: nn.Module) -> int:
    """Flat width entering the classifier - used by the shape self-check."""
    if hasattr(model, "flatten_eeg1"):
        return int(model.flatten_eeg1)
    if hasattr(model, "classifier"):
        return int(model.classifier.in_features)
    if hasattr(model, "fc"):
        return int(model.fc[-1].in_features)
    raise AttributeError("cannot determine classifier width for %r" % type(model).__name__)


def forward_logits(model: nn.Module, x):
    """Uniform logits accessor.

    The CTNet backbone returns ``(features, logits)`` because its Transformer
    output features are needed for the t-SNE figure, while the baselines return
    logits directly.  The training loop must not care which it is holding.
    """
    out = model(x)
    if isinstance(out, tuple):
        return out[-1]
    return out


def selfcheck() -> None:
    import torch

    print("%-22s %-6s %-10s %-9s %s" % ("model", "W", "head_dim", "params", "out"))
    for name in ALL_MODELS:
        for window_len in (1000, 2000):
            try:
                model = build_model(name, window_len=window_len)
                out = forward_logits(model, torch.zeros(2, 1, 62, window_len))
            except ValueError as exc:
                print("%-22s %-6d  skipped: %s" % (name, window_len, exc))
                continue
            n_params = sum(p.numel() for p in model.parameters())
            print("%-22s %-6d %-10d %-9d %s" % (
                name, window_len, classifier_width(model), n_params, tuple(out.shape)))
            assert out.shape == (2, 3), out.shape
    print("\nselfcheck: PASS")


if __name__ == "__main__":
    selfcheck()
