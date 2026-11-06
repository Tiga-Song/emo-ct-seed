"""Confusion-matrix figure for the SEED experiments.

Adapted from ``CTNet-main/plot_figures/confusion_matrix.py``, with three changes
that matter:

* three emotion classes instead of the motor-imagery class sets;
* it reads the per-window CSVs written by the runner rather than Excel sheets,
  so the figure can be regenerated from raw predictions at any time;
* the raw integer counts are plotted alongside the row-normalised rates.  The
  figure this replaces (Fig. 2 of the manuscript) showed percentages only,
  carried an embedded "CTNet on SEED Dataset" title, and every published
  percentage in the text was wrong.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np

from stats_analysis import confusion_from_predictions

CLASS_NAMES = ("Negative", "Neutral", "Positive")


def draw(cm: np.ndarray, counts: np.ndarray, title: str, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, fmt, label in (
        (axes[0], counts, "d", "counts"),
        (axes[1], cm * 100.0, ".2f", "row-normalised (%)"),
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(label)
        threshold = data.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > threshold else "black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "experiments"))
    ap.add_argument("--protocol", default="cross_session")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--models", default="ctnet_add,emo_ct_cat")
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args(argv)

    save_dir = args.save_dir or os.path.join(args.out, "aggregate", "figures")
    os.makedirs(save_dir, exist_ok=True)

    models: List[str] = [m for m in args.models.split(",") if m.strip()]
    for model in models:
        try:
            pooled_norm, per_unit, pooled_counts = confusion_from_predictions(
                args.out, args.protocol, args.window, model, per_subject=False)
            subject_norm, _, _ = confusion_from_predictions(
                args.out, args.protocol, args.window, model, per_subject=True)
        except SystemExit as exc:
            print("  skipped %s: %s" % (model, exc))
            continue

        title = "%s - SEED (%s, %.0f s windows, n=%d units)" % (
            model, args.protocol.replace("_", " "), args.window, len(per_unit))
        path = os.path.join(save_dir, "confusion_%s_%s_%gs.pdf" % (model, args.protocol, args.window))
        draw(pooled_norm, pooled_counts, title, path)
        print("wrote %s" % path)

        print("  pooled raw counts (rows = true, cols = predicted):")
        print("    " + "\n    ".join(str(row) for row in pooled_counts.tolist()))
        diagonal = np.diag(pooled_norm) * 100
        print("  pooled per-class recall: " + ", ".join(
            "%s %.2f%%" % (CLASS_NAMES[i], diagonal[i]) for i in range(3)))
        print("  pooled diagonal mean: %.2f%%  (= overall accuracy)"
              % (diagonal.mean()))
        print("  per-unit-averaged diagonal mean: %.2f%%"
              % (np.diag(subject_norm).mean() * 100))

        with open(os.path.join(save_dir, "confusion_counts_%s_%s_%gs.txt"
                               % (model, args.protocol, args.window)), "w", encoding="utf-8") as fh:
            fh.write("# pooled raw counts\n")
            fh.write(str(pooled_counts.tolist()) + "\n")
            fh.write("# per-unit raw counts (unit, counts)\n")
            for unit in sorted(per_unit):
                fh.write("%d %s\n" % (unit, per_unit[unit].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
