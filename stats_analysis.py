"""Aggregation, significance testing and effect sizes.

Nothing here existed in the CTNet repository: it computes accuracy/kappa but no
p-values, no effect sizes and no confidence intervals.  Because the manuscript
reports p-values and kappas, those have to be produced from the per-subject
predictions with code that can be re-run.

Statistics are paired by subject (subject-dependent protocols) or by fold
(LOSO), which is the correct unit of analysis: the 15 subjects/folds are the
independent units, and repeating a run with different seeds only reduces
initialisation noise within a unit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROPOSED = "emo_ct_cat"      # the concatenation variant
REFERENCE = "ctnet_add"      # CTNet as published


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def collect(out_dir: str) -> pd.DataFrame:
    """Read every completed run's metrics.json + config.json."""
    rows: List[Dict] = []
    for log_dir in sorted(glob.glob(os.path.join(out_dir, "logs", "*"))):
        metrics_path = os.path.join(log_dir, "metrics.json")
        config_path = os.path.join(log_dir, "config.json")
        if not (os.path.exists(metrics_path) and os.path.exists(config_path)):
            continue
        with open(metrics_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if "accuracy" not in metrics or not np.isfinite(metrics.get("accuracy", np.nan)):
            continue
        rows.append({
            "run_id": os.path.basename(log_dir),
            "model": cfg["model"],
            "protocol": cfg["protocol"],
            "window_seconds": cfg["window_seconds"],
            "seed": cfg["seed"],
            "unit": cfg["fold"] if cfg["protocol"] == "loso" else cfg["subject"],
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
            "f1": metrics.get("f1"),
            "epochs_run": metrics.get("epochs_run"),
            "seconds": metrics.get("seconds"),
        })
    if not rows:
        raise SystemExit("no completed runs found under %s/logs" % out_dir)
    return pd.DataFrame(rows)


def per_unit_table(df: pd.DataFrame, protocol: str, window: float) -> pd.DataFrame:
    """Mean over seeds, indexed by subject/fold, one column per model."""
    sub = df[(df.protocol == protocol) & (np.isclose(df.window_seconds, window))]
    if sub.empty:
        raise SystemExit("no runs for protocol=%s window=%gs" % (protocol, window))
    return sub.pivot_table(index="unit", columns="model", values="accuracy", aggfunc="mean")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (effect size for Wilcoxon)."""
    from scipy.stats import rankdata

    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    if w_plus + w_minus == 0:
        return 0.0
    return (w_plus - w_minus) / (w_plus + w_minus)


def hedges_g(diff: np.ndarray) -> float:
    """Hedges' g for paired differences (small-sample corrected)."""
    d = diff[np.isfinite(diff)]
    n = d.size
    if n < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float("nan")
    g = d.mean() / sd
    return float(g * (1 - 3.0 / (4 * n - 9)))


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        value = (n - rank) * p[i]
        running = max(running, value)
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def compare(df: pd.DataFrame, protocol: str, window: float,
            reference: str = REFERENCE, proposed: str = PROPOSED) -> pd.DataFrame:
    """Paired Wilcoxon of the proposed model against every other model."""
    from scipy.stats import wilcoxon

    table = per_unit_table(df, protocol, window)
    if proposed not in table.columns:
        raise SystemExit("model %r not present in protocol=%s window=%gs" % (proposed, protocol, window))

    base = table[proposed].to_numpy(dtype=float)
    rows = []
    for model in sorted(c for c in table.columns if c != proposed):
        other = table[model].to_numpy(dtype=float)
        diff = base - other
        finite = np.isfinite(diff)
        try:
            if finite.sum() < 2 or np.allclose(diff[finite], 0.0):
                stat, p = float("nan"), float("nan")
            else:
                stat, p = wilcoxon(base[finite], other[finite])
                stat, p = float(stat), float(p)
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append({
            "protocol": protocol,
            "window_seconds": window,
            "proposed": proposed,
            "baseline": model,
            "n_pairs": int(finite.sum()),
            "acc_proposed": float(np.nanmean(base)),
            "acc_baseline": float(np.nanmean(other)),
            "mean_diff": float(np.nanmean(diff)),
            "wilcoxon_stat": stat,
            "p_value": p,
            "rank_biserial": rank_biserial(diff),
            "hedges_g": hedges_g(diff),
        })

    frame = pd.DataFrame(rows, columns=[
        "protocol", "window_seconds", "proposed", "baseline", "n_pairs",
        "acc_proposed", "acc_baseline", "mean_diff", "wilcoxon_stat",
        "p_value", "rank_biserial", "hedges_g",
    ])
    if frame.empty:
        # only the proposed model is present (e.g. a pilot run) - nothing to compare
        frame["p_holm"] = pd.Series(dtype=float)
        frame["significant_0.05"] = pd.Series(dtype=bool)
        return frame
    frame["p_holm"] = holm(frame["p_value"].fillna(1.0).tolist())
    frame["significant_0.05"] = frame["p_holm"] < 0.05
    return frame


def describe(df: pd.DataFrame, protocol: str, window: float) -> pd.DataFrame:
    table = per_unit_table(df, protocol, window)
    rows = []
    for model in sorted(table.columns):
        vals = table[model].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "protocol": protocol,
            "window_seconds": window,
            "model": model,
            "n_units": int(np.isfinite(vals).sum()),
            "accuracy_mean": float(np.nanmean(vals)),
            "accuracy_std": float(np.nanstd(vals, ddof=1)),
            "accuracy_ci_lo": lo,
            "accuracy_ci_hi": hi,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# confusion matrices from the raw per-window predictions
# ---------------------------------------------------------------------------


def confusion_from_predictions(out_dir: str, protocol: str, window: float,
                               model: str, per_subject: bool = False):
    """Pooled (or per-subject-averaged) 3x3 confusion matrix from preds_test.csv."""
    per_unit: Dict[int, np.ndarray] = {}
    pooled = np.zeros((3, 3), dtype=np.int64)
    for log_dir in sorted(glob.glob(os.path.join(out_dir, "logs", "*"))):
        cfg_path = os.path.join(log_dir, "config.json")
        preds_path = os.path.join(log_dir, "preds_test.csv")
        if not (os.path.exists(cfg_path) and os.path.exists(preds_path)):
            continue
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg["model"] != model or cfg["protocol"] != protocol:
            continue
        if not np.isclose(cfg["window_seconds"], window):
            continue
        frame = pd.read_csv(preds_path)
        unit = cfg["fold"] if protocol == "loso" else cfg["subject"]
        cm = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(frame["true"].to_numpy(), frame["pred"].to_numpy()):
            cm[int(t), int(p)] += 1
        pooled += cm
        per_unit[int(unit)] = per_unit.get(int(unit), np.zeros((3, 3), dtype=np.int64)) + cm

    if not per_unit:
        raise SystemExit("no predictions for model=%s protocol=%s window=%gs" % (model, protocol, window))

    if per_subject:
        stack = np.stack([_row_normalise(m) for m in per_unit.values()])
        return stack.mean(axis=0), per_unit, pooled
    return _row_normalise(pooled), per_unit, pooled


def _row_normalise(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
    ap.add_argument("--protocols", default="cross_session,random_933,loso")
    ap.add_argument("--windows", default="")
    ap.add_argument("--proposed", default=PROPOSED)
    ap.add_argument("--reference", default=REFERENCE)
    args = ap.parse_args(argv)

    df = collect(args.out)
    agg_dir = os.path.join(args.out, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    df.to_csv(os.path.join(agg_dir, "all_runs.csv"), index=False)

    windows = ([float(w) for w in args.windows.split(",") if w.strip()]
               if args.windows else sorted(df.window_seconds.unique().tolist()))

    descriptions, comparisons = [], []
    for protocol in [p for p in args.protocols.split(",") if p.strip()]:
        for window in windows:
            try:
                descriptions.append(describe(df, protocol, window))
                comparisons.append(compare(df, protocol, window,
                                           reference=args.reference, proposed=args.proposed))
            except SystemExit as exc:
                print("  skipped %s @ %gs: %s" % (protocol, window, exc))

    if descriptions:
        desc = pd.concat(descriptions, ignore_index=True)
        desc.to_csv(os.path.join(agg_dir, "descriptive.csv"), index=False)
        print("\n=== descriptive (mean accuracy per model) ===")
        print(desc.to_string(index=False, float_format=lambda v: "%.4f" % v))

    if comparisons:
        comparisons = [c for c in comparisons if not c.empty]
    if comparisons:
        comp = pd.concat(comparisons, ignore_index=True)
        comp.to_csv(os.path.join(agg_dir, "stats.csv"), index=False)
        print("\n=== %s vs baselines (paired Wilcoxon) ===" % args.proposed)
        print(comp.to_string(index=False, float_format=lambda v: "%.4f" % v))

    print("\nwrote %s" % agg_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
