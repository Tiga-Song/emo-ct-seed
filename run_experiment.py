"""Resumable experiment matrix driver.

Enumerates every (model x protocol x window x seed x subject/fold) run, writes
each one's artifacts under ``experiments/logs/<run_id>/``, and keeps a single
``experiments/runs.csv`` ledger.  A run is skipped when its ``DONE`` sentinel
exists, so a multi-day campaign can be interrupted and restarted safely on a
shared machine.

Examples
--------
Stage 1, window-length pilot (3 subjects, one model)::

    python run_experiment.py --protocols cross_session \\
        --models emo_ct_cat --windows 2,5,10,30 --seeds 0 \\
        --subjects 1,2,3 --epochs 60 --patience 12 --tag pilot

Stage 2, subject-dependent main runs::

    python run_experiment.py --protocols cross_session,random_933 \\
        --models ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer \\
        --windows 10 --seeds 0,1,2 --epochs 300 --patience 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models import ABLATION_MODELS, ALL_MODELS, MAIN_MODELS
from runner import RunConfig, run_single
from seed_data import (
    DEFAULT_CACHE_DIR,
    SEED_FS,
    build_window_index,
    compute_channel_stats,
    load_manifest,
    open_memmaps,
)
from splits import PROTOCOLS, Split

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "experiments")

# CTNet used dropout 0.5 for subject-specific training and 0.25 for the larger
# cross-subject training sets; keep that convention.
DROPOUT_BY_PROTOCOL = {"cross_session": 0.5, "random_933": 0.5, "loso": 0.25}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_manifest(out_dir: str) -> str:
    """Hash the source tree so a result can be tied to exact code.

    This machine has no git, so the hash is the only provenance anchor.
    """
    lines = []
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(HERE, name)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        lines.append("%s  %s" % (digest, name))
    text = "\n".join(lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "MANIFEST.sha256"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_csv_ints(value: str) -> List[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def _window_index_cache(sessions, cache: Dict[int, np.ndarray]):
    def get(window_len: int) -> np.ndarray:
        if window_len not in cache:
            cache[window_len] = build_window_index(sessions, window_len)
        return cache[window_len]
    return get


def enumerate_runs(args) -> List[RunConfig]:
    runs: List[RunConfig] = []
    windows = [float(w) for w in args.windows.split(",") if w.strip()]
    models = [m for m in args.models.split(",") if m.strip()]
    seeds = parse_csv_ints(args.seeds)

    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit("unknown model(s): %s\nknown: %s" % (", ".join(unknown), ", ".join(ALL_MODELS)))

    for protocol in [p for p in args.protocols.split(",") if p.strip()]:
        if protocol not in PROTOCOLS:
            raise SystemExit("unknown protocol %r (known: %s)" % (protocol, ", ".join(PROTOCOLS)))
        if protocol == "loso":
            targets = parse_csv_ints(args.folds) if args.folds else list(range(1, 16))
            target_key = "fold"
        else:
            targets = parse_csv_ints(args.subjects) if args.subjects else list(range(1, 16))
            target_key = "subject"

        for window in windows:
            for model in models:
                for target in targets:
                    for seed in seeds:
                        kwargs = dict(
                            model=model, protocol=protocol, window_seconds=window,
                            seed=seed, epochs=args.epochs, patience=args.patience,
                            batch_size=args.batch_size, n_aug=args.n_aug,
                            dropout_rate=args.dropout if args.dropout is not None
                            else DROPOUT_BY_PROTOCOL[protocol],
                            head_dropout=args.head_dropout,
                            use_amp=args.amp,
                        )
                        kwargs[target_key] = target
                        runs.append(RunConfig(**kwargs))
    if args.limit:
        runs = runs[: args.limit]
    return runs


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def ledger_path(out_dir: str) -> str:
    return os.path.join(out_dir, "runs.csv")


def load_ledger(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_ledger(path: str, row: Dict) -> None:
    """Record one run, atomically, replacing any earlier row for the same run.

    Replacing rather than appending matters: a retry after a failure must not
    leave a stale ``failed`` row behind.  The write goes to a temp file and is
    renamed, so a killed process never leaves a half-written row.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            if "run_id" in existing.columns:
                existing = existing[existing["run_id"] != row["run_id"]]
        except Exception:
            existing = pd.DataFrame()
        frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--protocols", default="cross_session")
    ap.add_argument("--models", default=",".join(MAIN_MODELS))
    ap.add_argument("--windows", default="10", help="window length(s) in seconds")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--subjects", default="", help="subject-dependent targets, default all 15")
    ap.add_argument("--folds", default="", help="LOSO folds, default all 15")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-aug", type=int, default=3, help="S&R augmentation multiplier")
    ap.add_argument("--dropout", type=float, default=None,
                    help="default: 0.5 subject-dependent, 0.25 LOSO")
    ap.add_argument("--head-dropout", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true",
                    help="enable bf16 autocast (measured slower on this GPU; off by default)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N (smoke tests)")
    ap.add_argument("--list", action="store_true", help="enumerate and exit")
    ap.add_argument("--tag", default="", help="suffix for the experiments sub-directory")
    args = ap.parse_args(argv)

    out_dir = os.path.join(args.out, args.tag) if args.tag else args.out
    os.makedirs(out_dir, exist_ok=True)

    runs = enumerate_runs(args)
    print("enumerated %d run(s)" % len(runs))
    if args.list:
        for cfg in runs:
            print("  ", cfg.run_id())
        return 0
    if not runs:
        print("nothing to do")
        return 0

    manifest = write_manifest(out_dir)
    print("source manifest: %s" % manifest)

    sessions = load_manifest(args.cache)
    mmaps = open_memmaps(sessions, args.cache)
    window_index = _window_index_cache(sessions, {})
    # keyed by (protocol, window_len): the row ids in a split only mean anything
    # against the window index they were built from
    splits_cache: Dict[tuple, List[Split]] = {}

    path = ledger_path(out_dir)
    ledger = load_ledger(path)
    done_ids = set(ledger.loc[ledger.get("status", pd.Series(dtype=str)) == "done", "run_id"].tolist()) \
        if not ledger.empty and "run_id" in ledger.columns else set()

    started = time.time()
    completed = 0
    failed = 0
    skipped = 0

    for n, cfg in enumerate(runs, start=1):
        run_id = cfg.run_id()
        run_dir = os.path.join(out_dir, "logs", run_id)
        if run_id in done_ids or os.path.exists(os.path.join(run_dir, "DONE")):
            skipped += 1
            continue

        print("\n[%d/%d] %s" % (n, len(runs), run_id))
        t0 = time.time()

        index = window_index(cfg.window_len)
        cache_key = (cfg.protocol, cfg.window_len)
        if cache_key not in splits_cache:
            splits_cache[cache_key] = PROTOCOLS[cfg.protocol](index)
        target = cfg.subject if cfg.protocol != "loso" else cfg.fold
        match = [s for s in splits_cache[cache_key]
                 if (s.subject if cfg.protocol != "loso" else s.fold) == target]
        if not match:
            print("   no split for target %r - skipped" % target)
            continue
        split = match[0]

        mean, std = compute_channel_stats(index, cfg.window_len, mmaps, rows=split.train)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "split.json"), "w", encoding="utf-8") as fh:
            json.dump({**split.to_json(index), "manifest": manifest}, fh, indent=1)

        try:
            metrics = run_single(cfg, index, split.train, split.val, split.test,
                                 mmaps, mean, std, run_dir)
            status = "done"
        except Exception as exc:               # keep the campaign alive
            import traceback
            traceback.print_exc()
            metrics = {"accuracy": float("nan"), "kappa": float("nan"),
                       "error": "%s: %s" % (type(exc).__name__, exc)}
            status = "failed"
            with open(os.path.join(run_dir, "ERROR.txt"), "w", encoding="utf-8") as fh:
                fh.write(str(exc))

        append_ledger(path, {
            "run_id": run_id, "model": cfg.model, "protocol": cfg.protocol,
            "window_seconds": cfg.window_seconds, "seed": cfg.seed,
            "subject": cfg.subject, "fold": cfg.fold, "status": status,
            "accuracy": metrics.get("accuracy"), "kappa": metrics.get("kappa"),
            "f1": metrics.get("f1"), "epochs_run": metrics.get("epochs_run"),
            "best_epoch": metrics.get("best_epoch"),
            "seconds": round(time.time() - t0, 1),
            "n_train": metrics.get("n_train"), "n_test": metrics.get("n_test"),
            "manifest": manifest,
        })
        if status == "done":
            completed += 1
        else:
            failed += 1

    elapsed = time.time() - started
    print("\n%d completed, %d failed, %d skipped, %.1f min elapsed"
          % (completed, failed, skipped, elapsed / 60))
    print("ledger: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
