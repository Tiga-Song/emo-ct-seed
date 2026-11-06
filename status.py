"""Progress report for a running campaign.

Usage:
    python status.py [--out experiments] [--tag pilot]

Reads the append-only ledger and the per-run log directories.  Safe to run
while a campaign is writing: the ledger is updated with an atomic rename, so a
reader never sees a half-written row.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(here, "experiments"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--tail", type=int, default=12, help="rows to show in the ledger table")
    return ap.parse_args(argv)


def fmt_seconds(s: float) -> str:
    s = float(s)
    if s < 90:
        return "%.0f s" % s
    if s < 5400:
        return "%.1f min" % (s / 60)
    return "%.1f h" % (s / 3600)


def run_detail(log_dir: str) -> Dict:
    """What a single run directory can tell us."""
    info = {"run_id": os.path.basename(log_dir), "done": os.path.exists(os.path.join(log_dir, "DONE"))}
    metrics = os.path.join(log_dir, "metrics.json")
    if os.path.exists(metrics):
        with open(metrics, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        info.update(accuracy=m.get("accuracy"), kappa=m.get("kappa"),
                    epochs_run=m.get("epochs_run"), best_epoch=m.get("best_epoch"),
                    seconds=m.get("seconds"))
    log = os.path.join(log_dir, "train_log.csv")
    if os.path.exists(log):
        try:
            frame = pd.read_csv(log)
            info["epochs_logged"] = len(frame)
            if len(frame) and "seconds" in frame:
                info["s_per_epoch"] = float(frame["seconds"].mean())
        except Exception:
            pass
    return info


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = os.path.join(args.out, args.tag) if args.tag else args.out

    if not os.path.isdir(out_dir):
        print("no such campaign directory: %s" % out_dir)
        return 1

    print("campaign : %s" % out_dir)
    ledger_path = os.path.join(out_dir, "runs.csv")
    log_dirs = sorted(glob.glob(os.path.join(out_dir, "logs", "*")))

    finished, running = [], []
    for d in log_dirs:
        if not os.path.isdir(d):
            continue
        info = run_detail(d)
        (finished if info["done"] else running).append(info)

    if os.path.exists(ledger_path):
        ledger = pd.read_csv(ledger_path)
        counts = ledger["status"].value_counts().to_dict() if "status" in ledger else {}
        print("ledger   : %d row(s)  %s" % (len(ledger), counts))
        print("updated  : %s ago" % fmt_seconds(time.time() - os.path.getmtime(ledger_path)))
    else:
        print("ledger   : not created yet")

    print("runs     : %d finished, %d in progress" % (len(finished), len(running)))

    if running:
        print("\nin progress:")
        for info in running:
            extra = ""
            if "epochs_logged" in info:
                extra = "  %d epochs logged" % info["epochs_logged"]
                if "s_per_epoch" in info:
                    extra += " (%s/epoch)" % fmt_seconds(info["s_per_epoch"])
            print("  %-58s%s" % (info["run_id"], extra))

    if finished:
        secs = [i["seconds"] for i in finished if i.get("seconds")]
        accs = [i["accuracy"] for i in finished if i.get("accuracy") is not None]
        if secs:
            print("\nmean wall clock per finished run: %s" % fmt_seconds(sum(secs) / len(secs)))
        if accs:
            print("mean test accuracy so far       : %.4f  (n=%d)" % (sum(accs) / len(accs), len(accs)))

        print("\nlast %d finished:" % min(args.tail, len(finished)))
        for info in finished[-args.tail:]:
            print("  %-58s acc=%s  kappa=%s  ep=%s (best %s)  %s" % (
                info["run_id"],
                "%.4f" % info["accuracy"] if info.get("accuracy") is not None else "n/a",
                "%.4f" % info["kappa"] if info.get("kappa") is not None else "n/a",
                info.get("epochs_run", "?"), info.get("best_epoch", "?"),
                fmt_seconds(info["seconds"]) if info.get("seconds") else "",
            ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
