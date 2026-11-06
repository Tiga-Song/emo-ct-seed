"""Leak-free splits for the SEED experiments.

The single rule that everything here exists to enforce:

    **All windows of one trial land in exactly one of train / val / test.**

Windows cut from the same 226 s trial are near-duplicates of each other.  A
random window-level split therefore puts near-copies of the test set into the
training set and inflates accuracy by a large margin.  So every split below is
built on *trials*, and ``assert_no_trial_leak`` is run on every fold.

Two protocols are provided:

``cross_session``   sessions 1+2 train, session 3 test (the primary protocol)
``random_933``      per-class 9 train / 3 val / 3 test trials
``loso``            leave-one-subject-out, 15 folds

The split seed is deliberately decoupled from the model seed: every model sees
byte-identical data, so model-to-model comparisons are paired and the repetition
seed varies only initialisation, shuffling and augmentation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from seed_data import (
    CachedSession,
    W_FILE,
    W_LABEL,
    W_SESSION,
    W_SUBJECT,
    trial_keys,
)

SPLIT_SEED = 20241010  # fixed; independent of model seeds


# --------------------------------------------------------------------------
# trial bookkeeping
# --------------------------------------------------------------------------


def trial_table(index: np.ndarray):
    """Collapse the window index to a trial table.

    Returns
    -------
    trial_ids : (T,) int64, stable per (file_id, trial)
    inverse : (M,) int64, for each window row, its position in ``trial_ids``
    trial_labels : (T,) int64
    trial_subject : (T,) int64
    trial_session : (T,) int64
    """
    keys = trial_keys(index)
    trial_ids, inverse = np.unique(keys, return_inverse=True)
    n_trials = trial_ids.shape[0]

    labels = np.zeros(n_trials, dtype=np.int64)
    subjects = np.zeros(n_trials, dtype=np.int64)
    sessions = np.zeros(n_trials, dtype=np.int64)
    labels[inverse] = index[:, W_LABEL]
    subjects[inverse] = index[:, W_SUBJECT]
    sessions[inverse] = index[:, W_SESSION]
    return trial_ids, inverse, labels, subjects, sessions


def rows_for_trials(inverse: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Window rows whose trial position is selected by ``mask``."""
    return np.flatnonzero(mask[inverse])


def stratified_trial_masks(trial_labels: np.ndarray, *, val_frac: float,
                           test_frac: float, seed: int,
                           min_val_per_class: int = 1, min_test_per_class: int = 1,
                           allowed: Optional[np.ndarray] = None):
    """Class-stratified train/val/test masks over trials.

    ``allowed`` optionally restricts which trials may be split at all (used by
    LOSO to exclude the held-out subject before splitting).
    """
    rng = np.random.default_rng(seed)
    n = trial_labels.shape[0]
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    pool = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)

    for cls in np.unique(trial_labels[pool]):
        idx = np.flatnonzero(pool & (trial_labels == cls))
        idx = rng.permutation(idx)
        total = idx.shape[0]

        n_test = max(min_test_per_class, int(round(test_frac * total))) if test_frac > 0 else 0
        n_val = max(min_val_per_class, int(round(val_frac * total))) if val_frac > 0 else 0
        # never leave a class with no training trials
        if n_test + n_val >= total:
            n_test = min(n_test, max(0, total - 1))
            n_val = min(n_val, max(0, total - 1 - n_test))

        test[idx[:n_test]] = True
        val[idx[n_test:n_test + n_val]] = True
        train[idx[n_test + n_val:]] = True

    return train, val, test


# --------------------------------------------------------------------------
# split container
# --------------------------------------------------------------------------


@dataclass
class Split:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    subject: Optional[int] = None
    fold: Optional[int] = None
    meta: Dict = field(default_factory=dict)

    def sizes(self) -> Dict[str, int]:
        return {"train": int(self.train.size), "val": int(self.val.size), "test": int(self.test.size)}

    def to_json(self, index: np.ndarray) -> Dict:
        """Audit record: the exact trial ids behind each split."""
        keys = trial_keys(index)

        def trials(rows: np.ndarray) -> List[int]:
            return sorted(set(int(v) for v in keys[rows].tolist()))

        return {
            "name": self.name,
            "subject": self.subject,
            "fold": self.fold,
            "n_windows": self.sizes(),
            "n_trials": {
                "train": len(trials(self.train)),
                "val": len(trials(self.val)),
                "test": len(trials(self.test)),
            },
            "train_trials": trials(self.train),
            "val_trials": trials(self.val),
            "test_trials": trials(self.test),
            **self.meta,
        }


def assert_no_trial_leak(split: Split, index: np.ndarray) -> None:
    """The core safety property.  Raises AssertionError on any overlap."""
    keys = trial_keys(index)
    tr = set(keys[split.train].tolist())
    va = set(keys[split.val].tolist())
    te = set(keys[split.test].tolist())

    for name_a, a, name_b, b in (
        ("train", tr, "val", va),
        ("train", tr, "test", te),
        ("val", va, "test", te),
    ):
        overlap = a & b
        assert not overlap, (
            "%s: %s and %s share %d trial(s) - this leaks and invalidates the result"
            % (split.name, name_a, name_b, len(overlap))
        )
    assert split.train.size > 0 and split.val.size > 0 and split.test.size > 0, (
        "%s: empty split %r" % (split.name, split.sizes())
    )


# --------------------------------------------------------------------------
# protocol builders
# --------------------------------------------------------------------------


def splits_cross_session(index: np.ndarray, val_frac: float = 0.2,
                         seed: int = SPLIT_SEED) -> List[Split]:
    """Per subject: sessions 1+2 train (val carved out), session 3 test."""
    _, inverse, labels, subjects, sessions = trial_table(index)
    out: List[Split] = []
    for subject in sorted(set(subjects.tolist())):
        subj_mask = subjects == subject
        test_mask = subj_mask & (sessions == 3)
        pool = subj_mask & (sessions < 3)
        train_mask, val_mask, _ = stratified_trial_masks(
            labels, val_frac=val_frac, test_frac=0.0, seed=seed, allowed=pool
        )
        split = Split(
            name="cross_session",
            subject=int(subject),
            train=rows_for_trials(inverse, train_mask),
            val=rows_for_trials(inverse, val_mask),
            test=rows_for_trials(inverse, test_mask),
            meta={"protocol": "cross_session", "held_out": "session 3"},
        )
        assert_no_trial_leak(split, index)
        out.append(split)
    return out


def splits_random_933(index: np.ndarray, val_frac: float = 0.2,
                      seed: int = SPLIT_SEED) -> List[Split]:
    """Per subject: every trial in one pooled, class-stratified random split.

    With 45 trials (15 per class) and ``val_frac=0.2`` this yields
    9 train / 3 val / 3 test trials per class.
    """
    _, inverse, labels, subjects, _ = trial_table(index)
    out: List[Split] = []
    for subject in sorted(set(subjects.tolist())):
        pool = subjects == subject
        train_mask, val_mask, test_mask = stratified_trial_masks(
            labels, val_frac=val_frac, test_frac=val_frac, seed=seed, allowed=pool
        )
        split = Split(
            name="random_933",
            subject=int(subject),
            train=rows_for_trials(inverse, train_mask),
            val=rows_for_trials(inverse, val_mask),
            test=rows_for_trials(inverse, test_mask),
            meta={"protocol": "random_933", "held_out": "random trials"},
        )
        assert_no_trial_leak(split, index)
        out.append(split)
    return out


def splits_loso(index: np.ndarray, val_frac: float = 0.15,
                seed: int = SPLIT_SEED) -> List[Split]:
    """15 folds; fold ``s`` holds out subject ``s`` entirely."""
    _, inverse, labels, subjects, _ = trial_table(index)
    out: List[Split] = []
    for subject in sorted(set(subjects.tolist())):
        held_out = subjects == subject
        pool = ~held_out
        train_mask, val_mask, _ = stratified_trial_masks(
            labels, val_frac=val_frac, test_frac=0.0, seed=seed, allowed=pool
        )
        split = Split(
            name="loso",
            fold=int(subject),
            train=rows_for_trials(inverse, train_mask),
            val=rows_for_trials(inverse, val_mask),
            test=rows_for_trials(inverse, held_out),
            meta={"protocol": "loso", "held_out_subject": int(subject)},
        )
        assert_no_trial_leak(split, index)
        out.append(split)
    return out


PROTOCOLS = {
    "cross_session": splits_cross_session,
    "random_933": splits_random_933,
    "loso": splits_loso,
}


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------


def _selftest() -> int:
    import sys

    from seed_data import DEFAULT_CACHE_DIR, build_window_index, load_manifest

    cache_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CACHE_DIR
    sessions = load_manifest(cache_dir)

    failures = 0
    for window_len in (400, 1000, 2000, 6000):
        index = build_window_index(sessions, window_len)
        print("W=%5d  %d windows" % (window_len, index.shape[0]))
        for name, builder in PROTOCOLS.items():
            try:
                splits = builder(index)
            except Exception as exc:
                print("   %-14s FAILED to build: %s" % (name, exc))
                failures += 1
                continue
            assert len(splits) == 15, "expected 15 splits, got %d" % len(splits)
            tot = 0
            for s in splits:  # assert_no_trial_leak already ran inside the builder
                tot += s.train.size + s.val.size + s.test.size
            lens = [len(splits)]
            print("   %-14s %2d splits, no leak, %6d window-slots" % (name, lens[0], tot))
    print()
    print("selftest:", "PASS" if failures == 0 else "FAIL (%d)" % failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
