"""SEED data access for the Emo-CT / SEED experiments.

Handles three things:

1.  Discovery and one-time caching of the 45 SEED session files
    (`Preprocessed_EEG/{subject}_{date}.mat`, 62 channels @ 200 Hz, 15 trials each).
2.  Windowing: non-overlapping windows of a fixed length, with a full
    provenance index so that no window can be traced back ambiguously.
3.  A torch ``Dataset`` that slices windows straight out of memory-mapped
    float32 arrays.

Why a cache: the SEED ``.mat`` files are MATLAB v5 *compressed*, so every read
decompresses ~160 MB of float64.  Doing that inside a training loop is fatal.
We pay it once and write uncompressed float32 ``.npy`` files, which are
memory-mappable and slice with zero copy.

Data provenance note: the SEED authors already band-pass filtered and
downsampled these recordings to 200 Hz.  We perform **no** additional filtering
and no artifact removal, and the manuscript must not claim otherwise.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Dataset constants, verified against data/SEED/SEED_EEG/Preprocessed_EEG/
# --------------------------------------------------------------------------

SEED_FS = 200                  # Hz, after the dataset authors' downsampling
SEED_N_CHANNELS = 62
SEED_N_CLASSES = 3
SEED_TRIALS_PER_SESSION = 15
SEED_N_SESSIONS = 3

# Fixed stimulus order, documented in Preprocessed_EEG/readme.txt.
#   1 = positive, 0 = neutral, -1 = negative
# The k-th variable in a session file (`*_eeg{k}`) carries the k-th label.
SEED_LABEL_ORDER: Tuple[int, ...] = (1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1)

# Every session contains the same 15 film clips, so trial lengths are identical
# across subjects.  Measured from the data; used as an integrity check only -
# the loader always reads the real lengths.
EXPECTED_TRIAL_LENGTHS: Tuple[int, ...] = (
    47001, 46601, 41201, 47601, 37001, 39001, 47401, 43201, 53001,
    47401, 47001, 46601, 47001, 47601, 41201,
)

_TRIAL_RE = re.compile(r"^(.+)_eeg(\d+)$")
_SESSION_RE = re.compile(r"^(?P<subject>\d+)_(?P<date>\d{8})$")

# default location of the raw data and of the cache
DEFAULT_SEED_ROOT = r"d:\syl\brain science2\data\SEED\SEED_EEG\Preprocessed_EEG"
DEFAULT_CACHE_DIR = r"C:\Users\Administer\seed_cache"


def seed_to_zero_based(raw_label: int) -> int:
    """SEED raw labels {-1, 0, 1} -> {0, 1, 2}.

    Note this is ``+1``, deliberately *not* CTNet's ``-1`` (which shifts the
    1-based BCI IV-2a/2b labels).  Mixing the two silently mislabels every
    trial, so keep it in one place.
    """
    if raw_label not in (-1, 0, 1):
        raise ValueError("unexpected SEED label %r" % (raw_label,))
    return raw_label + 1


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileMeta:
    path: str
    subject: int          # 1..15
    session: int          # 1..3, ordered chronologically within subject
    date: str             # YYYYMMDD
    stem: str             # e.g. "1_20131027"

    @property
    def key(self) -> str:
        return "s%02d_%s" % (self.subject, self.date)


def discover_files(root: str = DEFAULT_SEED_ROOT) -> List[FileMeta]:
    """Return the 45 session files, with a chronological session index."""
    paths = sorted(glob.glob(os.path.join(root, "*.mat")))
    parsed = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = _SESSION_RE.match(stem)
        if not m:
            continue  # label.mat and friends
        parsed.append((int(m.group("subject")), m.group("date"), stem, p))

    metas: List[FileMeta] = []
    by_subject: Dict[int, List[Tuple[str, str, str]]] = {}
    for subject, date, stem, p in parsed:
        by_subject.setdefault(subject, []).append((date, stem, p))

    for subject in sorted(by_subject):
        entries = sorted(by_subject[subject])  # chronological
        for idx, (date, stem, p) in enumerate(entries, start=1):
            metas.append(FileMeta(path=p, subject=subject, session=idx, date=date, stem=stem))
    return metas


def read_label_file(root: str = DEFAULT_SEED_ROOT) -> np.ndarray:
    """Read ``label.mat`` and check it against the documented stimulus order."""
    import scipy.io

    path = os.path.join(root, "label.mat")
    if not os.path.exists(path):
        return np.asarray(SEED_LABEL_ORDER, dtype=np.int64)
    mat = scipy.io.loadmat(path)
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value).ravel()
        if arr.size == SEED_TRIALS_PER_SESSION:
            arr = arr.astype(np.int64)
            if tuple(arr.tolist()) != SEED_LABEL_ORDER:
                raise ValueError(
                    "label.mat order %r disagrees with readme order %r"
                    % (tuple(arr.tolist()), SEED_LABEL_ORDER)
                )
            return arr
    return np.asarray(SEED_LABEL_ORDER, dtype=np.int64)


# --------------------------------------------------------------------------
# One-time cache build
# --------------------------------------------------------------------------


def load_session_mat(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one session file.

    Returns
    -------
    data : (62, N_total) float32, the 15 trials concatenated along time
    offsets : (15, 2) int64, [start, stop) of each trial inside ``data``
    raw_labels : (15,) int64, SEED raw labels in {-1, 0, 1}
    """
    import scipy.io

    mat = scipy.io.loadmat(path)
    trials = []
    for key in mat:
        m = _TRIAL_RE.match(key)
        if m:
            trials.append((int(m.group(2)), key))
    if len(trials) != SEED_TRIALS_PER_SESSION:
        raise ValueError(
            "%s: found %d trial variables, expected %d"
            % (path, len(trials), SEED_TRIALS_PER_SESSION)
        )
    trials.sort()  # by integer k - lexicographic sort would put eeg10 before eeg2

    blocks = []
    offsets = np.zeros((len(trials), 2), dtype=np.int64)
    cursor = 0
    for i, (k, key) in enumerate(trials):
        arr = np.asarray(mat[key], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != SEED_N_CHANNELS:
            raise ValueError("%s: variable %s has shape %r, expected (62, N)" % (path, key, arr.shape))
        blocks.append(arr)
        offsets[i] = (cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]

    data = np.concatenate(blocks, axis=1)
    labels = np.asarray(SEED_LABEL_ORDER, dtype=np.int64)
    return data, offsets, labels


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_cache(root: str = DEFAULT_SEED_ROOT, cache_dir: str = DEFAULT_CACHE_DIR,
                verbose: bool = True) -> List[Dict]:
    """Convert every session file to float32 .npy + .json.  Idempotent."""
    raw_dir = os.path.join(cache_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    metas = discover_files(root)
    if not metas:
        raise FileNotFoundError("no session files found under %s" % root)

    manifest = []
    for meta in metas:
        npy_path = os.path.join(raw_dir, meta.key + ".npy")
        json_path = os.path.join(raw_dir, meta.key + ".json")
        if not (os.path.exists(npy_path) and os.path.exists(json_path)):
            data, offsets, labels = load_session_mat(meta.path)
            np.save(npy_path, data)
            record = {
                "key": meta.key,
                "subject": meta.subject,
                "session": meta.session,
                "date": meta.date,
                "source": os.path.basename(meta.path),
                "source_sha256": _sha256(meta.path),
                "fs": SEED_FS,
                "n_channels": SEED_N_CHANNELS,
                "shape": list(data.shape),
                "offsets": offsets.tolist(),
                "raw_labels": labels.tolist(),
                "labels": [seed_to_zero_based(int(v)) for v in labels],
            }
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1)
            if verbose:
                print("cached %s  %s" % (meta.key, data.shape))
            del data
        with open(json_path, "r", encoding="utf-8") as fh:
            manifest.append(json.load(fh))

    if verbose:
        print("cache ready: %d sessions in %s" % (len(manifest), raw_dir))
    return manifest


# --------------------------------------------------------------------------
# Reading the cache
# --------------------------------------------------------------------------


@dataclass
class CachedSession:
    key: str
    subject: int
    session: int
    date: str
    offsets: np.ndarray       # (15, 2) int64
    labels: np.ndarray        # (15,) int64, zero-based {0,1,2}

    @property
    def n_trials(self) -> int:
        return int(self.offsets.shape[0])

    @property
    def trial_lengths(self) -> np.ndarray:
        return self.offsets[:, 1] - self.offsets[:, 0]


def load_manifest(cache_dir: str = DEFAULT_CACHE_DIR) -> List[CachedSession]:
    raw_dir = os.path.join(cache_dir, "raw")
    sessions = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        sessions.append(
            CachedSession(
                key=rec["key"],
                subject=rec["subject"],
                session=rec["session"],
                date=rec["date"],
                offsets=np.asarray(rec["offsets"], dtype=np.int64),
                labels=np.asarray(rec["labels"], dtype=np.int64),
            )
        )
    if not sessions:
        raise FileNotFoundError("cache is empty - run build_cache() first")
    return sessions


def open_memmaps(sessions: Sequence[CachedSession], cache_dir: str = DEFAULT_CACHE_DIR):
    """Map integer ``file_id`` -> memory-mapped ``(62, N)`` float32 array.

    The key is the session's position in ``sessions``, which is exactly what
    ``build_window_index`` writes into the ``file_id`` column.  Keying by the
    session name instead would silently mismatch.
    """
    raw_dir = os.path.join(cache_dir, "raw")
    return {
        i: np.load(os.path.join(raw_dir, s.key + ".npy"), mmap_mode="r")
        for i, s in enumerate(sessions)
    }


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

# column layout of the window index matrix
W_SUBJECT, W_SESSION, W_TRIAL, W_LABEL, W_FILE, W_START = range(6)
WINDOW_INDEX_COLUMNS = ("subject", "session", "trial", "label", "file_id", "start")


def build_window_index(sessions: Sequence[CachedSession], window_len: int) -> np.ndarray:
    """Enumerate non-overlapping windows of ``window_len`` samples.

    The remainder of each trial is discarded.  Returns an int32 array of shape
    (M, 6) with columns (subject, session, trial, label, file_id, start), where
    ``file_id`` indexes into ``sessions`` and ``start`` is the offset *within
    that session's concatenated array*.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    rows = []
    for file_id, s in enumerate(sessions):
        for trial in range(s.n_trials):
            begin, end = int(s.offsets[trial][0]), int(s.offsets[trial][1])
            n = (end - begin) // window_len
            for j in range(n):
                rows.append(
                    (s.subject, s.session, trial + 1, int(s.labels[trial]), file_id,
                     begin + j * window_len)
                )
    if not rows:
        raise ValueError("window_len=%d yields no windows" % window_len)
    return np.asarray(rows, dtype=np.int32)


def trial_keys(index: np.ndarray) -> np.ndarray:
    """Stable identifier per trial, for leak checks and grouping."""
    return (index[:, W_FILE].astype(np.int64) * 1000 + index[:, W_TRIAL].astype(np.int64))


# --------------------------------------------------------------------------
# torch Dataset
# --------------------------------------------------------------------------


class WindowDataset:
    """Slices windows out of memory-mapped sessions.

    Kept framework-light on purpose: ``__getitem__`` returns a
    ``(1, 62, window_len)`` float32 tensor and an int64 label.  Per-channel
    normalisation uses statistics computed on training windows only.
    """

    def __init__(self, index: np.ndarray, window_len: int, mmaps, mean, std,
                 augment: bool = False, n_seg: int = 8, n_aug: int = 0, rng=None):
        import torch  # local import so the module can be inspected without torch

        self._torch = torch
        self.index = np.asarray(index, dtype=np.int64)
        self.window_len = int(window_len)
        self.mmaps = mmaps
        self.augment = bool(augment)
        self.n_seg = int(n_seg)
        self.n_aug = int(n_aug)
        self.rng = rng

        # (62,) float32 -> (62, 1, 1) for broadcasting against (1, 62, W)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return self.index.shape[0]

    def _take(self, row: np.ndarray) -> np.ndarray:
        mm = self.mmaps[int(row[W_FILE])]
        start = int(row[W_START])
        block = np.asarray(mm[:, start:start + self.window_len], dtype=np.float32)
        return (block - self.mean) / self.std

    def __getitem__(self, i: int):
        row = self.index[i]
        block = self._take(row)
        x = self._torch.from_numpy(block).unsqueeze(0)  # (1, 62, W)
        return x, int(row[W_LABEL])

    # -- augmentation is applied at batch level; see runner.py -------------
    def raw_rows(self, i: int) -> np.ndarray:
        return self.index[i]


def compute_channel_stats(index: np.ndarray, window_len: int, mmaps,
                          rows: Optional[np.ndarray] = None,
                          max_windows: int = 4000,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a sample of the given windows.

    ``index`` is the full (M, 6) window matrix; ``rows`` selects which windows to
    use (default: all of them).  Pass the *training* rows only - the same
    statistics are then applied to validation and test data, which is what keeps
    the normalisation leak-free.
    """
    if rows is None:
        rows = np.arange(index.shape[0])
    rows = np.asarray(rows)
    if rows.shape[0] == 0:
        raise ValueError("cannot compute statistics from an empty set of windows")

    rng = np.random.default_rng(seed)
    sel = rows[rng.choice(rows.shape[0], size=min(max_windows, rows.shape[0]), replace=False)]

    acc = np.zeros((SEED_N_CHANNELS,), dtype=np.float64)
    acc_sq = np.zeros((SEED_N_CHANNELS,), dtype=np.float64)
    total = 0
    for i in sel:
        row = index[i]
        mm = mmaps[int(row[W_FILE])]
        start = int(row[W_START])
        block = np.asarray(mm[:, start:start + window_len], dtype=np.float64)
        acc += block.sum(axis=1)
        acc_sq += (block ** 2).sum(axis=1)
        total += block.shape[1]
    mean = acc / total
    var = np.maximum(acc_sq / total - mean ** 2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# --------------------------------------------------------------------------
# Integrity checks
# --------------------------------------------------------------------------


def verify_data(sessions: Sequence[CachedSession], root: str = DEFAULT_SEED_ROOT) -> None:
    """Assert the cache matches the documented SEED structure."""
    expected = np.asarray(EXPECTED_TRIAL_LENGTHS, dtype=np.int64)
    assert len(sessions) == 15 * SEED_N_SESSIONS, "expected 45 sessions, got %d" % len(sessions)

    by_subject: Dict[int, int] = {}
    for s in sessions:
        by_subject[s.subject] = by_subject.get(s.subject, 0) + 1
        assert s.n_trials == SEED_TRIALS_PER_SESSION, "%s: %d trials" % (s.key, s.n_trials)
        got = s.trial_lengths
        if not np.array_equal(got, expected):
            raise AssertionError(
                "%s: trial lengths differ from the documented vector\n got %s\n exp %s"
                % (s.key, got.tolist(), expected.tolist())
            )
    assert len(by_subject) == 15 and all(v == SEED_N_SESSIONS for v in by_subject.values()), (
        "expected 15 subjects x 3 sessions, got %r" % by_subject
    )

    documented = read_label_file(root)
    for s in sessions:
        raw = s.labels - 1
        assert np.array_equal(raw, documented), "%s: labels do not match label.mat" % s.key


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _main(argv: Sequence[str]) -> int:
    root = argv[1] if len(argv) > 1 else DEFAULT_SEED_ROOT
    cache_dir = argv[2] if len(argv) > 2 else DEFAULT_CACHE_DIR

    print("source root:", root)
    print("cache dir  :", cache_dir)
    build_cache(root, cache_dir)

    sessions = load_manifest(cache_dir)
    verify_data(sessions, root)
    print("verify_data: ok (45 sessions, 675 trials, labels match)")

    for window_len in (400, 1000, 2000, 6000):
        index = build_window_index(sessions, window_len)
        n_trials = len(set(trial_keys(index).tolist()))
        print("  W=%5d (%4.1f s): %7d windows over %3d trials"
              % (window_len, window_len / SEED_FS, index.shape[0], n_trials))
        if n_trials != 675:
            print("    WARNING: expected 675 distinct trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
