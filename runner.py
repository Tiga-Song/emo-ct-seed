"""Training and evaluation harness.

One runner is used by every model, so that any difference in the results comes
from the architecture and not from the training code.  It is deliberately
faithful to CTNet where it can be:

* Adam with betas (0.5, 0.999) and lr 1e-3, cross-entropy loss;
* S&R ("segmentation and reconstruction") augmentation, generalised from
  CTNet's hardcoded 1000 samples to an arbitrary window length;
* best model chosen on validation accuracy (see the note in ``run_single``).

And deliberately different where faithfulness would be wrong:

* a real held-out validation set split by trial, not CTNet's per-mini-batch
  carve-out (which re-samples every epoch and overlaps the training data);
* early stopping (cap + patience) instead of a fixed 1000 epochs;
* bf16 autocast for the Blackwell GPU;
* ``state_dict`` checkpoints plus a config snapshot, not pickled model objects.
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from models import build_model, forward_logits
from seed_data import (
    SEED_FS,
    SEED_N_CHANNELS,
    W_FILE,
    W_LABEL,
    W_SESSION,
    W_START,
    W_SUBJECT,
    W_TRIAL,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    model: str
    protocol: str                      # cross_session | random_933 | loso
    window_seconds: float = 10.0
    seed: int = 0
    subject: Optional[int] = None      # cross_session / random_933
    fold: Optional[int] = None         # loso
    epochs: int = 300
    patience: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    beta1: float = 0.5
    beta2: float = 0.999
    n_aug: int = 3                     # S&R augmentation multiplier
    n_seg: int = 8                     # S&R segments per window
    dropout_rate: float = 0.5
    head_dropout: float = 0.5
    # Measured on this RTX 5060 / torch 2.14.1+cu130 build: bf16 autocast is
    # ~20% *slower* than fp32 for these shapes (204 ms vs 165 ms per batch-128
    # step), because the ops are small and the cast overhead dominates.  Peak
    # VRAM is under 1 GiB either way, so fp32 costs nothing here.
    use_amp: bool = False
    split_seed: int = 20241010
    extra: Dict = field(default_factory=dict)

    @property
    def window_len(self) -> int:
        return int(round(self.window_seconds * SEED_FS))

    @property
    def target(self) -> str:
        if self.protocol == "loso":
            return "subject%d" % self.fold
        return "subject%d" % self.subject

    def run_id(self) -> str:
        return "%s__%s__W%s__%s__seed%d" % (
            self.model, self.protocol, ("%g" % self.window_seconds).replace(".", "p"),
            self.target, self.seed,
        )

    def to_json(self) -> Dict:
        d = asdict(self)
        d["window_len"] = self.window_len
        d["run_id"] = self.run_id()
        return d


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _alloc_with_retry(shape, dtype, attempts: int = 6, delay: float = 2.0):
    """Allocate an array, waiting through transient host-memory pressure.

    This machine has 15.8 GiB total and the desktop normally holds most of it.
    A campaign run has already died with ``Unable to allocate 7.57 MiB`` - not
    because the array is large, but because the process was momentarily at zero
    free RAM.  Windows reclaims the memmap page cache under pressure, so a short
    backoff normally lets the allocation succeed instead of losing the run.
    """
    last = None
    for attempt in range(attempts):
        try:
            return np.empty(shape, dtype=dtype)
        except MemoryError as exc:
            last = exc
            gc.collect()
            time.sleep(delay * (attempt + 1))
    raise last


# ---------------------------------------------------------------------------
# batch construction straight from the memory maps
# ---------------------------------------------------------------------------


class BatchSource:
    """Gathers ``(B, 1, 62, W)`` tensors from the memmapped sessions.

    Rows are grouped by session so each group is a single fancy-index gather
    rather than a Python loop over samples.
    """

    def __init__(self, index: np.ndarray, window_len: int, mmaps, mean, std):
        self.index = index
        self.window_len = int(window_len)
        self.mmaps = mmaps
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1)

    def gather_rows(self, rows: np.ndarray) -> np.ndarray:
        """Return ``(B, 62, W)`` float32, normalised."""
        rows = np.asarray(rows)
        out = _alloc_with_retry((rows.shape[0], SEED_N_CHANNELS, self.window_len), np.float32)
        cols = np.arange(self.window_len)
        files = self.index[rows, W_FILE]
        # stored as (62, 1); reshape to (1, 62, 1) to broadcast against (B, 62, W)
        mean = self.mean.reshape(1, -1, 1)
        std = self.std.reshape(1, -1, 1)
        for file_id in np.unique(files):
            sel = np.flatnonzero(files == file_id)
            starts = self.index[rows[sel], W_START].astype(np.int64)
            # fancy-indexing the memmap gives (62, B, W); move channels to axis 1
            block = np.asarray(
                self.mmaps[int(file_id)][:, starts[:, None] + cols[None, :]],
                dtype=np.float32,
            ).transpose(1, 0, 2)
            out[sel] = (block - mean) / std
        return out

    def batch(self, rows: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.gather_rows(rows)
        x = torch.from_numpy(data).unsqueeze(1)          # (B, 1, 62, W)
        y = torch.from_numpy(self.index[rows, W_LABEL].astype(np.int64))
        return x, y

    def augmentation_pool(self, rows: np.ndarray) -> Dict[int, np.ndarray]:
        """Row ids grouped by class, **restricted to ``rows``**.

        ``rows`` must be the training rows only.  Grouping the whole index
        instead would let S&R draw synthetic training segments out of the
        validation set or the held-out test subject - precisely the leak the
        trial-grouped splits exist to prevent.
        """
        rows = np.asarray(rows)
        labels = self.index[rows, W_LABEL]
        return {int(c): rows[np.flatnonzero(labels == c)] for c in np.unique(labels)}


def sr_augment(source: BatchSource, pool: Dict[int, np.ndarray], n_classes: int,
               batch_rows: np.ndarray, n_aug: int, n_seg: int,
               rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Segmentation & reconstruction, generalised from CTNet's ``interaug``.

    CTNet splits each 1000-sample trial into ``number_seg`` chunks and rebuilds
    synthetic trials by drawing chunk *j* from a randomly chosen same-class
    trial.  Here the unit is a window, and the source is the full training pool,
    drawn lazily from the memmap so memory stays flat.

    Produces ``n_aug * len(batch_rows)`` synthetic windows, class balanced.

    Performance note: only the *segment* actually needed is read from the
    memmap, and every source read for the whole batch is issued as one
    vectorised fancy-index per session.  A per-segment Python loop here made
    each epoch ~40x slower than the forward/backward pass it feeds.
    """
    W = source.window_len
    seg_len = W // n_seg
    if seg_len <= 0:
        raise ValueError("n_seg=%d is too large for a window of %d samples" % (n_seg, W))

    # The segments must tile the window exactly.  CTNet's 1000-sample trial
    # divided by 8 gives clean 125-sample chunks; a general window length does
    # not, so any remainder goes into the last segment.  (Reshaping to (B,62,W)
    # while the segments only cover n_seg*(W//n_seg) samples is what broke
    # W=2.5 s: 8*62 = 496 != 500.)
    lengths = [seg_len] * n_seg
    lengths[-1] = W - seg_len * (n_seg - 1)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    n_per_class = max(1, len(batch_rows) // max(n_classes, 1))
    per_class = n_aug * n_per_class
    total = per_class * n_classes

    picks = np.empty((total, n_seg), dtype=np.int64)
    labels = np.empty((total,), dtype=np.int64)
    write = 0
    for cls in range(n_classes):
        choices = pool.get(cls)
        if choices is None or choices.size == 0:
            continue
        # per_class * n_seg segments are drawn from this class, so that - not
        # n_seg alone - is the quantity that decides whether sampling must be
        # with replacement
        replace = (per_class * n_seg) > choices.size
        picks[write:write + per_class] = rng.choice(
            choices, size=(per_class, n_seg), replace=replace)
        labels[write:write + per_class] = cls
        write += per_class
    if write == 0:
        return (torch.zeros(0, 1, SEED_N_CHANNELS, W),
                torch.zeros(0, dtype=torch.long))

    picks, labels = picks[:write], labels[:write]

    starts = source.index[picks, W_START].astype(np.int64)   # (write, n_seg)
    files = source.index[picks, W_FILE]                      # (write, n_seg)
    mean = source.mean.reshape(1, -1, 1)
    std = source.std.reshape(1, -1, 1)

    out = _alloc_with_retry((write, SEED_N_CHANNELS, W), np.float32)
    for seg in range(n_seg):
        width = int(lengths[seg])
        cols = np.arange(width)
        seg_starts = starts[:, seg] + int(offsets[seg])
        seg_files = files[:, seg]
        block = _alloc_with_retry((write, SEED_N_CHANNELS, width), np.float32)
        # one vectorised gather per session touched by this segment
        for file_id in np.unique(seg_files):
            sel = np.flatnonzero(seg_files == file_id)
            gathered = source.mmaps[int(file_id)][:, seg_starts[sel][:, None] + cols[None, :]]
            block[sel] = np.asarray(gathered, dtype=np.float32).transpose(1, 0, 2)
        out[:, :, offsets[seg]:offsets[seg + 1]] = (block - mean) / std

    perm = rng.permutation(write)
    return torch.from_numpy(out[perm]).unsqueeze(1), torch.from_numpy(labels[perm])


# ---------------------------------------------------------------------------
# evaluation / metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(model: nn.Module, source: BatchSource, rows: np.ndarray, device,
            batch_size: int = 128, use_amp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, labels)`` for the given rows."""
    model.eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for start in range(0, rows.shape[0], batch_size):
        chunk = rows[start:start + batch_size]
        x, y = source.batch(chunk)
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
            logits = forward_logits(model, x)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> Dict:
    """Accuracy, kappa and macro precision/recall/F1.

    Macro averaging is chosen from the number of *observed* classes.  (The
    upstream ``calMetrics`` used ``average='binary'`` whenever ``max(y)==2``,
    which silently treats a 3-class problem as binary.)
    """
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 precision_score, recall_score)

    present = np.unique(np.concatenate([y_true, y_pred]))
    average = "binary" if len(present) == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "average_mode": average,
        "n_test": int(y_true.shape[0]),
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run_single(cfg: RunConfig, index: np.ndarray, train_rows: np.ndarray,
               val_rows: np.ndarray, test_rows: np.ndarray, mmaps,
               mean: np.ndarray, std: np.ndarray, out_dir: str,
               device: Optional[torch.device] = None, verbose: bool = True) -> Dict:
    """Train one model on one split, evaluate, and persist every artifact."""
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg.to_json(), fh, indent=1)

    model = build_model(
        cfg.model, window_len=cfg.window_len, dropout_rate=cfg.dropout_rate,
        head_dropout=cfg.head_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 betas=(cfg.beta1, cfg.beta2))
    criterion = nn.CrossEntropyLoss()

    train_src = BatchSource(index, cfg.window_len, mmaps, mean, std)
    eval_src = BatchSource(index, cfg.window_len, mmaps, mean, std)
    pool = train_src.augmentation_pool(train_rows)
    rng = np.random.default_rng(cfg.seed)

    # Model selection is on validation ACCURACY, not validation loss.
    # Measured on this dataset: val loss is minimised at epoch 1 while val
    # accuracy is still climbing, so a loss criterion would freeze the model
    # essentially untrained.  This is a documented deviation from CTNet, which
    # selects on loss; val loss is still recorded for every epoch.
    best_loss = float("inf")
    best_acc = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(train_rows.shape[0])
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        t_epoch = time.time()

        for start in range(0, order.shape[0], cfg.batch_size):
            rows = train_rows[order[start:start + cfg.batch_size]]
            if rows.shape[0] == 0:
                continue
            x, y = train_src.batch(rows)
            if cfg.n_aug > 0:
                ax, ay = sr_augment(train_src, pool, 3, rows, cfg.n_aug, cfg.n_seg, rng)
                if ax.shape[0] > 0:
                    x = torch.cat([x, ax], dim=0)
                    y = torch.cat([y, ay], dim=0)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=cfg.use_amp and device.type == "cuda"):
                logits = forward_logits(model, x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach()) * y.shape[0]
            running_correct += int((logits.detach().argmax(1) == y).sum())
            running_n += int(y.shape[0])

        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        val_probs, val_y = predict(model, eval_src, val_rows, device,
                                   batch_size=max(cfg.batch_size, 64), use_amp=cfg.use_amp)
        val_pred = val_probs.argmax(1)
        val_acc = float((val_pred == val_y).mean())
        # cross-entropy of the true class under the predicted distribution
        val_loss = float(
            -np.log(np.clip(val_probs[np.arange(val_y.shape[0]), val_y], 1e-12, 1.0)).mean()
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "seconds": round(time.time() - t_epoch, 3),
        })

        best_loss = min(best_loss, val_loss)
        if val_acc > best_acc + 1e-9:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            print("    epoch %3d  train_loss %.4f  train_acc %.4f  val_loss %.4f  "
                  "val_acc %.4f  (best %d)" % (epoch, train_loss, train_acc,
                                               val_loss, val_acc, best_epoch))

        if bad_epochs >= cfg.patience:
            if verbose:
                print("    early stop at epoch %d (best %d)" % (epoch, best_epoch))
            break

    total_seconds = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # persist the checkpoint as a state_dict plus the config it belongs to
    ckpt = {"state_dict": model.state_dict(), "config": cfg.to_json(),
            "best_epoch": best_epoch, "val_loss": best_loss}
    torch.save(ckpt, os.path.join(out_dir, "best.pt"))

    test_probs, test_y = predict(model, eval_src, test_rows, device,
                                 batch_size=max(cfg.batch_size, 64), use_amp=cfg.use_amp)
    test_pred = test_probs.argmax(1)
    metrics = compute_metrics(test_y, test_pred, n_classes=3)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_acc"] = best_acc
    metrics["val_loss_at_best"] = best_loss
    metrics["epochs_run"] = len(history)
    metrics["seconds"] = round(total_seconds, 2)
    metrics["seconds_per_epoch"] = round(total_seconds / max(len(history), 1), 3)
    metrics["n_train"] = int(train_rows.shape[0])
    metrics["n_val"] = int(val_rows.shape[0])
    metrics["n_test"] = int(test_rows.shape[0])
    cm = confusion_counts(test_y, test_pred, n_classes=3)

    # per-window predictions, so every headline number can be recomputed
    import pandas as pd

    rows = test_rows
    pd.DataFrame({
        "window_row": rows,
        "subject": index[rows, W_SUBJECT],
        "session": index[rows, W_SESSION],
        "trial": index[rows, W_TRIAL],
        "true": test_y,
        "pred": test_pred,
        "prob_0": test_probs[:, 0],
        "prob_1": test_probs[:, 1],
        "prob_2": test_probs[:, 2],
    }).to_csv(os.path.join(out_dir, "preds_test.csv"), index=False)

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
    np.save(os.path.join(out_dir, "confusion_counts.npy"), cm)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)

    with open(os.path.join(out_dir, "DONE"), "w", encoding="utf-8") as fh:
        fh.write(cfg.run_id())

    if verbose:
        print("    test acc %.4f  kappa %.4f  (%.1f min)"
              % (metrics["accuracy"], metrics["kappa"], total_seconds / 60))
    return metrics


# ---------------------------------------------------------------------------
# batch-size probe
# ---------------------------------------------------------------------------


def probe_batch_size(window_len: int, model_name: str, device=None,
                     start: int = 16, cap: int = 256, vram_fraction: float = 0.8) -> int:
    """Largest power-of-two batch that fits, backed off to ``vram_fraction``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return 32

    best = start
    batch = start
    while batch <= cap:
        try:
            model = build_model(model_name, window_len=window_len).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            x = torch.zeros(batch, 1, 62, window_len, device=device)
            y = torch.zeros(batch, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_logits(model, x)
                loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            best = batch
            del model, opt, x, y, logits, loss
            torch.cuda.empty_cache()
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if free < 1.0:
        best = max(start, best // 2)
    return int(max(start, int(best * vram_fraction)))
