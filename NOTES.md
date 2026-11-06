# Emo-CT on SEED — implementation notes

This package adapts CTNet to the SEED emotion dataset and runs the full
experiment matrix. It exists because the original manuscript was found to be an
unattributed adaptation of CTNet whose reported numbers were CTNet's motor
imagery results. Everything below is written so that the resulting paper can
state precisely what was done and what differs from CTNet.

## Attribution

The backbone is **CTNet**:

> Zhao W, Jiang X, Zhang B, Xiao S, Weng S. *CTNet: a convolutional transformer
> network for EEG-based motor imagery classification.* Scientific Reports 14,
> 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7

MIT License, Copyright (c) 2024 snailpt, <https://github.com/snailpt/CTNet>.

`CTNet-main/` is left **untouched** as the citable original. All adapted code
lives in `emoct_seed/`; `model_ctnet.py` carries a header listing every change.

## Deviations from CTNet — these must be disclosed in the manuscript

| # | Deviation | Why | Cost |
|---|---|---|---|
| 1 | Dataset: SEED (62 ch, 200 Hz, 3-class emotion) instead of BCI IV-2a/2b (22/3 ch, 250 Hz, 4/2-class motor imagery) | the study's subject is emotion recognition | — |
| 2 | Temporal kernel `KC1 = 50` (one quarter of 200 Hz) instead of 64 (one quarter of 250 Hz) | preserves CTNet's stated rule | — |
| 3 | `flatten_eeg1` derived from the window length instead of hardcoded (240 → 496 for `add`, 992 for `cat` at a 10 s window) | the original constant only fits 1000-sample input | — |
| 4 | `fusion='cat'`: the CNN and Transformer outputs are concatenated, not added (`features = torch.cat([cnn, trans], dim=-1)`) | this is the study's actual architectural contribution, and is run as a formal ablation against CTNet's `add` | — |
| 5 | Model selection on **validation accuracy**, not validation loss | measured: val loss is minimised at epoch 1 while val accuracy is still climbing, so a loss criterion freezes the model essentially untrained | CTNet selects on loss |
| 6 | Early stopping (cap 300 epochs, patience 25) instead of a fixed 1000 epochs | compute budget | CTNet trains a fixed schedule |
| 7 | Real held-out validation set split **by trial**, instead of CTNet's carve-out from each mini-batch | CTNet's "validation set" is re-sampled every epoch and overlaps the training data, so it is not a usable selection signal | — |
| 8 | Per-channel z-scoring using training-window statistics, instead of one global scalar mean/std | standard for SEED; still leak-free | minor |
| 9 | S&R augmentation draws segments from the **training rows only** | grouping the full index would let augmentation draw segments from the held-out test subject (see "Bugs found") | — |
| 10 | fp32 instead of bf16 autocast | measured 20% *faster* here (165 vs 204 ms per batch-128 step): the ops are small and cast overhead dominates | — |
| 11 | `state_dict` + config checkpoints instead of pickled model objects | reproducible loading | cosmetic |
| 12 | No claim of self-performed filtering or artifact removal | the SEED authors already band-pass filtered and downsampled to 200 Hz | the original manuscript claimed 0.5–70 Hz filtering it did not do |

## Data

`data/SEED/SEED_EEG/Preprocessed_EEG/` — 45 session files (`{subject}_{date}.mat`),
variable names carry the subject's initials (`djc_eeg1`…, `jl_eeg1`…), so
discovery is by the `_eeg{k}` suffix, sorted on the integer *k*.

* 62 channels, 200 Hz, 15 trials per session, 675 trials total
* label order fixed: `1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1` → remapped `+1` to {0,1,2}
* trials are **variable length, 185–265 s (mean 226.3 s)** — so the manuscript's
  "180 s trials divided into eight 30 s segments" is not merely wrong, it is
  physically impossible (8 × 30 s = 240 s)

Cache: one-time conversion to uncompressed float32 `.npy` (7.06 GB) under
`C:\Users\Administer\seed_cache\raw\` — the `.mat` files are MATLAB-v5
*compressed*, so reading them inside a training loop is fatal. The cache lives
on `C:` because `D:` has only ~33 GB free.

## Splits (all leak-free, grouped by trial)

| protocol | train | val | test |
|---|---|---|---|
| `cross_session` | sessions 1+2, minus val | ~20% of training trials, class-stratified | session 3 |
| `random_933` | 9 trials/class | 3 trials/class | 3 trials/class |
| `loso` | other 14 subjects, minus val | ~15% of training trials | held-out subject, all 45 trials |

`assert_no_trial_leak` runs inside every builder and is exercised by
`splits.py`'s self-test across all four window lengths.

## Measured environment

* RTX 5060, 8 GB (≈4.8–6.9 GB free with the desktop loaded), sm_120
* torch 2.14.1+cu130, Python 3.12 — **cu126 does not cover sm_120**; PyPI's
  Windows wheels are CPU-only
* **Convolutions run at ~1 TFLOPS here** (12% of the card's 8.15 TFLOPS matmul
  rate), independent of cuDNN settings, channels_last, or precision. This, not
  the code, sets the budget:
  * subject-dependent: **~6 s/epoch**
  * LOSO: **~178 s/epoch**
* Peak VRAM under 1 GiB at batch 32

## Stages

```
python env_check.py
python seed_data.py [raw_dir] [cache_dir]      # one-time; verifies integrity
python splits.py [cache_dir]                   # leak self-test
python model_ctnet.py ; python baseline_models.py ; python models.py
```

```powershell
# Stage 1 - window-length pilot
python run_experiment.py --protocols cross_session --models emo_ct_cat `
  --windows 2,5,10,30 --seeds 0 --subjects 1,2,3 --epochs 100 --patience 15 --tag pilot

# Stage 2 - subject-dependent main runs (6 models x 3 seeds x 2 protocols)
python run_experiment.py --protocols cross_session,random_933 `
  --models ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer `
  --windows <chosen> --seeds 0,1,2 --epochs 300 --patience 25 --tag main

# Stage 3 - LOSO (1 seed)
python run_experiment.py --protocols loso --windows <chosen> --seeds 0 `
  --epochs 300 --patience 25 --tag loso

# Stage 4 - ablations (subject-dependent only)
python run_experiment.py --protocols cross_session `
  --models ctnet_add_cnn_only,emo_ct_cat_cnn_only --windows <chosen> `
  --seeds 0,1,2 --tag ablation

# Stage 5 - statistics and figures
python stats_analysis.py --out experiments/main
python plot_confusion_seed.py --protocol cross_session --window <chosen> `
  --models ctnet_add,emo_ct_cat
```

Every stage is resumable: a run is skipped when its `DONE` sentinel exists, so
a multi-day campaign survives an interrupted process.

## Artifacts per run

`experiments/<tag>/logs/<run_id>/` — `config.json`, `split.json` (the exact
trial ids per split, i.e. the auditable proof that no trial crosses splits),
`train_log.csv`, `best.pt`, `preds_test.csv` (per-window true/pred/probabilities,
so every headline number can be recomputed), `metrics.json`,
`confusion_counts.npy`, `DONE`. `experiments/<tag>/runs.csv` is the ledger and
`MANIFEST.sha256` ties results to exact source (there is no git on this machine).

## Known limitation

DeepConvNet's canonical four stride-3 max-pools need **≥ 441 samples (2.2 s)**.
At a 2 s window it cannot run and `_flat_width` raises a clear error rather than
silently retuning its pooling. If the pilot selects 2 s, DeepConvNet is absent
from that column and this must be stated.

## Bugs found while building this

Recorded because they are the kind of thing this project exists to catch:

1. `augmentation_pool()` grouped the **entire index**, so S&R augmentation drew
   synthetic training segments out of the held-out test subject. Every result it
   produced would have been inflated.
2. The S&R sampling condition tested `n_seg` rather than `per_class * n_seg`
   when deciding whether to sample with replacement.
3. CTNet returns `(features, logits)`; the baselines return logits. The training
   loop unpacked a tuple unconditionally, which would have broken every baseline.
4. The split cache was keyed by protocol alone, so a run sweeping several window
   lengths would have reused splits built against a different window index.
5. Upstream `utils.calMetrics` uses `average='binary'` whenever `max(y)==2`,
   which mis-scores a 3-class problem (SEED labels are {0,1,2}).
6. Upstream `utils.py:5` sets `CUDA_LAUNCH_BLOCKING='1'`, serialising every CUDA
   kernel; and `utils.py:1` hard-imports `cv2`, so the module fails to import at
   all without OpenCV. Both are avoided by not importing `CTNet-main/utils.py`.
