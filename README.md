# Emo-CT on SEED — code and results

Code, trained-model records and analysis scripts for the manuscript

> **Spatiotemporal EEG Feature Learning for Emotion Recognition Using a Convolution–Transformer Network**

The convolutional–Transformer backbone is adapted from **CTNet** (Zhao et al., *Scientific
Reports* 14, 20237, 2024; MIT licence). This repository contains the re-implementation for
three-class EEG emotion recognition on SEED, the complete experiment matrix, and the
statistics and figures reported in the paper.

---

## 1. What is in this repository

```
*.py                        # experiment code (see "Code" below)
run_campaign.ps1            # detached launcher for one experiment stage
run_all.ps1                 # detached launcher for all stages
NOTES.md                    # implementation notes and every deviation from CTNet
requirements.txt
figures/                    # manuscript figures (PNG + PDF)
results/
  main/    runs.csv, aggregate/{all_runs,descriptive,stats}.csv
  loso/    runs.csv, aggregate/{all_runs,descriptive,stats}.csv
  ablation/runs.csv, aggregate/{all_runs,descriptive,stats}.csv
  logs_main.zip, logs_loso.zip, logs_ablation.zip   # per-run artifacts (see below)
```

Each archived run directory (`logs_<tag>.zip`) contains:

| file | content |
|---|---|
| `config.json` | model, protocol, window length, seed, hyper-parameters |
| `split.json` | the exact trial ids in the train/val/test split (auditable, leak-free) |
| `train_log.csv` | per-epoch train/val loss and accuracy |
| `preds_test.csv` | per-window true label, prediction and class probabilities |
| `metrics.json` | accuracy, kappa, macro precision/recall/F1, epochs, runtime |
| `confusion_counts.npy` | 3×3 confusion matrix |
| `DONE` | completion sentinel |

Model checkpoints (`best.pt`, 312 MB in total) are **not** included; every run is fully
reproducible from the code and the split definitions. They are available from the authors
on request.

## 2. Data

SEED is not redistributed here: it is released by the Institute of Brain Functional
Genomics, Shanghai Jiao Tong University, under an application-based licence.

1. Request SEED at <https://bcmi.sjtu.edu.cn/home/seed/>.
2. Put the 45 session files (`Preprocessed_EEG/{subject}_{date}.mat`) anywhere on disk and
   pass that directory as the first positional argument to `seed_data.py` (the second
   positional argument is the cache directory).
3. Build the uncompressed float32 cache once (7.06 GB):

```powershell
python seed_data.py "data\SEED\SEED_EEG\Preprocessed_EEG" "C:\seed_cache"
```

Every later command takes `--cache C:\seed_cache`. The cache is memory-mapped; the
experiments do not copy the data into RAM.

## 3. Environment

Tested with Python 3.12 and:

```
torch 2.14.1+cu130   numpy 2.5.3   pandas 3.0.6   scipy 1.18.1
scikit-learn 1.9.1   matplotlib 3.11.2   einops 0.8.2
```

A CUDA build of PyTorch is required for the reported runtimes; `--device cpu` is not
implemented. Hardware used: one NVIDIA RTX 5060 (8 GB), Intel i5-14400F, 16 GB RAM,
SATA SSD.

> Memory note: the LOSO experiments hold a 7 GB memory-mapped cache and peak at ~6 GB
> resident. On a 16 GB machine, close other applications before launching a campaign —
> runs fail with `Unable to allocate N MiB` when free memory drops below ~1 GB.

## 4. Reproducing the results

```powershell
# sanity checks
python env_check.py
python splits.py "C:\seed_cache"          # leak self-test for all window lengths

# subject-dependent: 6 models x 2 protocols x 15 subjects x 3 seeds (540 runs)
python run_experiment.py --protocols cross_session,random_933 `
  --models ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer `
  --windows 5 --seeds 0,1,2 --epochs 300 --patience 25 --cache "C:\seed_cache" --tag main

# LOSO: 6 models x 15 folds (90 runs)
python run_experiment.py --protocols loso `
  --models ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer `
  --windows 5 --seeds 0 --epochs 300 --patience 25 --cache "C:\seed_cache" --tag loso

# ablation: CNN-only backbones (90 runs)
python run_experiment.py --protocols cross_session `
  --models ctnet_add_cnn_only,emo_ct_cat_cnn_only `
  --windows 5 --seeds 0,1,2 --epochs 300 --patience 25 --cache "C:\seed_cache" --tag ablation

# statistics and figures
python stats_analysis.py --out experiments\main     --protocols cross_session,random_933 --windows 5
python stats_analysis.py --out experiments\loso     --protocols loso --windows 5
python stats_analysis.py --out experiments\ablation --protocols cross_session --windows 5 `
  --proposed emo_ct_cat_cnn_only --reference ctnet_add_cnn_only
python plot_confusion_seed.py --out experiments\main --protocol cross_session --window 5 `
  --models ctnet_add,emo_ct_cat
```

Every stage is resumable: a run whose `DONE` sentinel exists is skipped, so an interrupted
campaign can be relaunched with the same command.

Typical runtimes on the hardware above: subject-dependent ~5–12 min per run; LOSO
1.7–11.7 h per fold (6.5 h on average).

## 5. Where each manuscript table comes from

| manuscript | file |
|---|---|
| Table 1 (cross-session) | `results/main/aggregate/descriptive.csv`, rows `protocol=cross_session` |
| Table 2 (random 9/3/3) | `results/main/aggregate/descriptive.csv`, rows `protocol=random_933` |
| Table 3 (LOSO summary) | `results/loso/aggregate/descriptive.csv` |
| Table 4 (per-subject LOSO) | `results/loso/aggregate/all_runs.csv` (one row per fold) |
| Table 5 (ablation) | `results/ablation/aggregate/descriptive.csv` |
| Table 6 (per-class recall) | recomputed from `preds_test.csv` |
| all p-values and effect sizes | `results/*/aggregate/stats.csv` |
| Fig. 2 (confusion matrix) | `figures/confusion_emo_ct_cat_cross_session_5s.png` |

`MANIFEST.sha256` inside each results package ties every run to the exact source files
that produced it (this machine had no git).

## 6. Code

| file | role |
|---|---|
| `seed_data.py` | SEED discovery, float32 cache, window index, memmap access |
| `splits.py` | cross-session / random 9-3-3 / LOSO splits, leak self-test |
| `model_ctnet.py` | CTNet adaptation (add and concat fusion), with change log |
| `baseline_models.py` | EEGNet, ShallowConvNet, DeepConvNet, Conformer |
| `models.py` | model registry and `forward_logits` |
| `runner.py` | training loop, S&R augmentation, evaluation, artifacts |
| `run_experiment.py` | resumable experiment-matrix driver |
| `stats_analysis.py` | aggregation, paired Wilcoxon, Holm correction, effect sizes |
| `plot_confusion_seed.py` | confusion-matrix figure from raw predictions |
| `status.py` | progress report for a running campaign |
| `env_check.py` | environment / dataset sanity checks |

## 7. Known limitations

- DeepConvNet's canonical four stride-3 max-pools require ≥ 441 samples; it cannot run on
  windows shorter than 2.2 s.
- The Transformer branch and the concatenation fusion show no measurable effect on SEED at
  this scale; see the manuscript for the full negative result.
- Conformer uses the published architecture but was re-trained here, so its numbers are not
  directly comparable to the original paper's.

## 8. Licence and attribution

The code in this repository is released under the MIT licence (see `LICENSE`).
It adapts CTNet (Copyright © 2024 snailpt, MIT licence); the original licence text is
reproduced in `THIRD_PARTY_NOTICES.md`. Any use of this code should cite both this work and:

> Zhao W, Jiang X, Zhang B, Xiao S, Weng S. CTNet: a convolutional transformer network for
> EEG-based motor imagery classification. *Scientific Reports* 14, 20237 (2024).
> https://doi.org/10.1038/s41598-024-71118-7

SEED remains the property of its authors and is subject to its own licence.

## 9. Citation

```
[author list]. Spatiotemporal EEG Feature Learning for Emotion Recognition Using a
Convolution–Transformer Network. [journal], [year]. [DOI]
```
