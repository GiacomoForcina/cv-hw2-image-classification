# Homework 2 — Image Classification with Transfer Learning

Breed classification (37 classes) on the Oxford-IIIT Pet dataset with a pre-trained
MobileNetV3-Small, comparing three transfer-learning strategies:

1. **Full fine-tuning** — the whole network is trained.
2. **Classifier-head training** — the convolutional feature extractor is frozen; only the
   classifier's fully connected layers are trained.
3. **Feature extraction + traditional classifier** — the frozen network produces 576-dimensional
   features (after global average pooling); scikit-learn classifiers (logistic regression and
   linear SVM) are trained on them.

Computer Vision course — University of Verona, Fall 2026. Author: Giacomo Federico Forcina.

## Environment

Tested with (exact versions are also written to `results/environment.json` by `src/env_info.py`):

| Software | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| TorchVision | 0.29.1+cu130 |
| scikit-learn | 1.6.1 |
| Hardware | Google Colab, NVIDIA T4 GPU |

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

## Dataset

**Oxford-IIIT Pet** — O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
*Cats and Dogs*, CVPR 2012. https://www.robots.ox.ac.uk/~vgg/data/pets/
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0); the copyright of
the images remains with their original owners.

The images are **not** included in this repository. They are downloaded automatically
(about 800 MB) into `data/` by the first command below, through `torchvision.datasets.OxfordIIITPet`.

**Splits.** The official `trainval` and `test` partitions are used. `trainval` is split into
training (80%) and validation (20%) with stratification by breed (`sklearn.model_selection.train_test_split`,
seed 42). The split files are committed in `splits/` (`index` refers to the order of the official
`trainval.txt` list). The test partition is used only for the final evaluation of each selected model.
Image counts per breed and split: `results/class_counts.csv`.

## Pre-trained model

`torchvision.models.mobilenet_v3_small` with `MobileNet_V3_Small_Weights.DEFAULT`
(ImageNet-1K). The last layer `classifier[3]` (Linear 1024→1000) is replaced by Linear 1024→37.

## Reproducing the experiments

Run from the project root, in this order:

```bash
python src/env_info.py                       # record software versions
python src/make_splits.py                    # download dataset, create splits and class counts
python src/train_nn.py --strategy full       # approach 1: full fine-tuning
python src/train_nn.py --strategy head       # approach 2: classifier-head training
python src/feature_extract.py                # approach 3: features + scikit-learn classifiers
python src/make_report.py                    # results table, confusion matrices, figures
```

On Windows, if the data loaders give errors, add `--workers 0` to the training commands.

## Training setup

| | Full fine-tuning | Head training | Feature extraction |
|---|---|---|---|
| Trainable layers | all | `classifier.0` (Linear 576→1024) and `classifier.3` (Linear 1024→37) | scikit-learn classifier only |
| Loss | cross-entropy | cross-entropy | log-loss (LogReg) / squared hinge (LinearSVC) |
| Optimizer | AdamW, weight decay 1e-4 | AdamW, weight decay 1e-4 | — |
| Learning rate | 1e-4, cosine schedule | 1e-3, cosine schedule | — |
| Batch size | 32 | 32 | 64 (extraction) |
| Epochs | 15 | 15 | — |
| Model selection | epoch with best validation macro-F1 | epoch with best validation macro-F1 | C ∈ {0.001, 0.01, 0.1, 1, 10} with best validation macro-F1 |

- **Preprocessing** (all splits): the transform of the pre-trained weights — resize to 256,
  center crop 224, ImageNet mean/std normalization.
- **Augmentation** (training set of approaches 1–2 only): `RandomResizedCrop(224, scale=(0.6, 1))`
  and horizontal flip. No augmentation for feature extraction. No class balancing (classes are
  almost balanced, see `results/class_counts.csv`).
- **Feature scaling**: `StandardScaler` fitted on training features only. No dimensionality reduction.
- In head training the frozen feature extractor is kept in `eval` mode, so its BatchNorm
  statistics are not updated.

## Reproducibility

- Seed **42** for Python (`random`), NumPy, PyTorch (CPU and CUDA), the train/val split and the
  DataLoader shuffling (seeded generator and `worker_init_fn`).
- Deterministic operations are enabled: `torch.backends.cudnn.deterministic = True`,
  `benchmark = False`, `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Remaining sources of variation: results can differ slightly across hardware (CPU vs GPU,
  different GPUs), library versions and number of data-loader workers; operations without a
  deterministic implementation only emit a warning.
- The selected checkpoint and hyperparameters of each approach are recorded in
  `results/<approach>/metrics.json`.

## Repository structure

```
├── README.md
├── requirements.txt
├── splits/                 # train.csv, val.csv (indices into the official trainval list)
├── src/
│   ├── common.py           # configuration, seeds, data, model, metrics
│   ├── env_info.py         # software/hardware versions
│   ├── make_splits.py      # download + stratified split + class counts
│   ├── train_nn.py         # approaches 1 and 2
│   ├── feature_extract.py  # approach 3
│   └── make_report.py      # tables and figures
├── checkpoints/            # full.pt, head.pt, feat_logreg.joblib, feat_linsvm.joblib
└── results/
    ├── environment.json, class_counts.csv, results_table.csv,
    │   per_class_f1.csv, top_confusions.csv
    ├── full/ head/ feat_logreg/ feat_linsvm/   # metrics.json, test_predictions.csv, history.csv
    └── figures/            # confusion matrices, prediction examples, learning curves
```

Licenses: MobileNetV3 weights from TorchVision (BSD-3-Clause). Code: MIT.
