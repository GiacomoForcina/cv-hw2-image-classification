"""Shared utilities: configuration, seeding, data loading, model building, evaluation."""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ----------------------------------------------------------------------------
# Global configuration (kept identical for all three approaches)
# ----------------------------------------------------------------------------
SEED = 42
DATA_ROOT = Path("data")            # the dataset is downloaded here
SPLIT_DIR = Path("splits")          # train/val index files (committed to the repo)
RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")
NUM_CLASSES = 37
VAL_FRACTION = 0.2                  # 20% of the official trainval -> validation
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT   # ImageNet-1K pre-trained weights


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: ops without a deterministic implementation emit a warning instead of failing
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a reproducible seed (PyTorch reproducibility guide)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def eval_transform():
    """Preprocessing bundled with the pre-trained weights:
    resize to 256, center crop 224, scale to [0, 1], ImageNet mean/std normalization."""
    return WEIGHTS.transforms()


def train_transform():
    """Training-only augmentation, followed by the same normalization as the weights."""
    ref = WEIGHTS.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(ref.crop_size[0], scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=ref.mean, std=ref.std),
    ])


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_pets(split: str, transform=None, download: bool = False):
    """Official Oxford-IIIT Pet partition ('trainval' or 'test'), 37 breed labels."""
    return datasets.OxfordIIITPet(root=str(DATA_ROOT), split=split,
                                  target_types="category",
                                  transform=transform, download=download)


def load_split_indices():
    """Indices into the official trainval partition, created by make_splits.py."""
    train_idx = pd.read_csv(SPLIT_DIR / "train.csv")["index"].to_numpy()
    val_idx = pd.read_csv(SPLIT_DIR / "val.csv")["index"].to_numpy()
    return train_idx, val_idx


def make_loader(dataset, batch_size, shuffle, num_workers):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=seed_worker,
                      generator=g, pin_memory=torch.cuda.is_available())


def get_datasets(train_augment: bool = True):
    """train / val / test datasets with fixed splits.
    Augmentation is applied only to the training subset."""
    train_idx, val_idx = load_split_indices()
    trainval_aug = load_pets("trainval", train_transform() if train_augment else eval_transform())
    trainval_eval = load_pets("trainval", eval_transform())
    test = load_pets("test", eval_transform())
    return {
        "train": Subset(trainval_aug, train_idx),
        "train_eval": Subset(trainval_eval, train_idx),   # train images, no augmentation
        "val": Subset(trainval_eval, val_idx),
        "test": test,
    }


def class_names():
    return load_pets("test").classes


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model():
    """MobileNetV3-Small with ImageNet weights; the last layer is replaced by 37 outputs.
    Original classifier: [0] Linear(576,1024) [1] Hardswish [2] Dropout [3] Linear(1024,1000)."""
    model = mobilenet_v3_small(weights=WEIGHTS)
    in_features = model.classifier[3].in_features          # 1024
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    """Return (true labels, predicted labels) for a whole loader."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_pred.append(logits.argmax(1).cpu())
        y_true.append(y)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def scores(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_predictions(y_true, y_pred, path):
    """Per-image test predictions (index into the official test partition)."""
    pd.DataFrame({"test_index": np.arange(len(y_true)),
                  "true": y_true, "pred": y_pred}).to_csv(path, index=False)
