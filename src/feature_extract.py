"""Approach 3: frozen MobileNetV3-Small as a feature extractor + scikit-learn classifiers.
Features are taken after the global average pooling (576-dimensional), before the classifier.
Usage: python src/feature_extract.py"""
import argparse
import time

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import (CKPT_DIR, RESULTS_DIR, SEED, WEIGHTS, get_datasets, get_device,
                    make_loader, save_json, save_predictions, scores, set_seed)
from torchvision.models import mobilenet_v3_small

# Candidate classifiers; C (inverse regularization strength) is chosen on the validation set
CLASSIFIERS = {
    "feat_logreg": lambda C: LogisticRegression(C=C, max_iter=3000, random_state=SEED),
    "feat_linsvm": lambda C: LinearSVC(C=C, max_iter=10000, random_state=SEED),
}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


@torch.no_grad()
def extract(backbone, loader, device):
    """Return (features, labels). Features = output of features + avgpool, flattened."""
    feats, labels = [], []
    for x, y in loader:
        f = backbone.avgpool(backbone.features(x.to(device)))   # (B, 576, 1, 1)
        feats.append(torch.flatten(f, 1).cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    set_seed(SEED)
    device = get_device()

    # Frozen pre-trained network: no layer is replaced or trained here
    backbone = mobilenet_v3_small(weights=WEIGHTS).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # No augmentation: every split uses the preprocessing of the pre-trained weights
    ds = get_datasets(train_augment=False)
    t0 = time.time()
    data = {}
    for split in ["train", "val", "test"]:
        loader = make_loader(ds[split], 64, False, args.workers)
        data[split] = extract(backbone, loader, device)
    extract_time = time.time() - t0
    print(f"Features: {data['train'][0].shape[1]}-dim, extraction time {extract_time:.1f}s")

    CKPT_DIR.mkdir(exist_ok=True)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    for name, make_clf in CLASSIFIERS.items():
        # Model selection on the validation set; the scaler is fitted on training data only
        search, t1 = [], time.time()
        best = None
        for C in C_GRID:
            pipe = make_pipeline(StandardScaler(), make_clf(C))
            t_fit = time.time()
            pipe.fit(Xtr, ytr)
            fit_time = time.time() - t_fit
            val = scores(yva, pipe.predict(Xva))
            search.append({"C": C, **val})
            print(f"{name}  C={C:<6}  val acc {val['accuracy']:.3f}  val F1 {val['macro_f1']:.3f}")
            if best is None or val["macro_f1"] > best[1]["macro_f1"]:
                best = (C, val, pipe, fit_time)
        search_time = time.time() - t1
        C, _, pipe, fit_time = best

        # Single evaluation of the selected model on train, val and test
        results = {"train": scores(ytr, pipe.predict(Xtr)),
                   "val": scores(yva, pipe.predict(Xva)),
                   "test": scores(yte, pipe.predict(Xte))}
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(yte, pipe.predict(Xte), out_dir / "test_predictions.csv")
        joblib.dump(pipe, CKPT_DIR / f"{name}.joblib")
        clf = pipe[-1]
        n_params = int(clf.coef_.size + clf.intercept_.size)
        save_json({"approach": name, "device": str(device),
                   "hyperparameters": {"classifier": type(clf).__name__, "C": C,
                                       "C_grid": C_GRID, "feature_scaling": "StandardScaler (fit on train)",
                                       "dimensionality_reduction": "none"},
                   "model_selection": "C with highest validation macro-F1",
                   "validation_search": search,
                   "checkpoint": str(CKPT_DIR / f"{name}.joblib"),
                   "trainable_parameters": n_params,
                   "feature_extraction_time_s": round(extract_time, 1),
                   "classifier_fit_time_s": round(fit_time, 1),
                   "grid_search_time_s": round(search_time, 1),
                   "train_time_s": round(extract_time + search_time, 1),
                   "metrics": results}, out_dir / "metrics.json")
        print(f"{name}: selected C={C}. Test: {results['test']}")


if __name__ == "__main__":
    main()
