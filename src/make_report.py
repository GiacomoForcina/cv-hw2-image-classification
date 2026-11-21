"""Collect the results of all approaches and generate tables and figures for the report.
Usage: python src/make_report.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from common import RESULTS_DIR, SEED, class_names, load_pets

APPROACHES = {"full": "Full fine-tuning", "head": "Classifier-head training",
              "feat_logreg": "Features + logistic regression",
              "feat_linsvm": "Features + linear SVM"}
FIG_DIR = RESULTS_DIR / "figures"


def load(name):
    m = json.load(open(RESULTS_DIR / name / "metrics.json"))
    p = pd.read_csv(RESULTS_DIR / name / "test_predictions.csv")
    return m, p


def results_table(available):
    rows = []
    for name in available:
        m, _ = load(name)
        rows.append({"approach": APPROACHES[name],
                     "trainable_params": m["trainable_parameters"],
                     "time_s": m["train_time_s"],
                     "selected": m.get("selected_epoch", m["hyperparameters"].get("C")),
                     "train_acc": m["metrics"]["train"]["accuracy"],
                     "val_acc": m["metrics"]["val"]["accuracy"],
                     "test_acc": m["metrics"]["test"]["accuracy"],
                     "test_macro_f1": m["metrics"]["test"]["macro_f1"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    print(df.round(4).to_string(index=False))


def plot_confusion(name, classes):
    _, p = load(name)
    cm = confusion_matrix(p["true"], p["pred"], labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ax.set_title(f"Confusion matrix (test set) - {APPROACHES[name]}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_{name}.png", dpi=200)
    plt.close(fig)

    # Most frequent confusions (off-diagonal cells)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(classes[i], classes[j], off[i, j]) for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape)) if off[i, j] > 0][:10]
    return pd.DataFrame(pairs, columns=["true", "predicted", "count"]).assign(approach=APPROACHES[name])


def per_class_f1(available, classes):
    df = pd.DataFrame({"breed": classes})
    for name in available:
        _, p = load(name)
        df[name] = f1_score(p["true"], p["pred"], labels=range(len(classes)), average=None)
    df.to_csv(RESULTS_DIR / "per_class_f1.csv", index=False)


def plot_examples(name, classes, n=6):
    """Representative correct and incorrect test predictions (random, fixed seed)."""
    _, p = load(name)
    test = load_pets("test")   # raw images, no transform
    rng = np.random.default_rng(SEED)
    correct = rng.choice(p.index[p.true == p.pred], min(n, (p.true == p.pred).sum()), replace=False)
    wrong = rng.choice(p.index[p.true != p.pred], min(n, (p.true != p.pred).sum()), replace=False)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    for row, (idxs, label) in enumerate([(correct, "Correct"), (wrong, "Incorrect")]):
        for ax in axes[row]:
            ax.axis("off")
        for ax, i in zip(axes[row], idxs):
            img, _ = test[int(p.test_index[i])]
            ax.imshow(img)
            ax.set_title(f"True: {classes[p.true[i]]}\nPred: {classes[p.pred[i]]}", fontsize=8,
                         color="green" if row == 0 else "red")
        axes[row][0].text(-0.15, 0.5, label, transform=axes[row][0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=12)
    fig.suptitle(f"Test predictions - {APPROACHES[name]}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"examples_{name}.png", dpi=150)
    plt.close(fig)


def plot_curves():
    """Training vs validation accuracy per epoch for the two neural approaches."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["full", "head"]):
        path = RESULTS_DIR / name / "history.csv"
        if not path.exists():
            continue
        h = pd.read_csv(path)
        ax.plot(h.epoch, h.train_acc_augmented, label="train (augmented)")
        ax.plot(h.epoch, h.val_acc, label="validation")
        ax.set_title(APPROACHES[name])
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    classes = class_names()
    available = [n for n in APPROACHES if (RESULTS_DIR / n / "metrics.json").exists()]
    results_table(available)
    pd.concat([plot_confusion(n, classes) for n in available]).to_csv(
        RESULTS_DIR / "top_confusions.csv", index=False)
    per_class_f1(available, classes)
    for n in available:
        plot_examples(n, classes)
    plot_curves()
    print(f"Tables in {RESULTS_DIR}/, figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
