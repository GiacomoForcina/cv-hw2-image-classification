"""Approaches 1 and 2: full fine-tuning and classifier-head training.
Usage:
    python src/train_nn.py --strategy full
    python src/train_nn.py --strategy head
"""
import argparse
import time

import pandas as pd
import torch
from torch import nn

from common import (CKPT_DIR, RESULTS_DIR, SEED, build_model, count_trainable,
                    get_datasets, get_device, make_loader, predict, save_json,
                    save_predictions, scores, set_seed)

# Hyperparameters of each strategy (fixed in advance, reported in the README/report)
CONFIG = {
    # whole network trained -> smaller learning rate, not to destroy pre-trained features
    "full": {"lr": 1e-4, "epochs": 15, "batch_size": 32, "weight_decay": 1e-4},
    # only the classifier trained -> a larger learning rate is safe
    "head": {"lr": 1e-3, "epochs": 15, "batch_size": 32, "weight_decay": 1e-4},
}


def configure_trainable(model, strategy):
    """full: every parameter is trainable.
    head: the convolutional feature extractor (model.features) is frozen; the two
    fully connected layers of the classifier (classifier.0 and classifier.3) are trained."""
    if strategy == "head":
        for p in model.features.parameters():
            p.requires_grad = False


def set_train_mode(model, strategy):
    model.train()
    if strategy == "head":
        # Frozen BatchNorm layers must stay in eval mode, otherwise their running
        # statistics would still be updated and the extractor would not be truly frozen.
        model.features.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["full", "head"], required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cfg = CONFIG[args.strategy]

    set_seed(SEED)
    device = get_device()
    ds = get_datasets(train_augment=True)
    train_loader = make_loader(ds["train"], cfg["batch_size"], True, args.workers)
    val_loader = make_loader(ds["val"], 64, False, args.workers)

    model = build_model()
    configure_trainable(model, args.strategy)
    model.to(device)
    n_trainable = count_trainable(model)
    print(f"Strategy: {args.strategy} | device: {device} | trainable parameters: {n_trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    CKPT_DIR.mkdir(exist_ok=True)
    out_dir = RESULTS_DIR / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.strategy}.pt"

    history, best_val_f1, best_epoch = [], -1.0, -1
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        set_train_mode(model, args.strategy)
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        scheduler.step()

        # Model selection uses ONLY the validation set (never the test set)
        yv, pv = predict(model, val_loader, device)
        val = scores(yv, pv)
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "train_acc_augmented": correct / seen,
                        "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        print(f"epoch {epoch:2d}  loss {total_loss / seen:.3f}  "
              f"train acc {correct / seen:.3f}  val acc {val['accuracy']:.3f}  val F1 {val['macro_f1']:.3f}")
        if val["macro_f1"] > best_val_f1:          # keep the best checkpoint on validation macro-F1
            best_val_f1, best_epoch = val["macro_f1"], epoch
            torch.save(model.state_dict(), ckpt_path)
    train_time = time.time() - t0
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Reload the selected checkpoint and evaluate it ONCE on each split
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    results = {}
    for split in ["train_eval", "val", "test"]:
        loader = make_loader(ds[split], 64, False, args.workers)
        y_true, y_pred = predict(model, loader, device)
        results[split.replace("_eval", "")] = scores(y_true, y_pred)
        if split == "test":
            save_predictions(y_true, y_pred, out_dir / "test_predictions.csv")

    save_json({"approach": args.strategy, "device": str(device),
               "hyperparameters": {**cfg, "optimizer": "AdamW", "loss": "CrossEntropyLoss",
                                   "scheduler": "CosineAnnealingLR"},
               "model_selection": "epoch with highest validation macro-F1",
               "selected_epoch": best_epoch, "checkpoint": str(ckpt_path),
               "trainable_parameters": n_trainable, "train_time_s": round(train_time, 1),
               "metrics": results}, out_dir / "metrics.json")
    print(f"Selected epoch {best_epoch}. Test: {results['test']}")


if __name__ == "__main__":
    main()
