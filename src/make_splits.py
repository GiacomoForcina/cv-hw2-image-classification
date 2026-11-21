"""Download Oxford-IIIT Pet, split the official trainval partition into train/val
(stratified by breed) and save the split files and the per-breed image counts.
Usage: python src/make_splits.py"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import RESULTS_DIR, SEED, SPLIT_DIR, VAL_FRACTION, load_pets


def main():
    trainval = load_pets("trainval", download=True)
    test = load_pets("test", download=True)
    classes = trainval.classes
    labels = trainval._labels            # breed index of each trainval image
    image_ids = [Path(p).stem for p in trainval._images]

    # Stratified split: every breed keeps the same train/val proportion
    idx = list(range(len(labels)))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_FRACTION,
                                          stratify=labels, random_state=SEED)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", sorted(train_idx)), ("val", sorted(val_idx))]:
        pd.DataFrame({"index": ids,
                      "image_id": [image_ids[i] for i in ids],
                      "label": [labels[i] for i in ids],
                      "breed": [classes[labels[i]] for i in ids]}
                     ).to_csv(SPLIT_DIR / f"{name}.csv", index=False)

    # Number of images per breed in each split
    counts = pd.DataFrame({"breed": classes})
    counts["train"] = pd.Series([labels[i] for i in train_idx]).value_counts().reindex(range(len(classes)), fill_value=0).values
    counts["val"] = pd.Series([labels[i] for i in val_idx]).value_counts().reindex(range(len(classes)), fill_value=0).values
    counts["test"] = pd.Series(test._labels).value_counts().reindex(range(len(classes)), fill_value=0).values
    counts.loc[len(counts)] = ["TOTAL", counts["train"].sum(), counts["val"].sum(), counts["test"].sum()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(RESULTS_DIR / "class_counts.csv", index=False)

    print(f"train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test)}")
    print(f"Split files saved in {SPLIT_DIR}/, class counts in {RESULTS_DIR}/class_counts.csv")


if __name__ == "__main__":
    main()
