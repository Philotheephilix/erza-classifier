"""
Reproducible build script — generates training data, builds feature matrices, and saves everything.
Run from the project root: python3 training/build_features.py
"""
import argparse
import datetime
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from data.generate_synthetic import generate_dataset
from data.feature_engineering import BertTinyEncoder, build_feature_matrix
from data.feature_schema import CATEGORY_LABELS


def build(n_samples: int = 9000, seed: int = 42, archive: bool = True):
    print(f"Generating dataset: n_samples={n_samples}, seed={seed}")
    df = generate_dataset(n_samples=n_samples, seed=seed)
    print(f"  Total samples: {len(df)}")
    print(f"  Category distribution:\n{df['label_category'].value_counts().to_string()}")

    print("Building feature matrix (BERT-tiny encoding)...")
    enc = BertTinyEncoder()
    X = build_feature_matrix(df, enc)
    print(f"  Feature matrix shape: {X.shape}")

    cat_label_map = {label: i for i, label in enumerate(CATEGORY_LABELS)}
    y_imp = df["label_importance"].values.astype(np.float32)
    y_urg = df["label_urgency"].values.astype(np.float32)
    y_cat = df["label_category"].map(cat_label_map).values.astype(np.int64)

    # Save to data/
    np.save("data/features_train.npy", X)
    np.save("data/labels_importance.npy", y_imp)
    np.save("data/labels_urgency.npy", y_urg)
    np.save("data/labels_category.npy", y_cat)
    df.to_json("data/training_data.json", orient="records", indent=2)
    print("Saved: data/features_train.npy, labels_*.npy, training_data.json")

    if archive:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path(f"data/archive/{ts}")
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy("data/features_train.npy", archive_dir / "features_train.npy")
        shutil.copy("data/labels_importance.npy", archive_dir / "labels_importance.npy")
        shutil.copy("data/labels_urgency.npy", archive_dir / "labels_urgency.npy")
        shutil.copy("data/labels_category.npy", archive_dir / "labels_category.npy")
        shutil.copy("data/training_data.json", archive_dir / "training_data.json")
        # Save build metadata
        (archive_dir / "meta.txt").write_text(
            f"n_samples={n_samples}\nseed={seed}\ntotal={len(df)}\nshape={X.shape}\n"
            f"built={ts}\ncategories={CATEGORY_LABELS}\n"
        )
        print(f"Archived to: {archive_dir}/")

    print("Done.")
    return X, y_imp, y_urg, y_cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training features")
    parser.add_argument("--n", type=int, default=9000, help="Number of synthetic samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-archive", action="store_true", help="Skip archiving")
    args = parser.parse_args()
    build(n_samples=args.n, seed=args.seed, archive=not args.no_archive)
