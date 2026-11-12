
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

CSV_PATH = Path("data/processed/ecg_labels.csv")
ECG_DIR = Path("data/raw/ecg_100hz")
OUTPUT_DIR = Path("data/processed")

df = pd.read_csv(CSV_PATH)

X = []
y = []
splits = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading ECGs"):
    record_path = ECG_DIR / row["filename_lr"]

    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal.astype(np.float32)

    # Convert from (1000, 12) to (12, 1000)
    signal = signal.T

    # Normalize each lead independently
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    signal = (signal - mean) / np.maximum(std, 1e-6)

    X.append(signal)
    y.append(int(row["label"]))
    splits.append(row["split"])

X = np.stack(X)
y = np.array(y, dtype=np.int64)
splits = np.array(splits)

for split in ["train", "validation", "test"]:
    mask = splits == split

    np.savez_compressed(
        OUTPUT_DIR / f"{split}.npz",
        X=X[mask],
        y=y[mask]
    )

    print(
        f"{split}: X={X[mask].shape}, "
        f"y={y[mask].shape}"
    )

print("\nDataset preparation complete!")
