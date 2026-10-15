#!/usr/bin/env python
"""Sensitivity sweep of the hand-set S3DA volume-threshold limits.

The original S3DA baseline is a *rule*, not a trained model: its only free
parameters are the two acceptance limits expressed as fractions of a nominal
reference volume (``low_frac``, ``high_frac``).  This script shows how the
reported accuracy varies when those hand-set constants are changed, so that a
reader can verify that the reported number is not the result of tuning on the
test set.

The nominal reference volume itself is calibrated as the median voxel volume
of the *training* features only (no class label is used), so the sweep below
is likewise free of test-set information.

Usage
-----
    python examples/s3da_threshold_sensitivity.py

Writes ``results/revision/table_s3da_threshold_sensitivity.csv``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from solderbench import S3DARuleClassifier, load_dataset
from solderbench.generator import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "revision")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
GRID = [(0.70, 1.30), (0.80, 1.20), (0.85, 1.15), (0.90, 1.10), (0.95, 1.05)]

rows = []
for seed in SEEDS:
    X, y = load_dataset(n_per_class=250, seed=seed, noise=0.005, n_points=1500)
    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y, test_size=0.3, stratify=y, random_state=seed
    )
    for lo, hi in GRID:
        clf = S3DARuleClassifier(low_frac=lo, high_frac=hi).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        rec = [(pred[yte == k] == k).mean() for k in range(5)]
        rows.append(
            {
                "seed": seed,
                "low_frac": lo,
                "high_frac": hi,
                "Accuracy": accuracy_score(yte, pred),
                **{f"recall_{c}": rec[i] for i, c in enumerate(CLASSES)},
            }
        )

df = pd.DataFrame(rows)
path = os.path.join(OUT, "table_s3da_threshold_sensitivity.csv")
df.to_csv(path, index=False)

agg = df.groupby(["low_frac", "high_frac"])["Accuracy"].agg(["mean", "std"])
print("S3DA rule accuracy vs. hand-set acceptance limits (5 seeds, mean +/- std):")
print(agg.to_string())
print(f"\nSaved per-seed detail to {path}")
print(
    "\nReported configuration: low_frac=0.90, high_frac=1.10 "
    "(hand-set process constants, not optimised on the test set)."
)
