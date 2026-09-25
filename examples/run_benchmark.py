#!/usr/bin/env python
"""Reproduce the full solderbench benchmark.

Runs the baseline comparison and the feature ablation on the synthetic
dataset (250 samples x 5 classes, seed=42) and writes all artefacts to
``results/``:

* ``results/tables/table_baseline.csv`` / ``.tex``
* ``results/tables/table_ablation.csv`` / ``.tex``
* ``results/figures/confusion_matrix.png``

Usage (from the repository root)::

    python examples/run_benchmark.py
"""

import os
import sys

# Make the package importable when running from the repository root
# without installing it first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solderbench import (  # noqa: E402
    confusion_matrix_ours,
    load_dataset,
    plot_confusion_matrix,
    run_ablation,
    run_comparison,
    write_ablation_tex,
    write_baseline_tex,
)

SEED = 42
N_PER_CLASS = 250
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def main() -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 68)
    print("solderbench -- baseline comparison & ablation (seed=%d)" % SEED)
    print("=" * 68)

    print(f"\n[1/4] Generating dataset ({N_PER_CLASS} samples x 5 classes) ...")
    X, y = load_dataset(n_per_class=N_PER_CLASS, seed=SEED)
    print(f"      Dataset: {X.shape[0]} samples, {X.shape[1]} features: "
          f"{list(X.columns)}")

    print("\n[2/4] Baseline comparison (stratified 7:3 split, 5-fold CV):")
    df_base = run_comparison(X, y, seed=SEED)
    for _, r in df_base.iterrows():
        mark = "  <== Ours" if r["ours"] else ""
        print(f"      {r['no']} {r['name']:28s} Acc={r['Accuracy']:.4f} "
              f"P={r['Precision']:.4f} R={r['Recall']:.4f} "
              f"F1={r['F1']:.4f} CV={r['CV_mean']:.4f}+-{r['CV_std']:.4f}{mark}")
    df_base.drop(columns=["ours"]).round(4).to_csv(
        os.path.join(TABLE_DIR, "table_baseline.csv"),
        index=False, encoding="utf-8-sig",
    )
    write_baseline_tex(df_base, os.path.join(TABLE_DIR, "table_baseline.tex"))

    print("\n[3/4] Ablation study (RandomForest, macro F1 on test set):")
    df_ab = run_ablation(X, y, seed=SEED)
    for _, r in df_ab.iterrows():
        delta = "" if r["Delta_F1"] != r["Delta_F1"] else f"  dF1={r['Delta_F1']:+.4f}"
        print(f"      {r['Setting']:15s} {r['Feature']:20s} F1={r['F1']:.4f}{delta}")
    df_ab.round(4).to_csv(
        os.path.join(TABLE_DIR, "table_ablation.csv"),
        index=False, encoding="utf-8-sig",
    )
    write_ablation_tex(df_ab, os.path.join(TABLE_DIR, "table_ablation.tex"))

    print("\n[4/4] Confusion matrix figure ...")
    cm = confusion_matrix_ours(X, y, seed=SEED)
    plot_confusion_matrix(cm, os.path.join(FIG_DIR, "confusion_matrix.png"))

    print("\nOutput files:")
    for p in ["tables/table_baseline.csv", "tables/table_baseline.tex",
              "tables/table_ablation.csv", "tables/table_ablation.tex",
              "figures/confusion_matrix.png"]:
        print("      " + os.path.join(OUT_DIR, p))
    print("\nDone.")


if __name__ == "__main__":
    main()
