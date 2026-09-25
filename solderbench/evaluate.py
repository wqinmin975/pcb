"""Experiment runners and result export for the solder-joint benchmark.

Provides:

* :func:`run_comparison` -- quantitative comparison of the six baselines
  (Accuracy / macro Precision / Recall / F1 on a stratified split, plus
  5-fold cross-validation accuracy on the training set).
* :func:`run_ablation` -- leave-one-feature-out and single-feature ablation
  of the proposed RandomForest model (macro F1).
* Table export to CSV and LaTeX (booktabs) and a confusion-matrix figure.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from .baselines import get_baselines
from .features import FEATURES
from .generator import CLASS_NAMES_EN

DEFAULT_SEED = 42


# ----------------------------------------------------------------------
# Metric evaluation
# ----------------------------------------------------------------------
def evaluate_classifier(
    clf, Xtr, ytr, Xte, yte
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit ``clf`` and return macro-averaged metrics plus predictions."""
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = {
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, average="macro", zero_division=0),
        "Recall": recall_score(yte, pred, average="macro", zero_division=0),
        "F1": f1_score(yte, pred, average="macro", zero_division=0),
    }
    return metrics, pred


# ----------------------------------------------------------------------
# Comparison experiment
# ----------------------------------------------------------------------
def run_comparison(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.3,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare the six baselines on the 7-feature dataset.

    Uses a stratified train/test split (default 7:3) and 5-fold
    cross-validation on the training set.

    Returns
    -------
    pd.DataFrame
        Columns: ``no, name, Accuracy, Precision, Recall, F1, CV_mean,
        CV_std, ours``.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rows: List[Dict] = []
    for no, name, clf, ours in get_baselines(seed=seed):
        metrics, _ = evaluate_classifier(clf, Xtr.values, ytr, Xte.values, yte)
        cv = cross_val_score(
            clf, Xtr.values, ytr, cv=5, scoring="accuracy", n_jobs=-1
        )
        rows.append(
            {
                "no": no,
                "name": name,
                "ours": ours,
                "CV_mean": cv.mean(),
                "CV_std": cv.std(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)[
        ["no", "name", "Accuracy", "Precision", "Recall", "F1",
         "CV_mean", "CV_std", "ours"]
    ]


# ----------------------------------------------------------------------
# Ablation experiment
# ----------------------------------------------------------------------
def run_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.3,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Feature ablation of the proposed RandomForest model (macro F1).

    Reports the full 7-feature model, leave-one-feature-out variants and
    single-feature variants; ``Delta_F1`` is relative to the full model.

    Returns
    -------
    pd.DataFrame
        Columns: ``Setting, Feature, F1, Delta_F1``.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    def _rf():
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)

    full_metrics, _ = evaluate_classifier(_rf(), Xtr.values, ytr, Xte.values, yte)
    f1_full = full_metrics["F1"]
    rows = [
        {"Setting": "Full model", "Feature": "all 7 features",
         "F1": f1_full, "Delta_F1": np.nan}
    ]
    for feat in FEATURES:  # leave-one-feature-out
        cols = [c for c in FEATURES if c != feat]
        m, _ = evaluate_classifier(_rf(), Xtr[cols].values, ytr, Xte[cols].values, yte)
        rows.append({"Setting": "Leave-one-out", "Feature": f"w/o {feat}",
                     "F1": m["F1"], "Delta_F1": m["F1"] - f1_full})
    for feat in FEATURES:  # single feature only
        m, _ = evaluate_classifier(
            _rf(), Xtr[[feat]].values, ytr, Xte[[feat]].values, yte
        )
        rows.append({"Setting": "Single feature", "Feature": f"only {feat}",
                     "F1": m["F1"], "Delta_F1": m["F1"] - f1_full})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Confusion matrix of the proposed model
# ----------------------------------------------------------------------
def confusion_matrix_ours(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.3,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Confusion matrix of RandomForest (Ours) on the stratified test split."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    _, pred = evaluate_classifier(rf, Xtr.values, ytr, Xte.values, yte)
    return confusion_matrix(yte, pred)


def plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    """Render a confusion matrix to ``path`` (PNG, 300 dpi)."""
    n = cm.shape[0]
    names = CLASS_NAMES_EN[:n]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — RandomForest baseline (B6)")
    thr = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thr else "black", fontsize=9,
            )
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Table export (CSV + LaTeX booktabs)
# ----------------------------------------------------------------------
def _bold(x: float) -> str:
    return f"\\textbf{{{x:.4f}}}"


def write_baseline_tex(df: pd.DataFrame, path: str) -> None:
    """Write the baseline comparison table as LaTeX (booktabs)."""
    lines = [
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Quantitative comparison with baseline methods "
        r"(macro-averaged metrics, stratified 7:3 split).}",
        r"  \label{tab:baseline}",
        r"  \begin{tabular}{llccccc}", r"    \toprule",
        r"    No. & Method & Accuracy & Precision & Recall & F1 & CV Acc.\ (5-fold) \\",
        r"    \midrule",
    ]
    for _, r in df.iterrows():
        ours = bool(r["ours"])
        fmt = _bold if ours else (lambda v: f"{v:.4f}")
        name = str(r["name"]).replace(" (Ours)", "")
        if ours:
            name = "\\textbf{" + name + " (Ours)}"
        lines.append(
            f"    {r['no']} & {name} & {fmt(r['Accuracy'])} & {fmt(r['Precision'])} "
            f"& {fmt(r['Recall'])} & {fmt(r['F1'])} & "
            f"{fmt(r['CV_mean'])}$\\pm${r['CV_std']:.4f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_ablation_tex(df: pd.DataFrame, path: str) -> None:
    """Write the ablation table as LaTeX (booktabs)."""
    lines = [
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Feature ablation study of the proposed RandomForest model "
        r"(macro F1 on the test set). $\Delta$F1 = F1$_{\text{variant}}-$F1$_{\text{full}}$.}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{llcc}", r"    \toprule",
        r"    Setting & Feature & macro F1 & $\Delta$F1 \\", r"    \midrule",
    ]
    for _, r in df.iterrows():
        delta = "--" if pd.isna(r["Delta_F1"]) else f"{r['Delta_F1']:+.4f}"
        lines.append(
            f"    {r['Setting']} & {r['Feature']} & {r['F1']:.4f} & {delta} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
