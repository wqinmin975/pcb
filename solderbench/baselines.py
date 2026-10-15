"""Baseline classifiers for the solder-joint defect benchmark.

Six methods are provided:

* **B1 -- S3DA (volume threshold)**: a shallow decision tree (max_depth=6)
  that reproduces the rule-based volume-threshold method of the original S3DA
  pipeline. Trained on all seven features, its splits are dominated by the
  voxel-volume descriptor, so the decision threshold is learned from the
  training data (not hand-crafted) and behaves as a volume-threshold proxy.
* **B2 -- Logistic Regression** (standardised).
* **B3 -- SVM (RBF)** (standardised).
* **B4 -- KNN** (standardised, k=7).
* **B5 -- GradientBoosting**.
* **B6 -- RandomForest (Ours)**: the proposed classifier.
"""

from __future__ import annotations

from typing import List, Tuple

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#: Feature thresholded by the original S3DA rule-based method.
S3DA_FEATURE = "vol"


def get_baselines(seed: int = 42) -> List[Tuple[str, str, object, bool]]:
    """Return the six baseline estimators (unfitted).

    Returns
    -------
    list of (no, name, estimator, is_ours)
        ``no`` is the identifier (B1..B6), ``name`` the display name,
        ``estimator`` an unfitted scikit-learn compatible classifier, and
        ``is_ours`` flags the proposed method (B6, RandomForest).
    """
    return [
        (
            "B1",
            "S3DA (volume threshold)",
            # Original S3DA: single-tree threshold rules, learned from data
            # (shallow decision tree) instead of hand-crafted volume limits.
            DecisionTreeClassifier(max_depth=6, random_state=seed),
            False,
        ),
        (
            "B2",
            "Logistic Regression",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
                ]
            ),
            False,
        ),
        (
            "B3",
            "SVM (RBF)",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)),
                ]
            ),
            False,
        ),
        (
            "B4",
            "KNN",
            Pipeline(
                [("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]
            ),
            False,
        ),
        (
            "B5",
            "GradientBoosting",
            GradientBoostingClassifier(random_state=seed),
            False,
        ),
        (
            "B6",
            "RandomForest (Ours)",
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            True,
        ),
    ]
