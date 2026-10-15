"""Baseline classifiers for the solder-joint defect benchmark.

Six methods are provided:

* **B1 -- S3DA (volume-threshold rule)**: the *original* rule-based algorithm
  of Xu et al. [6].  Solder volume is obtained by integral summation over the
  joint region (here: the voxel-count volume descriptor ``vol``) and compared
  against two hand-set process-specification limits, expressed as fractions of
  a calibrated nominal volume.  No class label and no model training is
  involved: the rule is deterministic and identical for every seed.
* **B2 -- Logistic Regression** (standardised).
* **B3 -- SVM (RBF)** (standardised).
* **B4 -- KNN** (standardised, k=7).
* **B5 -- GradientBoosting**.
* **B6 -- RandomForest (Ours)**: the proposed classifier.

A *learned* variant of S3DA is also provided for reference
(:func:`get_s3da_tree_proxy`): a shallow decision tree (``max_depth=6``)
trained on all seven geometric features, whose learned splits are dominated by
the voxel-volume descriptor and which therefore acts as a data-driven
volume-threshold proxy.  It is **not** used for any number reported in the
manuscript; the manuscript's S3DA row is the original hand-crafted rule above.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from .features import FEATURES

#: Feature thresholded by the original S3DA rule-based method.
S3DA_FEATURE = "vol"

#: Column index of ``S3DA_FEATURE`` in the feature matrix.
S3DA_FEATURE_INDEX = FEATURES.index(S3DA_FEATURE)

#: Class indices of the rule's three possible outputs.
CLASS_GOOD = 0
CLASS_INSUFFICIENT = 1
CLASS_EXCESSIVE = 2


class S3DARuleClassifier(BaseEstimator, ClassifierMixin):
    """Original S3DA: hand-set volume-threshold rule (no model training).

    Following Xu et al. [6], the solder volume of each joint is computed by
    integral summation (implemented here by the voxel-count volume descriptor
    ``vol``) and compared against two **hand-set** limits that are fractions of
    a nominal reference volume:

    * ``vol < low_frac  * V_ref``  -> ``insufficient solder``,
    * ``vol > high_frac * V_ref``  -> ``excessive solder``,
    * otherwise                    -> ``good``.

    ``V_ref`` is calibrated in :meth:`fit` as the **median volume of the
    training feature matrix**; this uses the unlabelled feature distribution
    only, so no class label and no supervised optimisation is involved.  The
    rule form, the two fractions and the calibration statistic are fixed
    constants, identical for every random seed.

    Because the published rule thresholds volume alone, it can only ever emit
    {good, insufficient, excessive}: bridging and collapse are not volume
    defects and are therefore never predicted by this baseline.

    Parameters
    ----------
    low_frac : float, default=0.90
        Lower acceptance limit, as a fraction of the nominal volume.
    high_frac : float, default=1.10
        Upper acceptance limit, as a fraction of the nominal volume.
    """

    def __init__(self, low_frac: float = 0.90, high_frac: float = 1.10) -> None:
        self.low_frac = float(low_frac)
        self.high_frac = float(high_frac)

    # Unsupervised calibration of the nominal reference volume.
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = int(X.shape[1])
        self.nominal_volume_ = float(np.median(X[:, S3DA_FEATURE_INDEX]))
        self.classes_ = (
            np.unique(y) if y is not None else np.array([0, 1, 2], dtype=int)
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        vol = np.asarray(X, dtype=float)[:, S3DA_FEATURE_INDEX]
        lo = self.low_frac * self.nominal_volume_
        hi = self.high_frac * self.nominal_volume_
        pred = np.where(
            vol < lo,
            CLASS_INSUFFICIENT,
            np.where(vol > hi, CLASS_EXCESSIVE, CLASS_GOOD),
        )
        return pred.astype(int)


def get_s3da_tree_proxy(seed: int = 42):
    """Return the *learned* S3DA proxy (reference only, not used in the paper).

    A shallow decision tree (``max_depth=6``) trained on all seven geometric
    features.  Its learned splits are dominated by the voxel-volume descriptor,
    so it behaves as a data-driven volume-threshold proxy whose decision
    threshold is *learned from the training split* rather than hand-crafted.
    """
    return DecisionTreeClassifier(max_depth=6, random_state=seed)


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
            "S3DA (volume rule)",
            # Original S3DA: hand-set volume-threshold rule, no training.
            S3DARuleClassifier(),
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
