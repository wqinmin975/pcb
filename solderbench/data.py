"""Unified dataset entry point.

.. important::

   **Real-data integration point.**  To evaluate on real 3D-camera point
   clouds instead of the synthetic simulator, replace the body of
   :func:`load_dataset` (or pass your own ``X, y``) with code that reads the
   real scans, segments individual solder joints, and calls
   :func:`solderbench.extract_features` to compute the same seven features.
   All downstream experiments (:func:`run_comparison`, :func:`run_ablation`)
   work unchanged on the returned ``X, y``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .generator import SolderJointGenerator


def load_dataset(
    n_per_class: int = 250,
    seed: int = 42,
    **generator_kwargs,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build the benchmark dataset and return ``(X, y)``.

    Parameters
    ----------
    n_per_class : int, default=250
        Number of samples per class (five classes).
    seed : int, default=42
        Random seed; identical seeds give byte-identical datasets.
    **generator_kwargs
        Forwarded to :class:`solderbench.SolderJointGenerator` (e.g.
        ``n_points``, ``noise``, defect-strength parameters).

    Returns
    -------
    X : pd.DataFrame, shape (5 * n_per_class, 7)
        Feature matrix (see :data:`solderbench.features.FEATURES`).
    y : np.ndarray of int
        Integer labels 0..4.
    """
    gen = SolderJointGenerator(seed=seed, **generator_kwargs)
    clouds, labels = gen.generate(n_per_class=n_per_class)
    return gen.to_features(clouds, labels)
