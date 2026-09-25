"""Geometric feature extraction for solder-joint point clouds.

Seven features are computed per joint (Open3D-free implementation):

============== ============================================================
feature        description
============== ============================================================
vol            volume by voxel counting (self-implemented voxel grid,
               ``VOXEL_SIZE = 0.01``)
max_h          maximum height (z) of the cloud
h_var          variance of the point heights
area           surface area of the convex hull (approximates the joint
               surface; note: dents of collapse joints are "covered" by
               the hull, a known approximation error)
vol_area_ratio ``vol / area``
convex_vol_ratio ``vol / ConvexHull.volume`` -- shape irregularity
min_dist       centre-to-centre distance to the nearest neighbouring joint
============== ============================================================
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from scipy.spatial import ConvexHull, QhullError

#: Ordered list of the seven feature names.
FEATURES: List[str] = [
    "vol",
    "max_h",
    "h_var",
    "area",
    "vol_area_ratio",
    "convex_vol_ratio",
    "min_dist",
]

#: Edge length of the voxel grid used for the voxel-count volume.
VOXEL_SIZE: float = 0.01


def extract_features(
    points: np.ndarray, min_dist: float = np.nan
) -> Dict[str, float]:
    """Extract the seven geometric features from one point cloud.

    Parameters
    ----------
    points : np.ndarray, shape (n, 3)
        Point cloud of a single solder joint.
    min_dist : float, optional
        Distance to the nearest neighbouring joint centre.  Pass ``np.nan``
        (default) when the neighbour layout is unknown; the ``min_dist``
        feature is then undefined (NaN).

    Returns
    -------
    dict
        Mapping from feature name (see :data:`FEATURES`) to value.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("points must have shape (n, 3) with n >= 4")

    z = points[:, 2]
    max_h = float(np.max(z))
    h_var = float(np.var(z))

    # Voxel-count volume.
    vox = np.floor(points / VOXEL_SIZE).astype(np.int64)
    n_vox = len(np.unique(vox, axis=0))
    vol = float(n_vox * (VOXEL_SIZE ** 3))

    # Convex-hull volume / surface area.
    try:
        hull = ConvexHull(points)
        convex_vol = float(hull.volume)
        area = float(hull.area)
    except QhullError:
        # Degenerate (e.g. coplanar) cloud: fall back to voxel volume.
        convex_vol, area = vol, 1.0

    vol_area_ratio = vol / area if area > 1e-9 else 0.0
    convex_vol_ratio = vol / convex_vol if convex_vol > 1e-9 else 1.0

    return {
        "vol": vol,
        "max_h": max_h,
        "h_var": h_var,
        "area": area,
        "vol_area_ratio": float(vol_area_ratio),
        "convex_vol_ratio": float(min(convex_vol_ratio, 1.0)),
        "min_dist": float(min_dist),
    }


def features_to_array(feats: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to an array ordered as :data:`FEATURES`."""
    return np.array([feats[f] for f in FEATURES], dtype=float)
