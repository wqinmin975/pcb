"""Synthetic point-cloud generator for PCB solder-joint defect inspection.

This module simulates 3D point clouds of surface-mount solder joints in five
classes: ``good``, ``insufficient``, ``excessive``, ``bridging`` and
``collapse``.  The implementation is pure NumPy (no Open3D dependency) and
approximates the generation logic of the S3DA-style simulator used in the
accompanying study.

Methodological notes (do not remove):

* **Fixed point count** -- every cloud is subsampled to exactly ``n_points``
  points so that voxel-count volume estimates are not biased by sampling
  density (this simulates a fixed scanner resolution).
* **Realistic class overlap** -- per-sample parameter jitter deliberately
  keeps the class distributions overlapping (e.g. the upper tail of
  "insufficient" overlaps the lower tail of "good"), so no classifier can
  reach a perfect score on this benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import FEATURES, extract_features

#: Canonical class identifiers used throughout the package.
CLASSES: List[str] = ["good", "insufficient", "excessive", "bridging", "collapse"]

#: Human-readable English class names (for figures and tables).
CLASS_NAMES_EN: List[str] = ["Good", "Insufficient", "Excessive", "Bridging", "Collapse"]

#: Mapping from class identifier to integer label.
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class SolderSample:
    """One simulated solder joint.

    Attributes
    ----------
    points : np.ndarray, shape (n_points, 3)
        3D point cloud of the joint (plus defect geometry, if any).
    min_dist : float
        Distance from this joint's centre to the nearest neighbouring joint
        centre.  On real data this is measured as the Euclidean distance
        between segmented joint centres.
    label : str
        One of :data:`CLASSES`.
    """

    points: np.ndarray
    min_dist: float
    label: str


class SolderJointGenerator:
    """Configurable simulator of PCB solder-joint point clouds.

    Parameters
    ----------
    n_points : int, default=1500
        Fixed number of points per cloud (uniform sampling density; see the
        module docstring for why this matters).
    noise : float, default=0.005
        Standard deviation of the Gaussian measurement noise added to every
        coordinate (sensor noise), in the same units as the geometry.
    seed : int or None, default=None
        Seed for the internal random number generator.  The same seed
        reproduces byte-identical datasets.
    base_radius : float, default=0.12
        Nominal horizontal radius ``a`` of a good joint (half-ellipsoid).
    base_height : float, default=0.10
        Nominal height ``c`` of a good joint.
    good_scale_std : float, default=0.09
        Relative standard deviation of the natural size variation of good
        joints.
    insufficient_scale_range : (float, float), default=(0.55, 0.85)
        Range of the uniform shrink factor applied to insufficient-solder
        joints (the upper bound overlaps the lower tail of good joints).
    excess_scale_mean : float, default=1.3
        Mean inflation factor of excessive-solder joints.
    excess_scale_std : float, default=0.12
        Standard deviation of the inflation factor (the lower tail overlaps
        the upper tail of good joints).
    excess_bump_range : (int, int), default=(150, 350)
        Range of the number of extra points forming the irregular bump on
        excessive-solder joints.
    normal_dist_mean : float, default=0.65
        Mean centre-to-centre distance to the nearest neighbour joint for
        non-bridging classes.
    normal_dist_std : float, default=0.07
        Standard deviation of that distance.
    bridge_dist_mean : float, default=0.34
        Mean centre distance for bridging joints (abnormally small).
    bridge_dist_std : float, default=0.06
        Standard deviation of the bridging distance (tail overlaps normal
        spacing).
    bridge_points_range : (int, int), default=(80, 180)
        Range of the number of points forming the low solder bridge between
        two joints.
    collapse_depth_range : (float, float), default=(0.28, 0.5)
        Range of the dent depth (fraction of joint height) for
        collapse/void joints.

    Examples
    --------
    >>> gen = SolderJointGenerator(n_points=1500, noise=0.02, seed=42)
    >>> clouds, labels = gen.generate(n_per_class=10)
    >>> X, y = gen.to_features(clouds, labels)
    >>> X.shape
    (50, 7)
    """

    def __init__(
        self,
        n_points: int = 1500,
        noise: float = 0.005,
        seed: Optional[int] = None,
        base_radius: float = 0.12,
        base_height: float = 0.10,
        good_scale_std: float = 0.09,
        insufficient_scale_range: Tuple[float, float] = (0.55, 0.85),
        excess_scale_mean: float = 1.3,
        excess_scale_std: float = 0.12,
        excess_bump_range: Tuple[int, int] = (150, 350),
        normal_dist_mean: float = 0.65,
        normal_dist_std: float = 0.07,
        bridge_dist_mean: float = 0.34,
        bridge_dist_std: float = 0.06,
        bridge_points_range: Tuple[int, int] = (80, 180),
        collapse_depth_range: Tuple[float, float] = (0.28, 0.50),
    ) -> None:
        self.n_points = int(n_points)
        self.noise = float(noise)
        self.seed = seed
        self.base_radius = float(base_radius)
        self.base_height = float(base_height)
        self.good_scale_std = float(good_scale_std)
        self.insufficient_scale_range = tuple(insufficient_scale_range)
        self.excess_scale_mean = float(excess_scale_mean)
        self.excess_scale_std = float(excess_scale_std)
        self.excess_bump_range = tuple(excess_bump_range)
        self.normal_dist_mean = float(normal_dist_mean)
        self.normal_dist_std = float(normal_dist_std)
        self.bridge_dist_mean = float(bridge_dist_mean)
        self.bridge_dist_std = float(bridge_dist_std)
        self.bridge_points_range = tuple(bridge_points_range)
        self.collapse_depth_range = tuple(collapse_depth_range)

    # ------------------------------------------------------------------
    # Low-level geometry
    # ------------------------------------------------------------------
    def _half_ellipsoid(self, rng: np.random.Generator, a: float, c: float) -> np.ndarray:
        """Sample ``n_points`` points uniformly inside a half ellipsoid.

        The ellipsoid has horizontal radius ``a`` and height ``c`` with
        ``z >= 0``; Gaussian measurement noise is added afterwards.
        """
        n = self.n_points
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        rad = rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)  # volume-uniform
        pts = dirs * rad[:, None]
        pts[:, 2] = np.abs(pts[:, 2])                  # upper half
        pts *= np.array([a, a, c])
        pts += rng.normal(0.0, self.noise, pts.shape)  # sensor noise
        pts[:, 2] = np.clip(pts[:, 2], 0.0, None)
        return pts

    # ------------------------------------------------------------------
    # Per-class simulation
    # ------------------------------------------------------------------
    def simulate_sample(
        self, solder_type: str, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float]:
        """Simulate one joint of the given class.

        Returns ``(points[n_points, 3], min_dist)``.  Per-sample parameter
        jitter keeps a realistic overlap between classes.
        """
        if solder_type not in CLASS_TO_IDX:
            raise ValueError(
                f"unknown solder type {solder_type!r}; expected one of {CLASSES}"
            )

        a0, c0 = self.base_radius, self.base_height
        scale = rng.normal(1.0, self.good_scale_std)  # natural size variation

        if solder_type == "good":
            a, c = a0 * scale, c0 * scale
            pts = self._half_ellipsoid(rng, a, c)
            min_dist = rng.normal(self.normal_dist_mean, self.normal_dist_std)

        elif solder_type == "insufficient":
            # shrunk volume; upper bound overlaps the lower tail of "good"
            lo, hi = self.insufficient_scale_range
            s = rng.uniform(lo, hi) * scale
            a, c = a0 * s, c0 * s * rng.uniform(0.8, 1.0)
            pts = self._half_ellipsoid(rng, a, c)
            min_dist = rng.normal(self.normal_dist_mean, self.normal_dist_std)

        elif solder_type == "excessive":
            # inflated volume + irregular bump; lower tail overlaps "good"
            s = rng.normal(self.excess_scale_mean, self.excess_scale_std)
            a, c = a0 * s, c0 * (s + rng.normal(0.25, 0.08))
            pts = self._half_ellipsoid(rng, a, c)
            nb = int(rng.integers(*self.excess_bump_range))
            bump = rng.normal(
                loc=[0.5 * a, 0.0, 0.75 * c],
                scale=[0.3 * a, 0.3 * a, 0.2 * c],
                size=(nb, 3),
            )
            bump[:, 2] = np.clip(bump[:, 2], 0.0, None)
            pts = np.vstack([pts, bump])
            min_dist = rng.normal(self.normal_dist_mean, self.normal_dist_std)

        elif solder_type == "bridging":
            # abnormally small distance to the neighbour + a low solder bridge
            a, c = a0 * scale, c0 * scale
            pts = self._half_ellipsoid(rng, a, c)
            min_dist = rng.normal(self.bridge_dist_mean, self.bridge_dist_std)
            nb = int(rng.integers(*self.bridge_points_range))
            t = rng.uniform(0.3, 1.0, nb)
            bridge_pts = np.stack(
                [
                    t * min_dist * rng.uniform(0.9, 1.0, nb),
                    rng.normal(0, 0.02, nb),
                    np.clip(rng.normal(0.35 * c, 0.1 * c, nb), 0.0, None),
                ],
                axis=1,
            )
            pts = np.vstack([pts, bridge_pts])

        else:  # "collapse"
            # dented top (void / collapsed joint); dent centre is randomly
            # offset because real voids appear at arbitrary positions
            a, c = a0 * scale, c0 * scale
            pts = self._half_ellipsoid(rng, a, c)
            ox, oy = rng.uniform(-0.3, 0.3, 2) * a
            r_xy = np.linalg.norm(pts[:, :2] - np.array([ox, oy]), axis=1)
            lo, hi = self.collapse_depth_range
            dent_depth = c * rng.uniform(lo, hi)
            dent = dent_depth * np.exp(-((r_xy / (0.35 * a)) ** 2))
            pts[:, 2] = np.clip(
                pts[:, 2] - dent * (pts[:, 2] / max(c, 1e-6)), 0.0, None
            )
            min_dist = rng.normal(self.normal_dist_mean, self.normal_dist_std)

        # Fixed point count: prevents voxel-count volume from being biased
        # by sampling density (simulates a fixed scanner resolution).
        if pts.shape[0] > self.n_points:
            idx = rng.choice(pts.shape[0], self.n_points, replace=False)
            pts = pts[idx]
        return pts, float(min_dist)

    # ------------------------------------------------------------------
    # Dataset-level API
    # ------------------------------------------------------------------
    def generate(
        self, n_per_class: int = 250, classes: Optional[Sequence[str]] = None
    ) -> Tuple[List[SolderSample], List[str]]:
        """Generate a balanced dataset of simulated joints.

        Parameters
        ----------
        n_per_class : int, default=250
            Number of samples per class.
        classes : sequence of str, optional
            Subset/order of classes to generate; defaults to all five.

        Returns
        -------
        clouds : list of SolderSample
        labels : list of str
            Class identifier of each sample (parallel to ``clouds``).
        """
        classes = list(classes) if classes is not None else list(CLASSES)
        rng = np.random.default_rng(self.seed)
        clouds: List[SolderSample] = []
        labels: List[str] = []
        for ctype in classes:
            for _ in range(int(n_per_class)):
                pts, md = self.simulate_sample(ctype, rng)
                clouds.append(SolderSample(points=pts, min_dist=md, label=ctype))
                labels.append(ctype)
        return clouds, labels

    def to_features(
        self,
        clouds: Sequence[SolderSample],
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract the 7 geometric features for a list of samples.

        Parameters
        ----------
        clouds : sequence of SolderSample
        labels : sequence of str, optional
            Class identifiers; if omitted, taken from ``sample.label``.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, 7)
            Feature matrix with columns :data:`solderbench.features.FEATURES`.
        y : np.ndarray of int
            Integer labels (0..4, see :data:`CLASS_TO_IDX`).
        """
        if labels is None:
            labels = [s.label for s in clouds]
        rows = [extract_features(s.points, s.min_dist) for s in clouds]
        X = pd.DataFrame(rows, columns=FEATURES)
        y = np.array([CLASS_TO_IDX[l] for l in labels], dtype=int)
        return X, y
