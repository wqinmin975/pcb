"""Unit tests for solderbench.features (standard-library unittest)."""

import unittest

import numpy as np
from scipy import stats

from solderbench import FEATURES, SolderJointGenerator, extract_features


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One shared dataset keeps the statistical test fast.  The full
        # point count (1500) is used because the voxel-count volume only
        # separates class scales well at the benchmark sampling density.
        cls.gen = SolderJointGenerator(n_points=1500, seed=7)
        cls.clouds, cls.labels = cls.gen.generate(n_per_class=25)
        cls.rows = [
            extract_features(s.points, s.min_dist) for s in cls.clouds
        ]

    def test_feature_dict_keys(self):
        feats = self.rows[0]
        self.assertEqual(set(feats.keys()), set(FEATURES))

    def test_features_are_finite(self):
        for feats in self.rows:
            for name, value in feats.items():
                self.assertTrue(
                    np.isfinite(value), f"{name} is not finite: {value}"
                )

    def test_volume_positive(self):
        for feats in self.rows:
            self.assertGreater(feats["vol"], 0.0)

    def test_convex_vol_ratio_in_unit_interval(self):
        for feats in self.rows:
            r = feats["convex_vol_ratio"]
            self.assertGreater(r, 0.0)
            self.assertLessEqual(r, 1.0 + 1e-9)

    def test_insufficient_volume_significantly_smaller_than_good(self):
        vols = {"good": [], "insufficient": []}
        for feats, label in zip(self.rows, self.labels):
            if label in vols:
                vols[label].append(feats["vol"])
        good = np.array(vols["good"])
        less = np.array(vols["insufficient"])
        # Welch's t-test: insufficient-solder volume must be significantly
        # smaller than good-joint volume.
        t_stat, p_value = stats.ttest_ind(less, good, equal_var=False)
        self.assertLess(t_stat, 0.0)  # insufficient < good
        self.assertLess(less.mean(), good.mean())
        self.assertLess(p_value, 0.01)
        # Effect size sanity check: at least a 15% mean reduction.
        self.assertLess(less.mean(), 0.85 * good.mean())


if __name__ == "__main__":
    unittest.main()
