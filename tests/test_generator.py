"""Unit tests for solderbench.generator (standard-library unittest)."""

import unittest

import numpy as np

from solderbench import CLASSES, CLASS_TO_IDX, SolderJointGenerator


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = SolderJointGenerator(n_points=300, seed=42)

    def test_output_shapes_and_class_count(self):
        n_per_class = 5
        clouds, labels = self.gen.generate(n_per_class=n_per_class)
        self.assertEqual(len(clouds), n_per_class * len(CLASSES))
        self.assertEqual(len(labels), n_per_class * len(CLASSES))
        self.assertEqual(set(labels), set(CLASSES))
        for sample in clouds:
            self.assertEqual(sample.points.shape, (300, 3))
            self.assertTrue(np.all(np.isfinite(sample.points)))
            self.assertGreater(sample.min_dist, 0.0)
        # balanced classes
        for c in CLASSES:
            self.assertEqual(labels.count(c), n_per_class)

    def test_labels_match_samples(self):
        clouds, labels = self.gen.generate(n_per_class=3)
        for sample, label in zip(clouds, labels):
            self.assertEqual(sample.label, label)
            self.assertIn(label, CLASS_TO_IDX)

    def test_same_seed_fully_reproducible(self):
        g1 = SolderJointGenerator(n_points=300, seed=123)
        g2 = SolderJointGenerator(n_points=300, seed=123)
        clouds1, labels1 = g1.generate(n_per_class=4)
        clouds2, labels2 = g2.generate(n_per_class=4)
        self.assertEqual(labels1, labels2)
        for s1, s2 in zip(clouds1, clouds2):
            np.testing.assert_array_equal(s1.points, s2.points)
            self.assertEqual(s1.min_dist, s2.min_dist)

    def test_different_seed_differs(self):
        g1 = SolderJointGenerator(n_points=300, seed=1)
        g2 = SolderJointGenerator(n_points=300, seed=2)
        clouds1, _ = g1.generate(n_per_class=3)
        clouds2, _ = g2.generate(n_per_class=3)
        identical = all(
            np.array_equal(s1.points, s2.points)
            for s1, s2 in zip(clouds1, clouds2)
        )
        self.assertFalse(identical)

    def test_fixed_point_count_for_defects_with_extra_points(self):
        # excessive / bridging add extra points internally but must still
        # return exactly n_points (no sampling-density leakage).
        rng = np.random.default_rng(0)
        for ctype in ("excessive", "bridging"):
            pts, _ = self.gen.simulate_sample(ctype, rng)
            self.assertEqual(pts.shape[0], 300)

    def test_unknown_type_raises(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            self.gen.simulate_sample("nonexistent_class", rng)


if __name__ == "__main__":
    unittest.main()
