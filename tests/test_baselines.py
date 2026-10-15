"""Unit tests for solderbench.baselines / evaluate (unittest)."""

import unittest

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from solderbench import (
    S3DARuleClassifier,
    SolderJointGenerator,
    get_baselines,
    get_s3da_tree_proxy,
    run_comparison,
)


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gen = SolderJointGenerator(n_points=400, seed=42)
        clouds, labels = gen.generate(n_per_class=40)
        cls.X, cls.y = gen.to_features(clouds, labels)
        cls.Xtr, cls.Xte, cls.ytr, cls.yte = train_test_split(
            cls.X, cls.y, test_size=0.3, stratify=cls.y, random_state=42
        )

    def test_six_baselines_fit_and_predict(self):
        baselines = get_baselines(seed=42)
        self.assertEqual(len(baselines), 6)
        for no, name, clf, ours in baselines:
            with self.subTest(method=name):
                clf.fit(self.Xtr.values, self.ytr)
                pred = clf.predict(self.Xte.values)
                self.assertEqual(pred.shape, self.yte.shape)
                self.assertTrue(set(np.unique(pred)) <= set(np.unique(self.y)))

    def test_exactly_one_ours_method(self):
        ours = [name for _, name, _, flag in get_baselines() if flag]
        self.assertEqual(len(ours), 1)
        self.assertIn("RandomForest", ours[0])

    def test_random_forest_accuracy_in_reasonable_range(self):
        baselines = dict((name, clf) for _, name, clf, _ in get_baselines())
        rf = baselines["RandomForest (Ours)"]
        rf.fit(self.Xtr.values, self.ytr)
        acc = accuracy_score(self.yte, rf.predict(self.Xte.values))
        self.assertGreaterEqual(acc, 0.80)
        self.assertLessEqual(acc, 1.00)

    def test_s3da_rule_is_label_free_and_deterministic(self):
        """The original S3DA rule must not depend on class labels."""
        Xtr, Xte = self.Xtr.values, self.Xte.values
        a = S3DARuleClassifier().fit(Xtr, self.ytr).predict(Xte)
        b = S3DARuleClassifier().fit(Xtr, None).predict(Xte)
        np.testing.assert_array_equal(a, b)
        # The published rule thresholds volume only: it can never emit the
        # bridging (3) or collapse (4) class.
        self.assertTrue(set(np.unique(a)) <= {0, 1, 2})
        # Deterministic: two independent fits give identical predictions.
        c = S3DARuleClassifier().fit(Xtr, self.ytr).predict(Xte)
        np.testing.assert_array_equal(a, c)

    def test_s3da_tree_proxy_available_for_reference(self):
        clf = get_s3da_tree_proxy(seed=42)
        clf.fit(self.Xtr.values, self.ytr)
        self.assertEqual(clf.predict(self.Xte.values).shape, self.yte.shape)

    def test_run_comparison_returns_dataframe(self):
        # Small subset to keep the CV loop fast.
        X = self.X.groupby(self.y, group_keys=False).head(20)
        y = self.y[X.index]
        df = run_comparison(X, y, test_size=0.3, seed=42)
        self.assertEqual(len(df), 6)
        for col in ("Accuracy", "Precision", "Recall", "F1",
                    "CV_mean", "CV_std"):
            self.assertIn(col, df.columns)
            self.assertTrue(np.all(np.isfinite(df[col])))


if __name__ == "__main__":
    unittest.main()
