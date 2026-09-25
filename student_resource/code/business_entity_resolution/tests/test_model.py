"""Unit tests for the two-stage model helpers in src.model.

Run with: python -m unittest tests.test_model -v
(from code/business_entity_resolution/)
"""

# lightgbm must be imported before pandas (see src/model.py's docstring).
import lightgbm  # noqa: F401

import unittest

import numpy as np

from src import model


class TestStage2ContextFeatures(unittest.TestCase):
    def test_ranks_gap_and_counts(self):
        s1 = ["A", "A", "A", "B", "B"]
        cand = ["c1", "c2", "c3", "c1", "c4"]
        p1 = np.array([0.9, 0.6, 0.1, 0.95, 0.2])
        f = model.stage2_context_features(s1, cand, p1)
        self.assertEqual(f.shape, (5, len(model.STAGE2_EXTRA_COLUMNS)))
        rank, gap, rrank, n_above = f.T
        self.assertEqual(rank.tolist(), [1, 2, 3, 1, 2])                       # within each S1, best = 1
        np.testing.assert_allclose(gap, [0.0, 0.3, 0.8, 0.0, 0.75], atol=1e-6)  # best-in-S1 minus this
        self.assertEqual(rrank.tolist(), [2, 1, 1, 1, 1])                       # c1 is claimed by A (0.9) and B (0.95): B wins
        self.assertEqual(n_above.tolist(), [2, 2, 2, 1, 1])                     # A has 2 candidates > 0.5, B has 1

    def test_deterministic_under_ties(self):
        s1, cand, p1 = ["A", "A"], ["c1", "c2"], np.array([0.5, 0.5])
        a = model.stage2_context_features(s1, cand, p1)
        b = model.stage2_context_features(s1, cand, p1)
        np.testing.assert_array_equal(a, b)


def _synthetic(n_groups=120, per_group=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    s1 = np.repeat([f"S1-{i}" for i in range(n_groups)], per_group)
    cand = np.array([f"S2-{i}" for i in range(n)])
    X = rng.normal(size=(n, 4)).astype(np.float32)
    return X, s1, cand, rng


class TestTwoStage(unittest.TestCase):
    params = dict(n_estimators=40, num_leaves=7, min_child_samples=5, n_jobs=1)

    def test_oof_is_aligned_and_uses_signal(self):
        X, s1, cand, rng = _synthetic()
        y = (X[:, 0] + 0.3 * rng.normal(size=len(X)) > 0.8).astype(np.float32)
        p2, info = model.train_two_stage_oof(X, y, s1, s1, cand, ["a", "b", "c", "d"], n_folds=4, params=self.params)
        self.assertEqual(p2.shape, (len(X),))
        self.assertTrue(((p2 >= 0) & (p2 <= 1)).all())
        self.assertEqual(len(info["importance"]), 4 + len(model.STAGE2_EXTRA_COLUMNS))
        from sklearn.metrics import roc_auc_score
        self.assertGreater(roc_auc_score(y, p2), 0.8)

    def test_no_leakage_on_random_labels(self):
        # Labels unrelated to the features: an honest OOF pipeline cannot beat chance.
        # If stage 2 saw in-sample stage-1 probabilities it would score far above 0.5.
        from sklearn.metrics import roc_auc_score
        X, s1, cand, rng = _synthetic(n_groups=200, seed=1)
        y = (rng.random(len(X)) < 0.3).astype(np.float32)
        p2, _ = model.train_two_stage_oof(X, y, s1, s1, cand, list("abcd"), n_folds=4, params=self.params)
        self.assertLess(abs(roc_auc_score(y, p2) - 0.5), 0.08)

    def test_final_models_predict_unseen_rows(self):
        X, s1, cand, rng = _synthetic()
        y = (X[:, 0] > 0.5).astype(np.float32)
        p1_oof, _, _ = model.train_oof(X, y, s1, n_folds=4, params=self.params)
        final = model.fit_two_stage_final(X, y, s1, cand, p1_oof, params=self.params)
        self.assertEqual(final["kind"], "two_stage")
        Xt, s1t, candt, _ = _synthetic(n_groups=30, seed=5)
        p = model.predict_two_stage_final(final, Xt, s1t, candt)
        self.assertEqual(p.shape, (len(Xt),))
        self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__":
    unittest.main()
