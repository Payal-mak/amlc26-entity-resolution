"""Unit tests for src.decide.

Run with: python -m unittest tests.test_decide -v
(from code/business_entity_resolution/)
"""

import unittest

import pandas as pd

from src import decide


class TestOneToOne(unittest.TestCase):
    def test_keeps_highest_scoring_s1_per_candidate(self):
        df = pd.DataFrame(
            {
                "s1_id": ["S1-1", "S1-2", "S1-3"],
                "cand_id": ["S2-1", "S2-1", "S2-2"],
                "proba": [0.6, 0.9, 0.3],
            }
        )
        out = decide.one_to_one(df)
        # S2-1 should go to S1-2 (0.9 > 0.6), S2-2 stays with S1-3
        row = out[out["cand_id"] == "S2-1"].iloc[0]
        self.assertEqual(row["s1_id"], "S1-2")
        self.assertEqual(len(out), 2)


class TestExpectedFBetaSubsetSelection(unittest.TestCase):
    def test_high_confidence_candidate_is_kept(self):
        df = pd.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-1"], "proba": [0.95]})
        result = decide.expected_f_beta_subset_selection(df)
        self.assertEqual(result["S1-1"], {"S2-1"})

    def test_low_confidence_candidate_is_dropped(self):
        df = pd.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-1"], "proba": [0.02]})
        result = decide.expected_f_beta_subset_selection(df)
        self.assertEqual(result["S1-1"], set())

    def test_keeps_multiple_high_confidence_candidates(self):
        df = pd.DataFrame(
            {
                "s1_id": ["S1-1", "S1-1", "S1-1"],
                "cand_id": ["S2-1", "S2-2", "S3-1"],
                "proba": [0.9, 0.85, 0.05],
            }
        )
        result = decide.expected_f_beta_subset_selection(df)
        self.assertEqual(result["S1-1"], {"S2-1", "S2-2"})


class TestGlobalThreshold(unittest.TestCase):
    def test_apply_threshold_filters_correctly(self):
        df = pd.DataFrame(
            {
                "s1_id": ["S1-1", "S1-1", "S1-2"],
                "cand_id": ["S2-1", "S2-2", "S2-3"],
                "proba": [0.8, 0.3, 0.9],
            }
        )
        result = decide.apply_global_threshold(df, 0.5)
        self.assertEqual(result, {"S1-1": {"S2-1"}, "S1-2": {"S2-3"}})

    def test_tune_picks_threshold_that_maximizes_f_beta(self):
        df = pd.DataFrame(
            {
                "s1_id": ["S1-1", "S1-2"],
                "cand_id": ["S2-1", "S2-2"],
                "proba": [0.9, 0.3],
            }
        )
        # threshold=0.5: S1-1 predicts its true match (1.0), S1-2 stays
        # empty and is a true singleton (1.0) -> macro 1.0.
        # threshold=0.95: neither predicts -> S1-1 misses its match (0.0),
        # S1-2 correct (1.0) -> macro 0.5. 0.5 should clearly win.
        truths = {"S1-1": {"S2-1"}, "S1-2": set()}
        best_t, best_f = decide.tune_global_threshold(df, truths, thresholds=[0.5, 0.95])
        self.assertAlmostEqual(best_t, 0.5)
        self.assertEqual(best_f, 1.0)


class TestDecideEndToEnd(unittest.TestCase):
    def test_returns_predictions_for_every_eval_id_including_zero_candidate(self):
        df = pd.DataFrame(
            {
                "s1_id": ["S1-1", "S1-2"],
                "cand_id": ["S2-1", "S2-2"],
                "proba": [0.9, 0.9],
            }
        )
        eval_ids = {"S1-1", "S1-2", "S1-3"}  # S1-3 has no rows at all
        preds, policy = decide.decide(df, eval_ids)
        self.assertEqual(set(preds.keys()), eval_ids)
        self.assertEqual(preds["S1-3"], set())


if __name__ == "__main__":
    unittest.main()
