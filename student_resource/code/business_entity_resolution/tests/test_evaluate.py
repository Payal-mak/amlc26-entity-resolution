"""Unit tests for src.evaluate -- the exact official-metric reimplementation.

Run with: python -m unittest tests.test_evaluate -v
(from code/business_entity_resolution/)
"""

import unittest

from src import evaluate


class TestFBetaScore(unittest.TestCase):
    def test_worked_example_from_problem_statement(self):
        # pred = [S2-00047, S2-00193, S3-00812], truth = [S2-00047, S3-00812]
        # precision = 2/3, recall = 1.0 -> F0.5 = 0.714 (problem statement).
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        truth = {"S2-00047", "S3-00812"}
        score = evaluate.f_beta_score(pred, truth)
        self.assertAlmostEqual(score, 0.714, places=3)

    def test_empty_pred_empty_truth_is_perfect_singleton(self):
        self.assertEqual(evaluate.f_beta_score(set(), set()), 1.0)

    def test_nonempty_pred_empty_truth_is_zero(self):
        self.assertEqual(evaluate.f_beta_score({"S2-1"}, set()), 0.0)

    def test_empty_pred_nonempty_truth_is_zero(self):
        self.assertEqual(evaluate.f_beta_score(set(), {"S2-1"}), 0.0)

    def test_perfect_match_is_one(self):
        ids = {"S2-1", "S3-2"}
        self.assertEqual(evaluate.f_beta_score(set(ids), set(ids)), 1.0)

    def test_disjoint_nonempty_sets_is_zero(self):
        self.assertEqual(evaluate.f_beta_score({"S2-1"}, {"S2-2"}), 0.0)

    def test_precision_weighted_more_than_recall(self):
        # One false positive should hurt more than one false negative, since
        # F0.5 weights precision 2x over recall.
        truth = {"S2-1", "S2-2", "S2-3", "S2-4"}
        extra_fp = evaluate.f_beta_score(truth | {"S2-5"}, truth)  # 4/5 precision, 1.0 recall
        missing_fn = evaluate.f_beta_score(truth - {"S2-4"}, truth)  # 1.0 precision, 3/4 recall
        self.assertLess(extra_fp, missing_fn)


class TestMacroFBeta(unittest.TestCase):
    def test_macro_average_across_entities(self):
        truths = {
            "S1-1": {"S2-1"},          # perfect -> 1.0
            "S1-2": set(),             # true singleton, correctly empty -> 1.0
            "S1-3": {"S2-3"},          # missed entirely -> 0.0
        }
        preds = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": set(),
        }
        self.assertAlmostEqual(evaluate.macro_f_beta(preds, truths), 2 / 3, places=6)

    def test_missing_prediction_key_treated_as_empty(self):
        truths = {"S1-1": {"S2-1"}}
        preds = {}  # S1-1 absent -> treated as empty prediction -> 0.0
        self.assertEqual(evaluate.macro_f_beta(preds, truths), 0.0)

    def test_empty_truths_returns_zero_not_error(self):
        self.assertEqual(evaluate.macro_f_beta({}, {}), 0.0)


class TestSingletonAccuracy(unittest.TestCase):
    def test_basic(self):
        truths = {"S1-1": set(), "S1-2": set(), "S1-3": {"S2-1"}}
        preds = {"S1-1": set(), "S1-2": {"S2-9"}, "S1-3": {"S2-1"}}
        # two true singletons, one predicted correctly empty -> 1/2
        self.assertEqual(evaluate.singleton_accuracy(preds, truths), 0.5)

    def test_no_singletons_returns_none(self):
        truths = {"S1-1": {"S2-1"}}
        self.assertIsNone(evaluate.singleton_accuracy({}, truths))


class TestMicroPrecisionRecall(unittest.TestCase):
    def test_basic(self):
        truths = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}}
        preds = {"S1-1": {"S2-1"}, "S1-2": {"S3-1", "S3-2"}}
        result = evaluate.micro_precision_recall(preds, truths)
        # tp = 2 (S2-1, S3-1), n_pred = 3, n_truth = 3
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["n_pred"], 3)
        self.assertEqual(result["n_truth"], 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)


class TestPerGroupMacroFBeta(unittest.TestCase):
    def test_grouping_by_country(self):
        truths = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
        preds = {"S1-1": {"S2-1"}, "S1-2": set()}
        group_of = {"S1-1": "US", "S1-2": "India"}
        report = evaluate.per_group_macro_f_beta(preds, truths, group_of)
        self.assertEqual(report["US"]["f_beta"], 1.0)
        self.assertEqual(report["India"]["f_beta"], 0.0)
        self.assertEqual(report["US"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
