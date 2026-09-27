"""Unit tests for the pure pieces of scripts/scale_realistic_val.py.

Run with: python -m unittest tests.test_scale_realistic_val -v
(from code/business_entity_resolution/)
"""

# lightgbm must be imported before pandas -- see src/model.py's docstring.
import lightgbm  # noqa: F401

import sys
import unittest

# scale_realistic_val.py parses argv at import time (same pattern as scripts/run_pipeline.py --
# CLI flags must become AML_* env vars BEFORE `from src import config` runs anywhere in the
# process, since config.py reads them once at import time). A minimal, valid argv lets this
# module be imported for its pure helpers (loss_buckets, below) without a real dataset -- these
# tests never call run()/main(), so --country/--n-s1's actual values don't matter here.
_saved_argv = sys.argv
sys.argv = ["scale_realistic_val.py", "--country", "India", "--n-s1", "1"]
try:
    from scripts import scale_realistic_val as sv
finally:
    sys.argv = _saved_argv


class TestLossBuckets(unittest.TestCase):
    def test_perfect_prediction_has_zero_loss(self):
        truths = {"S1-1": {"S2-1", "S2-2"}, "S1-2": set()}
        preds = {"S1-1": {"S2-1", "S2-2"}, "S1-2": set()}
        capped = {"S1-1": {"S2-1", "S2-2"}, "S1-2": set()}
        df = sv.loss_buckets(preds, truths, capped)
        self.assertTrue((df["block"] == 0).all())
        self.assertTrue((df["model"] == 0).all())
        self.assertTrue((df["fp_singleton"] == 0).all())
        self.assertTrue((df["fp_extra"] == 0).all())

    def test_blocking_miss_when_true_match_never_in_candidates(self):
        truths, preds, capped = {"S1-1": {"S2-1"}}, {"S1-1": set()}, {"S1-1": set()}   # S2-1 was never a candidate
        row = sv.loss_buckets(preds, truths, capped).iloc[0]
        self.assertGreater(row["block"], 0)
        self.assertEqual(row["model"], 0)

    def test_model_miss_when_true_match_was_a_candidate_but_not_predicted(self):
        truths, preds, capped = {"S1-1": {"S2-1"}}, {"S1-1": set()}, {"S1-1": {"S2-1"}}
        row = sv.loss_buckets(preds, truths, capped).iloc[0]
        self.assertEqual(row["block"], 0)
        self.assertGreater(row["model"], 0)

    def test_false_positive_on_singleton_vs_extra_on_matched(self):
        truths_single = {"S1-1": set()}
        preds_single = {"S1-1": {"S2-9"}}       # wrong match on a true singleton
        capped_single = {"S1-1": {"S2-9"}}
        row = sv.loss_buckets(preds_single, truths_single, capped_single).iloc[0]
        self.assertGreater(row["fp_singleton"], 0)
        self.assertEqual(row["fp_extra"], 0)

        truths_matched = {"S1-1": {"S2-1"}}
        preds_matched = {"S1-1": {"S2-1", "S2-9"}}   # one right, one wrong extra
        capped_matched = {"S1-1": {"S2-1", "S2-9"}}
        row2 = sv.loss_buckets(preds_matched, truths_matched, capped_matched).iloc[0]
        self.assertEqual(row2["fp_singleton"], 0)
        self.assertGreater(row2["fp_extra"], 0)


if __name__ == "__main__":
    unittest.main()
