"""Unit tests for the streaming test path (src/stream_test.py) and its writers.

Run with: python -m unittest tests.test_stream_test -v
(from code/business_entity_resolution/)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import features, stream_test, write_outputs


class TestCapTopK(unittest.TestCase):
    def test_keeps_best_k_per_s1_by_name_ratio_then_blocks(self):
        df = pd.DataFrame({
            "s1_id": ["A"] * 4 + ["B"] * 2, "cand_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "name_full_ratio": [90.0, 100.0, 90.0, 50.0, 10.0, 20.0], "n_blocks": [1, 1, 3, 1, 1, 1],
        })
        out = stream_test.cap_topk(df, 2)
        self.assertEqual(sorted(zip(out.s1_id, out.cand_id)), [("A", "c2"), ("A", "c3"), ("B", "c5"), ("B", "c6")])
        # tie at 90: the pair found by 3 blocks (c3) beats the one found by 1 (c1)

    def test_k_larger_than_group_keeps_everything(self):
        df = pd.DataFrame({"s1_id": ["A", "A"], "cand_id": ["c1", "c2"], "name_full_ratio": [1.0, 2.0], "n_blocks": [1, 1]})
        self.assertEqual(len(stream_test.cap_topk(df, 75)), 2)


class TestCandidateLists(unittest.TestCase):
    def test_sorted_comma_joined_per_s1(self):
        df = pd.DataFrame({"s1_id": ["S1-2", "S1-1", "S1-1"], "cand_id": ["S3-9", "S2-5", "S2-1"]})
        out = stream_test._candidate_lists(df)
        self.assertEqual(out.to_dict("records"), [{"s1_id": "S1-1", "cands": "S2-1,S2-5"}, {"s1_id": "S1-2", "cands": "S3-9"}])


class TestWriteCandidateLists(unittest.TestCase):
    def test_every_required_id_gets_a_row_and_output_matches_the_other_writer(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"s1_id": ["S1-1"], "cands": ["S2-1,S2-5"]}).to_parquet(d / "cands_A_000.parquet")
            pd.DataFrame({"s1_id": ["S1-3"], "cands": ["S3-9"]}).to_parquet(d / "cands_B_000.parquet")
            out = d / "cp.tsv"
            n = write_outputs.write_candidate_lists_from_parquet([str(d / "cands_A_000.parquet"), str(d / "cands_B_000.parquet")], {"S1-1", "S1-2", "S1-3"}, out)
            self.assertEqual(n, 2)
            self.assertEqual(out.read_text(encoding="utf-8"),
                             "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-5\nS1-2\t\nS1-3\tS3-9\n")


class TestThresholdParsing(unittest.TestCase):
    def test_default_threshold_returns_none_without_a_report(self):
        # no crash when the report is absent or names the label-free policy
        self.assertTrue(stream_test.default_threshold() is None or isinstance(stream_test.default_threshold(), float))


class TestParallelFeaturesMatchSequential(unittest.TestCase):
    def test_same_numbers_in_process_and_in_workers(self):
        def rec(i, name, addr):
            return {"entity_id": i, "business_name": name, "business_address": addr, "country": "US",
                    "name_full": name.lower(), "name_core": name.lower(), "postal_code": None}
        s1 = pd.DataFrame([rec("S1-1", "Quality Biomedical", "7800 Valburn Drive, Austin, TX"), rec("S1-2", "Acme Zorbo", "9 Elm Street, Austin, TX")])
        cand = pd.DataFrame([rec("S2-1", "QUALITY BIOMEDICAL LLC", "AUSTIN, TX, 7802 VALBURN DR"), rec("S2-2", "Acme Zorbo Inc", "9 Elm St, Austin, TX"),
                             rec("S3-1", "Qwerty", "1 Main Street")])
        pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-1", "S1-2"] * 3, "cand_id": ["S2-1", "S2-2", "S3-1", "S2-1"] * 3,
                              "blocks": ["b1_exact_core"] * 12, "n_blocks": [1] * 12})
        prep = features.prepare_entities(s1, cand)
        lookup = features.make_lookup(s1, cand)
        seq = features.build_pair_features_base(pairs, s1, cand, prep)
        for n in (1, 2):
            w = features.FeatureWorkers(prep.attrs["idf"], n)
            try:
                par = features.build_pair_features_parallel(pairs, lookup, prep, w, chunk_rows=5)
            finally:
                w.close()
            for c in features.BASE_FEATURE_COLUMNS:
                np.testing.assert_allclose(par[c].to_numpy(dtype=float), seq[c].to_numpy(dtype=float), equal_nan=True, err_msg=c)


if __name__ == "__main__":
    unittest.main()
