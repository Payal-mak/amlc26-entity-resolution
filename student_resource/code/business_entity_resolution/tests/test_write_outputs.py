"""Unit tests for the streaming candidate writer in src.write_outputs.

Run with: python -m unittest tests.test_write_outputs -v
(from code/business_entity_resolution/)
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import write_outputs


class TestCandidatePairsFromParquet(unittest.TestCase):
    def test_one_row_per_required_id_sorted_lists_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"s1_id": ["S1-2", "S1-1", "S1-1"], "cand_id": ["S3-9", "S2-5", "S2-1"]}).to_parquet(d / "capped_India.parquet")
            pd.DataFrame({"s1_id": ["S1-4"], "cand_id": ["S2-7"]}).to_parquet(d / "capped_France.parquet")
            out = d / "candidate_pairs.tsv"
            n = write_outputs.write_candidate_pairs_from_parquet((d / "capped_*.parquet").as_posix(), {"S1-1", "S1-2", "S1-3", "S1-4"}, out)
            self.assertEqual(n, 3)
            lines = out.read_text(encoding="utf-8").split("\n")
            self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
            self.assertEqual(lines[1:5], ["S1-1\tS2-1,S2-5", "S1-2\tS3-9", "S1-3\t", "S1-4\tS2-7"])  # S1-3 has no candidates -> empty
            self.assertEqual(lines[5], "")  # single trailing newline

    def test_matches_the_dict_writer(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"s1_id": ["S1-1", "S1-1"], "cand_id": ["S3-2", "S2-9"]}).to_parquet(d / "capped_US.parquet")
            a, b = d / "a.tsv", d / "b.tsv"
            write_outputs.write_candidate_pairs_from_parquet((d / "*.parquet").as_posix(), {"S1-1", "S1-2"}, a)
            write_outputs.write_candidate_pairs({"S1-1": {"S3-2", "S2-9"}}, {"S1-1", "S1-2"}, b)
            self.assertEqual(a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
