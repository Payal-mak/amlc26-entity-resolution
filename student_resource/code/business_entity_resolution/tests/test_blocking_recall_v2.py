"""Unit tests for the recall-v2 blocks (src.blocking): B_tfidf (char-3-gram
top-K) and B_address (rare address tokens + house-number bonus). Tiny
synthetic data, in-memory DuckDB -- correctness only, not scale (see
scripts/recall_v2_b2_df_sweep.py / recall_v2_tfidf_subset_test.py /
recall_v2_address_block_eval.py for the real validation-slice measurements).

Run with: python -m unittest tests.test_blocking_recall_v2 -v
(from code/business_entity_resolution/)
"""

import unittest

import duckdb
import pandas as pd

from src import blocking


def _con():
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit = '1GB'")
    return con


class TestBlockTfidfCharNgram(unittest.TestCase):
    def test_catches_domain_glued_name(self):
        # "butlerhall.com" (candidate) is ONE token after normalization --
        # structurally invisible to B1 (exact match) and B2 (shared token,
        # since "butlerhall" and "com" never appear as separate tokens in
        # the S1 side's "butler hall inc"). Char-3-gram cosine still shares
        # plenty of trigrams ("but", "utl", "tle", "ler", "hal", "all"...).
        con = _con()
        s1 = pd.DataFrame(
            {"entity_id": ["S1-1", "S1-2"], "country": ["US", "US"],
             "name_core": ["butler hall", "acme"]}
        )
        cand = pd.DataFrame(
            {"entity_id": ["S2-1", "S2-2", "S2-3"], "country": ["US", "US", "US"],
             "name_core": ["butlerhallcom", "unrelated widgets", "acme"]}
        )
        con.register("s1v", s1)
        con.register("candv", cand)
        con.execute("CREATE TABLE s1t AS SELECT * FROM s1v")
        con.execute("CREATE TABLE candt AS SELECT * FROM candv")

        out = blocking.block_b_tfidf_char_ngram(con, "s1t", "candt", top_k=5, min_similarity=0.1, max_df=1.0)
        pairs = set(map(tuple, con.execute(f"SELECT s1_id, cand_id FROM {out}").fetchdf().values))
        self.assertIn(("S1-1", "S2-1"), pairs)
        con.close()

    def test_empty_input_does_not_crash(self):
        con = _con()
        empty = pd.DataFrame({"entity_id": [], "country": [], "name_core": []})
        con.register("s1v", empty)
        con.register("candv", empty)
        con.execute("CREATE TABLE s1t AS SELECT * FROM s1v")
        con.execute("CREATE TABLE candt AS SELECT * FROM candv")
        out = blocking.block_b_tfidf_char_ngram(con, "s1t", "candt")
        self.assertEqual(con.execute(f"SELECT COUNT(*) FROM {out}").fetchone()[0], 0)
        con.close()

    def test_respects_top_k_cap(self):
        con = _con()
        n = 30
        s1 = pd.DataFrame({"entity_id": ["S1-1"], "country": ["US"], "name_core": ["acme corp"]})
        cand = pd.DataFrame(
            {"entity_id": [f"S2-{i}" for i in range(n)], "country": ["US"] * n,
             "name_core": ["acme corp variant " + str(i) for i in range(n)]}
        )
        con.register("s1v", s1)
        con.register("candv", cand)
        con.execute("CREATE TABLE s1t AS SELECT * FROM s1v")
        con.execute("CREATE TABLE candt AS SELECT * FROM candv")
        out = blocking.block_b_tfidf_char_ngram(con, "s1t", "candt", top_k=5, min_similarity=0.0, max_df=1.0)
        n_pairs = con.execute(f"SELECT COUNT(*) FROM {out}").fetchone()[0]
        self.assertEqual(n_pairs, 5)
        con.close()


class TestBlockAddressRareTokens(unittest.TestCase):
    def test_catches_dba_same_address_different_name(self):
        # Completely unrelated names, but same rare street token + same
        # house number -- exactly the case B1/B2/B_tfidf (all name-based)
        # structurally cannot reach.
        con = _con()
        s1 = pd.DataFrame(
            {"entity_id": ["S1-1"], "country": ["US"],
             "business_address": ["221 Zenithbrook Lane, Austin"]}
        )
        cand = pd.DataFrame(
            {"entity_id": ["S2-1", "S2-2"], "country": ["US", "US"],
             "business_address": ["221 Zenithbrook Lane, Austin", "999 Main St, Reno"]}
        )
        con.register("s1v", s1)
        con.register("candv", cand)
        blocking.register_house_number(con, "s1v", "s1h")
        blocking.register_house_number(con, "candv", "candh")
        blocking.register_address_tokens(con, "s1v", "s1at")
        blocking.register_address_tokens(con, "candv", "candat")

        orig_min_df = blocking.ADDRESS_MIN_TOKEN_DF
        blocking.ADDRESS_MIN_TOKEN_DF = 1  # tiny synthetic corpus: relax the >=3-occurrence noise filter
        try:
            out = blocking.block_b_address_rare_tokens(con, "s1h", "candh", "s1at", "candat")
        finally:
            blocking.ADDRESS_MIN_TOKEN_DF = orig_min_df
        pairs = set(map(tuple, con.execute(f"SELECT s1_id, cand_id FROM {out}").fetchdf().values))
        self.assertIn(("S1-1", "S2-1"), pairs)
        self.assertNotIn(("S1-1", "S2-2"), pairs)
        con.close()

    def test_extract_house_number(self):
        self.assertEqual(blocking.extract_house_number("221B Baker Street"), "221")
        self.assertIsNone(blocking.extract_house_number("Baker Street, no number"))
        self.assertIsNone(blocking.extract_house_number(None))


if __name__ == "__main__":
    unittest.main()
