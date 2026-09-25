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


class TestBlockGeoDfCap(unittest.TestCase):
    """block_b_geo's max_cand_token_df cap (recall-v2, added after a real
    Kaggle full-train-scale OOM -- see config.GEO_MAX_CAND_TOKEN_DF's
    docstring). Excludes a candidate from the JOIN entirely if its own
    chosen geo_token's candidate-side document frequency is above the cap --
    same principle as B2_MAX_CAND_TOKEN_DF, checked in the join condition so
    the expensive fan-out never happens, not just filtered out afterward.
    """

    def _setup(self, con):
        s1 = pd.DataFrame(
            {
                "entity_id": ["S1-rare", "S1-common"],
                "country": ["US", "US"],
                "business_address": ["1 Raretown Road", "2 Commonplaza Road"],
            }
        )
        # "raretown" appears exactly 3x on the candidate side (passes
        # MIN_GEO_TOKEN_DF=3); "commonplaza" appears 4x -- both qualify as
        # SOME record's rarest available token, but commonplaza is the one
        # a tight cap should exclude.
        cand = pd.DataFrame(
            {
                "entity_id": [f"S2-{i}" for i in range(7)],
                "country": ["US"] * 7,
                "business_address": (
                    ["1 Raretown Road"] * 3 + ["2 Commonplaza Road"] * 4
                ),
            }
        )
        con.register("s1v", s1)
        con.register("candv", cand)
        # register_geo_tokens computes each token's df WITHIN the table it's
        # given -- MIN_GEO_TOKEN_DF=3 (the real default) would exclude every
        # token on this 2-row S1 table outright (nothing repeats 3x there),
        # unrelated to the cap this test is actually about.
        orig_min = blocking.MIN_GEO_TOKEN_DF
        blocking.MIN_GEO_TOKEN_DF = 1
        try:
            blocking.register_geo_tokens(con, "s1v", "s1geo")
            blocking.register_geo_tokens(con, "candv", "candgeo")
        finally:
            blocking.MIN_GEO_TOKEN_DF = orig_min

    def test_tight_cap_excludes_common_token_but_keeps_rare_one(self):
        con = _con()
        self._setup(con)
        out = blocking.block_b_geo(con, "s1geo", "candgeo", max_cand_token_df=3)
        pairs = set(map(tuple, con.execute(f"SELECT s1_id, cand_id FROM {out}").fetchdf().values))
        rare_pairs = {p for p in pairs if p[0] == "S1-rare"}
        common_pairs = {p for p in pairs if p[0] == "S1-common"}
        self.assertEqual(len(rare_pairs), 3)  # raretown, df=3 <= cap
        self.assertEqual(len(common_pairs), 0)  # commonplaza, df=4 > cap -- excluded
        con.close()

    def test_loose_cap_keeps_both(self):
        con = _con()
        self._setup(con)
        out = blocking.block_b_geo(con, "s1geo", "candgeo", max_cand_token_df=10)
        pairs = set(map(tuple, con.execute(f"SELECT s1_id, cand_id FROM {out}").fetchdf().values))
        common_pairs = {p for p in pairs if p[0] == "S1-common"}
        self.assertEqual(len(common_pairs), 4)  # df=4 <= cap=10 -- now included
        con.close()


if __name__ == "__main__":
    unittest.main()
