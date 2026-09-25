"""Unit tests for src.features (pair-level helpers).

Run with: python -m unittest tests.test_features -v
(from code/business_entity_resolution/)
"""

import unittest

import numpy as np
import pandas as pd

from src import features


class TestHouseNumberRelation(unittest.TestCase):
    rel = staticmethod(features.house_number_relation)

    def test_both_missing(self):
        self.assertEqual(self.rel((), ()), features.HN_BOTH_MISSING)

    def test_one_missing(self):
        self.assertEqual(self.rel(("13202",), ()), features.HN_ONE_MISSING)
        self.assertEqual(self.rel((), ("5",)), features.HN_ONE_MISSING)

    def test_equal_if_any_run_shared(self):
        self.assertEqual(self.rel(("453",), ("453", "1")), features.HN_EQUAL)

    def test_prefix_or_suffix(self):
        self.assertEqual(self.rel(("6104",), ("104",)), features.HN_PREFIX_SUFFIX)   # dropped leading digit
        self.assertEqual(self.rel(("16549",), ("1654",)), features.HN_PREFIX_SUFFIX)  # dropped trailing digit
        self.assertEqual(self.rel(("1205",), ("120",)), features.HN_PREFIX_SUFFIX)

    def test_single_digit_prefix_is_not_enough(self):
        self.assertEqual(self.rel(("12",), ("1",)), features.HN_CONFLICT)

    def test_one_edit(self):
        self.assertEqual(self.rel(("7800",), ("7802",)), features.HN_ONE_EDIT)
        self.assertEqual(self.rel(("1234",), ("12934",)), features.HN_ONE_EDIT)  # inserted in the middle

    def test_conflict(self):
        self.assertEqual(self.rel(("3153",), ("3490",)), features.HN_CONFLICT)

    def test_dropped_leading_digit_kept_as_raw_string(self):
        self.assertEqual(self.rel(("302",), ("02",)), features.HN_PREFIX_SUFFIX)  # "H.no 02" vs "No.302"
        self.assertEqual(self.rel(("604",), ("04",)), features.HN_PREFIX_SUFFIX)

    def test_leading_zeros_ignored_for_equality(self):
        self.assertEqual(self.rel(("03153",), ("3153",)), features.HN_EQUAL)


class TestHouseNumberSignals(unittest.TestCase):
    sig = staticmethod(features.house_number_signals)

    def test_small_difference(self):
        rel, small, mind, sfx = self.sig((("7800", ""),), (("7802", ""),))
        self.assertEqual((rel, small, sfx), (features.HN_ONE_EDIT, 1.0, 0.0))
        self.assertAlmostEqual(mind, np.log10(3))
        self.assertEqual(self.sig((("169", ""),), (("171", ""),))[1], 1.0)

    def test_large_difference_is_not_small(self):
        rel, small, mind, _ = self.sig((("3153", ""),), (("3490", ""),))
        self.assertEqual((rel, small), (features.HN_CONFLICT, 0.0))
        self.assertAlmostEqual(mind, np.log10(338))

    def test_equal_number_is_not_a_small_difference(self):
        self.assertEqual(self.sig((("453", ""),), (("453", ""),))[1], 0.0)

    def test_suffix_differs_only_when_both_have_one_and_they_disagree(self):
        self.assertEqual(self.sig((("12", "a"),), (("12", "b"),))[3], 1.0)
        self.assertEqual(self.sig((("12", "a"),), (("12", ""),))[3], 0.0)
        self.assertEqual(self.sig((("12", "b"),), (("12", "b"),))[3], 0.0)

    def test_missing_number_gives_nan_distance(self):
        rel, small, mind, sfx = self.sig((("12", ""),), ())
        self.assertEqual((rel, small, sfx), (features.HN_ONE_MISSING, 0.0, 0.0))
        self.assertTrue(np.isnan(mind))


class TestBuildPairFeaturesBase(unittest.TestCase):
    def _frames(self):
        s1 = pd.DataFrame({
            "entity_id": ["S1-1"], "business_name": ["Quality Biomedical Holdings"],
            "business_address": ["7800 Valburn Drive, Austin, TX"], "country": ["US"],
            "name_full": ["quality biomedical holdings"], "name_core": ["quality biomedical"], "postal_code": [None],
        })
        cand = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2"], "business_name": ["QUALITY BIOMEDICAL", "Other Co"],
            "business_address": ["AUSTIN, TX, 7802 VALBURN DR", ""], "country": ["US", "US"],
            "name_full": ["quality biomedical", "other company"], "name_core": ["quality biomedical", "other"],
            "postal_code": [None, None],
        })
        pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-1", "S2-2"],
                              "blocks": ["b1_exact_core", "bgeo_address_token"], "n_blocks": [1, 1]})
        return pairs, s1, cand

    def test_address_features(self):
        pairs, s1, cand = self._frames()
        out = features.build_pair_features_base(pairs, s1, cand)
        r0, r1 = out.iloc[0], out.iloc[1]
        self.assertEqual(r0["hn_one_edit"], 1.0)
        self.assertEqual(r0["hn_equal"], 0.0)
        self.assertGreaterEqual(r0["street_token_set_ratio"], 95)  # "dr" expanded to "drive", numbers ignored
        self.assertEqual(r1["hn_one_missing"], 1.0)
        self.assertTrue(np.isnan(r1["street_ratio"]))  # blank candidate address -> no street signal

    def test_all_feature_columns_present_after_context(self):
        pairs, s1, cand = self._frames()
        out = features.build_pair_features(pairs, s1, cand)
        for c in features.FEATURE_COLUMNS:
            self.assertIn(c, out.columns)


if __name__ == "__main__":
    unittest.main()


class TestNameFeatures(unittest.TestCase):
    def test_containment(self):
        c = features._containment
        self.assertEqual(c(frozenset({"quality", "biomedical"}), frozenset({"quality", "biomedical", "holdings"})), 1.0)
        self.assertEqual(c(frozenset({"a", "b"}), frozenset({"a", "c", "d"})), 0.5)
        self.assertEqual(c(frozenset(), frozenset({"a"})), 0.0)

    def test_name_cleanup_features(self):
        s1 = pd.DataFrame({
            "entity_id": ["S1-1"], "business_name": ["Jarlus Pmv"], "business_address": ["131 Carroll Avenue, Mamaroneck, NY"],
            "country": ["US"], "name_full": ["jarlus pmv"], "name_core": ["jarlus pmv"], "postal_code": [None],
        })
        cand = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2"], "business_name": ["jarluspmv.com", "Jarlus Pmv (ID: 28974)"],
            "business_address": ["13 CARROLL AVE, MAMARONECK, NY"] * 2, "country": ["US", "US"],
            "name_full": ["jarluspmv com", "jarlus pmv id 28974"], "name_core": ["jarluspmv com", "jarlus pmv id 28974"],
            "postal_code": [None, None],
        })
        pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-1", "S2-2"],
                              "blocks": ["x", "x"], "n_blocks": [1, 1]})
        out = features.build_pair_features_base(pairs, s1, cand)
        self.assertEqual(out.loc[0, "name_nospace_ratio"], 100.0)   # "jarluspmv" vs "jarlus pmv" with spaces removed
        self.assertEqual(out.loc[1, "name_clean_full_ratio"], 100.0)  # "(ID: 28974)" stripped
        self.assertEqual(out.loc[1, "name_core_containment"], 1.0)


class TestRarityFeatures(unittest.TestCase):
    def _frames(self):
        def rec(i, name, addr):
            return {"entity_id": i, "business_name": name, "business_address": addr, "country": "US",
                    "name_full": name.lower(), "name_core": name.lower(), "postal_code": None}
        s1 = pd.DataFrame([rec("S1-1", "Zyxel Acme", "5 Main Street, Austin, TX"),
                           rec("S1-2", "Acme Zorbo", "9 Elm Street, Austin, TX")])
        # "acme" is in every name (common), "zyxel" only in two (rare).
        cand = pd.DataFrame([rec("S2-1", "Zyxel Acme", "5 Main Street, Austin, TX"),
                             rec("S2-2", "Acme Zorbo", "9 Elm Street, Austin, TX"),
                             rec("S2-3", "Qwerty Plumbing", "5 Main Street, Austin, TX")])
        return s1, cand

    def test_rare_shared_token_outweighs_common_one(self):
        s1, cand = self._frames()
        pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-2"], "cand_id": ["S2-1", "S2-1"], "blocks": ["x", "x"], "n_blocks": [1, 1]})
        out = features.build_pair_features_base(pairs, s1, cand)
        # S1-1 vs S2-1: identical names, all tokens shared -> 1.0
        self.assertAlmostEqual(out.loc[0, "name_idf_jaccard"], 1.0, places=5)
        # S1-2 vs S2-1: only the common token "acme" is shared -> low weighted overlap
        self.assertLess(out.loc[1, "name_idf_jaccard"], 0.35)
        self.assertGreater(out.loc[0, "name_max_shared_idf"], out.loc[1, "name_max_shared_idf"])

    def test_no_shared_token_is_zero(self):
        s1, cand = self._frames()
        pairs = pd.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-3"], "blocks": ["x"], "n_blocks": [1]})
        out = features.build_pair_features_base(pairs, s1, cand)
        self.assertEqual(out.loc[0, "name_idf_jaccard"], 0.0)
        self.assertEqual(out.loc[0, "name_idf_containment"], 0.0)

    def test_same_address_low_name_flag(self):
        s1, cand = self._frames()
        pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-3", "S2-1"], "blocks": ["x", "x"], "n_blocks": [1, 1]})
        out = features.build_pair_features_base(pairs, s1, cand)
        self.assertEqual(out.loc[0, "same_addr_low_name"], 1.0)  # same address, unrelated name
        self.assertEqual(out.loc[1, "same_addr_low_name"], 0.0)  # same address, same name
