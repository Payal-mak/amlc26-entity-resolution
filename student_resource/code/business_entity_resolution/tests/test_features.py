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
