"""Unit tests for the pure parts of scripts/rules_baseline.py.

Run with: python -m unittest tests.test_rules_baseline -v
(from code/business_entity_resolution/)
"""

import unittest

import numpy as np
import pandas as pd

from scripts import rules_baseline as rb
from src import features


def _pairs(**over):
    base = dict(s1_id=["S1-1"], cand_id=["S2-1"], core_eq=[False], postal_eq=[False], postal_conflict=[False],
                street_eq=[False], street_sim=[0.0], name_sim=[0.0], name_sort=[0.0], idf=[0.0],
                hn_rel=[features.HN_CONFLICT])
    base.update({k: [v] for k, v in over.items()})
    return pd.DataFrame(base)


class TestRules(unittest.TestCase):
    cfg = dict(rules=["R1", "R2", "R3"], x=90, hn_mode="near", idf_min=0.5, street_min=100, r3_street=0, sim="name_sim")

    def test_r1_needs_core_equal_and_postal_or_street_with_compatible_number(self):
        self.assertEqual(len(rb.apply_rules(_pairs(core_eq=True, postal_eq=True), {**self.cfg, "rules": ["R1"]})), 1)
        self.assertEqual(len(rb.apply_rules(_pairs(core_eq=True, street_sim=100.0, hn_rel=features.HN_EQUAL), {**self.cfg, "rules": ["R1"]})), 1)
        self.assertEqual(len(rb.apply_rules(_pairs(core_eq=True, street_sim=100.0, hn_rel=features.HN_CONFLICT), {**self.cfg, "rules": ["R1"]})), 0)
        self.assertEqual(len(rb.apply_rules(_pairs(core_eq=False, postal_eq=True), {**self.cfg, "rules": ["R1"]})), 0)

    def test_r2_needs_similar_names_and_street_and_number(self):
        f = dict(name_sim=95.0, street_sim=100.0, hn_rel=features.HN_ONE_EDIT)
        self.assertEqual(len(rb.apply_rules(_pairs(**f), {**self.cfg, "rules": ["R2"]})), 1)
        self.assertEqual(len(rb.apply_rules(_pairs(**f), {**self.cfg, "rules": ["R2"], "hn_mode": "strict"})), 0)   # one-edit not allowed in strict mode
        self.assertEqual(len(rb.apply_rules(_pairs(**{**f, "name_sim": 80.0}), {**self.cfg, "rules": ["R2"]})), 0)

    def test_r3_needs_rare_name_and_no_postal_conflict(self):
        cfg = {**self.cfg, "rules": ["R3"]}
        self.assertEqual(len(rb.apply_rules(_pairs(core_eq=True, idf=0.8), cfg)), 1)
        self.assertEqual(len(rb.apply_rules(_pairs(core_eq=True, idf=0.2), cfg)), 0)                      # common name
        self.assertEqual(len(rb.apply_rules(_pairs(core_eq=True, idf=0.8, postal_conflict=True), cfg)), 0)


class TestOneToOne(unittest.TestCase):
    def test_each_candidate_keeps_only_its_best_s1(self):
        m = pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-2"], "cand_id": ["S2-1", "S2-1", "S3-9"], "score": [150.0, 250.0, 120.0]})
        out = rb.one_to_one(m)
        self.assertEqual(sorted(zip(out["s1_id"], out["cand_id"])), [("S1-2", "S2-1"), ("S1-2", "S3-9")])

    def test_ties_go_to_the_smaller_s1_id(self):
        m = pd.DataFrame({"s1_id": ["S1-9", "S1-2"], "cand_id": ["S2-1", "S2-1"], "score": [100.0, 100.0]})
        self.assertEqual(rb.one_to_one(m)["s1_id"].tolist(), ["S1-2"])


class TestEntityKeys(unittest.TestCase):
    def test_keys_reuse_normalize_and_blocking(self):
        df = pd.DataFrame({
            "entity_id": ["S1-1", "S2-1"], "country": ["France", "France"],
            "business_name": ["Boulangerie Dupont SARL", "DUPONT Boulangerie (ID: 4455)"],
            "business_address": ["12 Rue Victor Hugo, Lille, 59000", "12 R Victor Hugo, Lille"],
        })
        k = rb.entity_keys(df, {"France": {"sarl"}}, {})
        self.assertEqual(k["src"].tolist(), ["1", "2"])
        self.assertEqual(k.loc[0, "name_sorted"], k.loc[1, "name_sorted"])       # order-insensitive key, decoration stripped
        self.assertEqual(k.loc[0, "street"], k.loc[1, "street"])                # 'r' expanded to 'rue' after a house number
        self.assertEqual(k.loc[0, "hn1"], "12")


if __name__ == "__main__":
    unittest.main()
