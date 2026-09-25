"""Recall-v2, fix #2: SMALL-SUBSET local test of B_tfidf (src.blocking.
block_b_tfidf_char_ngram -- char-3-gram TF-IDF cosine top-K, within
country), per explicit instruction: full-country-scale char-trigram cosine
(185k x 658k for India) is a fundamentally different memory/CPU profile than
the SQL equality-join blocks (a promiscuous shared trigram densifies the
similarity matrix in a way no per-block DF cap on this 8GB machine can fully
tame) -- this machine tests CORRECTNESS/QUALITY on a subset; the real
full-scale recall/candidate-cost number is a Kaggle (31GB RAM) measurement,
not claimed here.

Subset construction: N_S1_SAMPLE S1 entities per country (random), plus
every one of their true-match candidates (so recall on the sample is
measurable) plus a random background of N_CAND_BACKGROUND other candidates
from the same country (so the vocabulary/IDF distribution and the
top-K/min-similarity competition are realistic, not just "S1 vs its own
answers"). Uses an in-memory DuckDB connection (no file lock, so this can
run alongside other recall-v2 scripts without contention).

Usage (from code/business_entity_resolution/):
    python -m scripts.recall_v2_tfidf_subset_test
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blocking, config  # noqa: E402
from scripts.phase3_blocking_report import load_suffix_sets, load_translit_map, load_eval_truth, VAL_SLICE_DIR  # noqa: E402

OUT_DIR = config.PARQUET_DIR / "recall_v2"
COUNTRIES = ["India", "US"]
N_S1_SAMPLE = 1500
N_CAND_BACKGROUND = 15000
SEED = config.RANDOM_SEED


def build_subset(country: str, eval_truth: dict, suffix_sets: dict, translit_map: dict) -> tuple:
    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet")
    s1 = s1[(s1["country"] == country) & (s1["is_eval"])]
    sample_s1 = s1.sample(n=min(N_S1_SAMPLE, len(s1)), random_state=SEED)

    s2 = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet")
    s3 = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet")
    cand_all = pd.concat([s2[s2["country"] == country], s3[s3["country"] == country]], ignore_index=True)

    true_ids = set()
    for sid in sample_s1["entity_id"]:
        true_ids |= eval_truth.get(sid, set())
    true_cands = cand_all[cand_all["entity_id"].isin(true_ids)]
    background = cand_all[~cand_all["entity_id"].isin(true_ids)].sample(
        n=min(N_CAND_BACKGROUND, len(cand_all)), random_state=SEED
    )
    sample_cand = pd.concat([true_cands, background], ignore_index=True).drop_duplicates("entity_id")

    sample_s1 = blocking.add_normalized_columns(sample_s1, suffix_sets, translit_map)
    sample_cand = blocking.add_normalized_columns(sample_cand, suffix_sets, translit_map)
    return sample_s1, sample_cand, len(true_ids)


def run() -> dict:
    t0 = time.time()
    suffix_sets = load_suffix_sets()
    translit_map = load_translit_map()
    eval_truth = load_eval_truth()

    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit = '2GB'")

    results = {}
    for country in COUNTRIES:
        tc = time.time()
        s1_sub, cand_sub, n_true_cands = build_subset(country, eval_truth, suffix_sets, translit_map)
        con.register("s1v", s1_sub)
        con.register("candv", cand_sub)
        con.execute("CREATE OR REPLACE TABLE s1t AS SELECT * FROM s1v")
        con.execute("CREATE OR REPLACE TABLE candt AS SELECT * FROM candv")
        con.unregister("s1v")
        con.unregister("candv")

        out = blocking.block_b_tfidf_char_ngram(con, "s1t", "candt")
        pairs = con.execute(f"SELECT s1_id, cand_id FROM {out}").fetchdf()

        found = pairs.groupby("s1_id")["cand_id"].apply(set).to_dict()
        s1_ids = set(s1_sub["entity_id"])
        total = recovered = 0
        for sid in s1_ids:
            truth = eval_truth.get(sid, set())
            for mid in truth:
                total += 1
                recovered += mid in found.get(sid, set())

        cand_counts = pairs.groupby("s1_id").size()
        results[country] = {
            "n_s1_sampled": len(s1_sub),
            "n_cand_pool": len(cand_sub),
            "n_true_pairs_in_subset": total,
            "n_recovered_by_b_tfidf_alone": recovered,
            "recall_on_subset": recovered / total if total else None,
            "n_tfidf_pairs": len(pairs),
            "avg_candidates_per_s1": float(cand_counts.mean()) if len(cand_counts) else 0.0,
            "seconds": round(time.time() - tc, 1),
        }
        print(f"  {country}: {results[country]}", flush=True)

    con.close()
    report = {"caveat": "SMALL SUBSET ONLY, not the real validation-slice scale -- see module docstring", "by_country": results, "build_time_seconds": round(time.time() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "tfidf_subset_test.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
