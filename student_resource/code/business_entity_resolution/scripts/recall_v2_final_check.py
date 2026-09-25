"""Recall-v2: final combined LOCAL measurement of all 6 always-on blocks
(the original 5 plus B_address) at the new defaults
(B2_MAX_CAND_TOKEN_DF=2000, CANDIDATES_PER_S1_CAP=75), with the
has_indic_script/transliterate_indic fix live (add_normalized_columns calls
normalize_full unconditionally, so every block benefits automatically).
B_tfidf excluded (AML_ENABLE_TFIDF_BLOCK=false expected in the environment)
-- per fix #2's explicit "never full country scale locally" instruction.

Lighter-weight than just rerunning scripts/phase3_blocking_report.py as-is:
that script accumulates BOTH countries' full tagged/capped dataframes in
memory across its loop AND writes them to parquet, which is what actually
OOM'd on this 8GB machine at the new (bigger) candidate counts -- see
PROJECT_LOG.md. This script computes recall directly from each country's
in-memory tagged/capped result (no parquet write, no cross-country
accumulation of the raw pair data -- only small running recall counters),
processing one country fully before starting the next.

Usage (from code/business_entity_resolution/):
    AML_ENABLE_TFIDF_BLOCK=false python -m scripts.recall_v2_final_check
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("AML_DUCKDB_MEMORY_LIMIT", "2GB")
os.environ.setdefault("AML_DUCKDB_THREADS", "1")

from src import blocking, config, io_utils  # noqa: E402
from scripts.phase3_blocking_report import load_suffix_sets, load_translit_map, load_eval_truth, VAL_SLICE_DIR  # noqa: E402

OUT_DIR = config.PARQUET_DIR / "recall_v2"
COUNTRIES = ["India", "US"]


def recall_of(pairs_df: pd.DataFrame, eval_truth: dict, s1_ids: set) -> dict:
    found = pairs_df.groupby("s1_id")["cand_id"].apply(set).to_dict()
    total = recovered = 0
    for s1id in s1_ids:
        truth = eval_truth.get(s1id, set())
        cands = found.get(s1id, set())
        for mid in truth:
            total += 1
            recovered += mid in cands
    return total, recovered


def run() -> dict:
    t0 = time.time()
    suffix_sets = load_suffix_sets()
    translit_map = load_translit_map()
    eval_truth = load_eval_truth()

    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()

    per_country = {}
    total_true = total_recovered_uncapped = total_recovered_capped = 0
    total_capped_pairs = 0
    all_cand_counts = []

    for country in COUNTRIES:
        tc = time.time()
        con = io_utils.get_connection()
        blocking.register_suffix_table(con, suffix_sets)

        s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet")
        s2 = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet")
        s3 = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet")
        s1_c = blocking.add_normalized_columns(s1[s1["country"] == country], suffix_sets, translit_map)
        cand_c = blocking.add_normalized_columns(
            pd.concat([s2[s2["country"] == country], s3[s3["country"] == country]], ignore_index=True),
            suffix_sets, translit_map,
        )
        del s1, s2, s3
        gc.collect()
        s1_ids = set(s1_c["entity_id"])

        block_views = blocking.run_all_blocks(con, s1_c, cand_c)
        del s1_c, cand_c
        gc.collect()

        union_view = blocking.union_and_score(con, block_views)
        capped_view = blocking.cap_candidates(con, union_view)

        union_df = con.execute(f"SELECT DISTINCT s1_id, cand_id FROM all_pairs_tagged").fetchdf()
        capped_df = con.execute(f"SELECT s1_id, cand_id FROM {capped_view}").fetchdf()
        con.close()
        if config.DUCKDB_PATH.exists():
            config.DUCKDB_PATH.unlink()

        n_true_c, rec_unc_c = recall_of(union_df, eval_truth, s1_ids)
        _, rec_cap_c = recall_of(capped_df, eval_truth, s1_ids)
        cand_counts = capped_df.groupby("s1_id").size()

        per_country[country] = {
            "n_true_pairs": n_true_c,
            "recall_uncapped": rec_unc_c / n_true_c if n_true_c else None,
            "recall_capped": rec_cap_c / n_true_c if n_true_c else None,
            "n_union_pairs": len(union_df),
            "n_capped_pairs": len(capped_df),
            "avg_candidates_per_s1": float(cand_counts.mean()) if len(cand_counts) else 0.0,
            "seconds": round(time.time() - tc, 1),
        }
        total_true += n_true_c
        total_recovered_uncapped += rec_unc_c
        total_recovered_capped += rec_cap_c
        total_capped_pairs += len(capped_df)
        all_cand_counts.append(cand_counts)
        print(f"  {country}: {per_country[country]}", flush=True)

        del union_df, capped_df, cand_counts
        gc.collect()

    combined_counts = pd.concat(all_cand_counts)
    report = {
        "enable_tfidf_block": config.ENABLE_TFIDF_BLOCK,
        "b2_max_cand_token_df": blocking.B2_MAX_CAND_TOKEN_DF,
        "candidates_per_s1_cap": blocking.CANDIDATES_PER_S1_CAP,
        "block_names": blocking.BLOCK_NAMES,
        "per_country": per_country,
        "overall": {
            "n_total_true_pairs": total_true,
            "recall_uncapped": total_recovered_uncapped / total_true if total_true else None,
            "recall_capped": total_recovered_capped / total_true if total_true else None,
            "n_capped_pairs": total_capped_pairs,
            "avg_candidates_per_s1": float(combined_counts.mean()),
            "p99_candidates_per_s1": float(combined_counts.quantile(0.99)),
        },
        "build_time_seconds": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "final_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
