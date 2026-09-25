"""Recall-v2, fix #1 follow-up: tests whether raising B2_MAX_CAND_TOKEN_DF
only pays off in recall_CAPPED if CANDIDATES_PER_S1_CAP is raised alongside
it.

recall_v2_b2_df_sweep.py's first two points (150, 400) showed the expected
pattern from blocking.py's own docstring history repeating itself: raising
the DF cap alone improves recall_UNCAPPED (more true pairs are found by SOME
block) but leaves recall_CAPPED flat-to-worse -- the newly-found B2 pairs
compete with the other 4 blocks' pairs for the same fixed
CANDIDATES_PER_S1_CAP=50 slots per S1, tie-broken by n_blocks-then-cand_id,
so B2's single-block-tagged pairs often lose that tie to pairs multiple
blocks already agree on. This script holds B2_MAX_CAND_TOKEN_DF at one
relaxed value and sweeps CANDIDATES_PER_S1_CAP instead, to see if that's
really the mechanism and whether it's worth paying for (each extra candidate
is extra feature-build + training compute on the real Kaggle run).

Uses an in-memory DuckDB connection (no file lock) so it can run alongside
recall_v2_b2_df_sweep.py without contention.

Usage (from code/business_entity_resolution/):
    python -m scripts.recall_v2_cap_interaction
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blocking, config  # noqa: E402
from scripts.phase3_blocking_report import load_suffix_sets, load_translit_map, load_eval_truth, VAL_SLICE_DIR, PHASE3_DIR  # noqa: E402
from scripts.recall_v2_b2_df_sweep import load_other_blocks_fixed, cap_like_production, recall_of  # noqa: E402

OUT_DIR = config.PARQUET_DIR / "recall_v2"
COUNTRIES = ["India", "US"]
B2_DF_FOR_THIS_TEST = 2000  # a relaxed-but-not-extreme value, per miss_analysis.py's own CONFIGS
CAP_SWEEP = [50, 75, 100, 150, 250]  # 50 = current production CANDIDATES_PER_S1_CAP


def run() -> dict:
    t0 = time.time()
    suffix_sets = load_suffix_sets()
    translit_map = load_translit_map()
    eval_truth = load_eval_truth()

    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit = '3GB'")
    con.execute("SET threads = 2")
    # :memory: connections still spill temp data to disk under memory pressure,
    # but default to a small system temp location (C: on this machine, ~5GB
    # free) unless told otherwise -- point it at D: (490GB free) like every
    # other script's config.DUCKDB_TMP_DIR. Hit exactly this wall on the first
    # attempt (OutOfMemoryException: "7.3GiB/7.3GiB used" on C:'s temp dir).
    config.DUCKDB_TMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{config.DUCKDB_TMP_DIR.as_posix()}'")
    con.execute("SET max_temp_directory_size = '100GB'")

    other_fixed = {c: load_other_blocks_fixed(c) for c in COUNTRIES}
    b2_at_relaxed_df = {}
    orig_df_cap = blocking.B2_MAX_CAND_TOKEN_DF
    blocking.B2_MAX_CAND_TOKEN_DF = B2_DF_FOR_THIS_TEST
    try:
        for c in COUNTRIES:
            s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet")
            s2 = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet")
            s3 = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet")
            s1_c = blocking.add_normalized_columns(s1[s1["country"] == c], suffix_sets, translit_map)
            cand_c = blocking.add_normalized_columns(
                pd.concat([s2[s2["country"] == c], s3[s3["country"] == c]], ignore_index=True), suffix_sets, translit_map
            )
            con.register("s1_raw", s1_c)
            con.register("cand_raw", cand_c)
            con.execute("CREATE OR REPLACE TABLE s1_norm AS SELECT * FROM s1_raw")
            con.execute("CREATE OR REPLACE TABLE cand_norm AS SELECT * FROM cand_raw")
            con.unregister("s1_raw")
            con.unregister("cand_raw")
            blocking.register_name_tokens(con, "s1_norm", "s1_tok")
            blocking.register_name_tokens(con, "cand_norm", "cand_tok")
            blocking.register_suffix_table(con, suffix_sets)
            out = blocking.block_b2_rare_token(con, "s1_tok", "cand_tok", "suffix_table")
            b2_df = con.execute(f"SELECT DISTINCT s1_id, cand_id FROM {out}").fetchdf()
            b2_df["block"] = "b2_rare_token"
            b2_at_relaxed_df[c] = b2_df
            print(f"  {c}: b2 pairs at df={B2_DF_FOR_THIS_TEST}: {len(b2_df)} ({time.time()-t0:.1f}s)", flush=True)
    finally:
        blocking.B2_MAX_CAND_TOKEN_DF = orig_df_cap
    con.close()

    all_tagged = pd.concat(
        [pd.concat([other_fixed[c], b2_at_relaxed_df[c]], ignore_index=True) for c in COUNTRIES], ignore_index=True
    )

    results = []
    for cap in CAP_SWEEP:
        capped_pairs = cap_like_production(all_tagged, cap=cap)
        cand_counts = capped_pairs.groupby("s1_id").size()
        row = {
            "candidates_per_s1_cap": cap,
            "recall_capped": recall_of(capped_pairs, eval_truth)["recall"],
            "n_capped_pairs": len(capped_pairs),
            "avg_candidates_per_s1": float(cand_counts.mean()) if len(cand_counts) else 0.0,
            "p99_candidates_per_s1": float(cand_counts.quantile(0.99)) if len(cand_counts) else 0.0,
        }
        results.append(row)
        print(f"  cap={cap}: {row}", flush=True)

    report = {
        "b2_max_cand_token_df_used": B2_DF_FOR_THIS_TEST,
        "cap_sweep": results,
        "build_time_seconds": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cap_interaction.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
