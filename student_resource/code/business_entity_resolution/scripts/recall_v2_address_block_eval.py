"""Recall-v2, fix #4: measure the new B_address block (src.blocking.
block_b_address_rare_tokens -- rare shared address tokens + house-number
bonus, for DBA / renamed-business cases no name-based block can reach) on
the real validation slice.

Same "hold the other blocks fixed, only compute the new one" trick as
scripts/recall_v2_b2_df_sweep.py: loads the existing 5-block
tagged_{country}.parquet (from scripts/phase3_blocking_report.py's last real
run) as the baseline, adds B_address's own pairs on top, and recomputes
recall_uncapped/recall_capped/candidate stats for "baseline" vs
"baseline + B_address" so the marginal contribution is directly visible.
SQL/join-based (not a sparse matmul like B_tfidf), so -- like B1/B2/B3/
B_geo -- this runs at the real full validation-slice scale locally, not a
subset.

Usage (from code/business_entity_resolution/):
    python -m scripts.recall_v2_address_block_eval
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("AML_DUCKDB_MEMORY_LIMIT", "3GB")
os.environ.setdefault("AML_DUCKDB_THREADS", "2")

from src import blocking, config, io_utils  # noqa: E402
from scripts.phase3_blocking_report import load_eval_truth, VAL_SLICE_DIR, PHASE3_DIR  # noqa: E402
from scripts.recall_v2_b2_df_sweep import cap_like_production, recall_of  # noqa: E402

OUT_DIR = config.PARQUET_DIR / "recall_v2"
COUNTRIES = ["India", "US"]


def compute_address_block(con, country: str) -> pd.DataFrame:
    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet", columns=["entity_id", "country", "business_address"])
    s2 = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet", columns=["entity_id", "country", "business_address"])
    s3 = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet", columns=["entity_id", "country", "business_address"])
    s1_c = s1[s1["country"] == country]
    cand_c = pd.concat([s2[s2["country"] == country], s3[s3["country"] == country]], ignore_index=True)

    con.register("s1_raw", s1_c)
    con.register("cand_raw", cand_c)
    con.execute("CREATE OR REPLACE TABLE s1_addr AS SELECT * FROM s1_raw")
    con.execute("CREATE OR REPLACE TABLE cand_addr AS SELECT * FROM cand_raw")
    con.unregister("s1_raw")
    con.unregister("cand_raw")

    blocking.register_house_number(con, "s1_addr", "s1_house")
    blocking.register_house_number(con, "cand_addr", "cand_house")
    blocking.register_address_tokens(con, "s1_addr", "s1_atok")
    blocking.register_address_tokens(con, "cand_addr", "cand_atok")

    out = blocking.block_b_address_rare_tokens(con, "s1_house", "cand_house", "s1_atok", "cand_atok")
    df = con.execute(f"SELECT DISTINCT s1_id, cand_id FROM {out}").fetchdf()
    df["block"] = "b_address_rare_tokens"
    return df


def run() -> dict:
    t0 = time.time()
    eval_truth = load_eval_truth()

    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()
    con = io_utils.get_connection()

    baseline_frames, addr_frames = [], []
    for c in COUNTRIES:
        baseline_frames.append(pd.read_parquet(PHASE3_DIR / f"tagged_{c}.parquet")[["s1_id", "cand_id", "block"]])
        addr = compute_address_block(con, c)
        addr_frames.append(addr)
        print(f"  {c}: b_address raw pairs = {len(addr)} ({time.time()-t0:.1f}s)", flush=True)

    con.close()
    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()

    baseline = pd.concat(baseline_frames, ignore_index=True)
    addr_all = pd.concat(addr_frames, ignore_index=True)
    combined = pd.concat([baseline, addr_all], ignore_index=True)

    def stats(tagged: pd.DataFrame, label: str) -> dict:
        union_pairs = tagged[["s1_id", "cand_id"]].drop_duplicates()
        capped_pairs = cap_like_production(tagged)
        cand_counts = capped_pairs.groupby("s1_id").size()
        return {
            "label": label,
            "n_union_pairs": len(union_pairs),
            "n_capped_pairs": len(capped_pairs),
            "recall_uncapped": recall_of(union_pairs, eval_truth)["recall"],
            "recall_capped": recall_of(capped_pairs, eval_truth)["recall"],
            "avg_candidates_per_s1": float(cand_counts.mean()) if len(cand_counts) else 0.0,
            "p99_candidates_per_s1": float(cand_counts.quantile(0.99)) if len(cand_counts) else 0.0,
        }

    baseline_stats = stats(baseline, "baseline_5_blocks")
    combined_stats = stats(combined, "baseline_plus_b_address")

    # Pairs B_address alone recovers that the baseline 5 blocks miss entirely.
    baseline_union_keys = set(map(tuple, baseline[["s1_id", "cand_id"]].drop_duplicates().values))
    addr_union_keys = set(map(tuple, addr_all[["s1_id", "cand_id"]].drop_duplicates().values))
    net_new_pairs = addr_union_keys - baseline_union_keys
    net_new_true_pairs = 0
    for sid, cid in net_new_pairs:
        if cid in eval_truth.get(sid, set()):
            net_new_true_pairs += 1

    report = {
        "n_address_block_raw_pairs": len(addr_all),
        "n_net_new_pairs_vs_baseline": len(net_new_pairs),
        "n_net_new_TRUE_pairs_vs_baseline": net_new_true_pairs,
        "baseline": baseline_stats,
        "with_b_address": combined_stats,
        "build_time_seconds": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "address_block_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
