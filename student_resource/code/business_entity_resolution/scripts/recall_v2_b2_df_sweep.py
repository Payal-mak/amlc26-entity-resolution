"""Recall-v2, fix #1: sweep B2_MAX_CAND_TOKEN_DF to find where recall gain
flattens relative to candidate growth (miss_analysis.py's dominant single
lever: additional_catch_from_lifting_b2_df_cap=33 of 100 sampled misses).

Reuses the REAL src/blocking.py block_b2_rare_token function (only its
module-level B2_MAX_CAND_TOKEN_DF constant is monkeypatched per sweep value,
restored after every call) against the on-disk tagged_{country}.parquet
files from scripts/phase3_blocking_report.py's last real run -- the other
four blocks (b1/b3/bneighbor/bgeo) are unaffected by this constant, so they
are held fixed (loaded from disk) rather than re-run, which is what makes
sweeping several DF values locally cheap: only B2 itself (a single indexed
join over pre-built token views) reruns per value, not the full five-block
pipeline DuckDB already spent ~530s on once.

For each DF value: recompute b2_pairs, recombine with the fixed other-block
pairs, recompute the union (recall_uncapped) and the real cap_candidates
logic (recall_capped, top CANDIDATES_PER_S1_CAP per S1 by n_blocks desc)
in pandas, and report recall + candidate-count stats.

Usage (from code/business_entity_resolution/):
    python -m scripts.recall_v2_b2_df_sweep
"""

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Local-dev machine (8GB RAM) -- must override BEFORE `from src import
# config` anywhere in this process, or config.py's Kaggle-oriented default
# (20GB) applies here too and DuckDB doesn't spill to disk soon enough.
# First sweep attempt hit an OutOfMemoryException at df_cap=800 because of
# exactly this (see PROJECT_LOG.md).
os.environ.setdefault("AML_DUCKDB_MEMORY_LIMIT", "3GB")
os.environ.setdefault("AML_DUCKDB_THREADS", "2")

from src import blocking, config, io_utils  # noqa: E402
from scripts.phase3_blocking_report import (  # noqa: E402
    load_suffix_sets, load_translit_map, load_eval_truth, VAL_SLICE_DIR, PHASE3_DIR,
)

OUT_DIR = config.PARQUET_DIR / "recall_v2"
COUNTRIES = ["India", "US"]

# Current production value first (sanity check: should reproduce report.json
# exactly), then increasing relaxation up to effectively "no cap".
DF_SWEEP = [150, 400, 800, 1500, 3000, 6000, 12000, 30000, 10_000_000]


def load_other_blocks_fixed(country: str) -> pd.DataFrame:
    """tagged_{country}.parquet rows from every block EXCEPT b2_rare_token --
    unaffected by the DF sweep, so loaded once and reused across all values.
    """
    tagged = pd.read_parquet(PHASE3_DIR / f"tagged_{country}.parquet")
    return tagged.loc[tagged["block"] != "b2_rare_token", ["s1_id", "cand_id", "block"]]


def register_token_views(con, country: str, suffix_sets: dict, translit_map: dict) -> None:
    """Load+normalize one country's S1/candidate rows and register the
    s1_tok/cand_tok/suffix_table views block_b2_rare_token needs -- same
    setup run_all_blocks does, isolated here so it runs once per country
    instead of once per sweep value.
    """
    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet")
    s2 = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet")
    s3 = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet")
    s1_c = blocking.add_normalized_columns(s1[s1["country"] == country], suffix_sets, translit_map)
    cand = pd.concat([s2[s2["country"] == country], s3[s3["country"] == country]], ignore_index=True)
    cand_c = blocking.add_normalized_columns(cand, suffix_sets, translit_map)

    con.register("s1_raw", s1_c)
    con.register("cand_raw", cand_c)
    con.execute("CREATE OR REPLACE TABLE s1_norm AS SELECT * FROM s1_raw")
    con.execute("CREATE OR REPLACE TABLE cand_norm AS SELECT * FROM cand_raw")
    con.unregister("s1_raw")
    con.unregister("cand_raw")
    blocking.register_name_tokens(con, "s1_norm", "s1_tok")
    blocking.register_name_tokens(con, "cand_norm", "cand_tok")
    blocking.register_suffix_table(con, suffix_sets)
    return len(s1_c), len(cand_c)


def cap_like_production(tagged_pairs: pd.DataFrame, cap: int = blocking.CANDIDATES_PER_S1_CAP) -> pd.DataFrame:
    """Pandas reimplementation of blocking.union_and_score + cap_candidates
    (n_blocks = count of distinct blocks that found the pair, top `cap` per
    S1 by n_blocks desc then cand_id asc) -- avoids round-tripping through
    DuckDB again for something this cheap once the tagging is already in a
    dataframe.
    """
    scored = (
        tagged_pairs.drop_duplicates(["s1_id", "cand_id", "block"])
        .groupby(["s1_id", "cand_id"])["block"].nunique().rename("n_blocks").reset_index()
    )
    scored = scored.sort_values(["s1_id", "n_blocks", "cand_id"], ascending=[True, False, True])
    scored["rank"] = scored.groupby("s1_id").cumcount() + 1
    return scored.loc[scored["rank"] <= cap]


def recall_of(pairs_df: pd.DataFrame, eval_truth: dict) -> dict:
    found = pairs_df.groupby("s1_id")["cand_id"].apply(set).to_dict()
    total = recovered = 0
    for s1id, truth in eval_truth.items():
        cands = found.get(s1id, set())
        for mid in truth:
            total += 1
            recovered += mid in cands
    return {"recall": recovered / total if total else None, "n_total": total, "n_recovered": recovered}


def run() -> dict:
    t0 = time.time()
    suffix_sets = load_suffix_sets()
    translit_map = load_translit_map()
    eval_truth = load_eval_truth()

    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()
    con = io_utils.get_connection()

    other_fixed = {c: load_other_blocks_fixed(c) for c in COUNTRIES}
    sizes = {}
    for c in COUNTRIES:
        n_s1, n_cand = register_token_views(con, c, suffix_sets, translit_map)
        sizes[c] = {"n_s1": n_s1, "n_cand": n_cand}
        print(f"registered token views for {c}: s1={n_s1} cand={n_cand} ({time.time()-t0:.1f}s)", flush=True)
        # rename per-country so the next country's register doesn't clobber it
        con.execute(f"ALTER TABLE s1_tok RENAME TO s1_tok_{c}")
        con.execute(f"ALTER TABLE cand_tok RENAME TO cand_tok_{c}")

    orig_df_cap = blocking.B2_MAX_CAND_TOKEN_DF
    results = []
    try:
        for df_cap in DF_SWEEP:
            blocking.B2_MAX_CAND_TOKEN_DF = df_cap
            tc = time.time()
            try:
                per_country_b2 = {}
                for c in COUNTRIES:
                    out = blocking.block_b2_rare_token(con, f"s1_tok_{c}", f"cand_tok_{c}", "suffix_table")
                    b2_df = con.execute(f"SELECT DISTINCT s1_id, cand_id FROM {out}").fetchdf()
                    b2_df["block"] = "b2_rare_token"
                    per_country_b2[c] = b2_df

                all_tagged = pd.concat(
                    [pd.concat([other_fixed[c], per_country_b2[c]], ignore_index=True) for c in COUNTRIES],
                    ignore_index=True,
                )
                union_pairs = all_tagged[["s1_id", "cand_id"]].drop_duplicates()
                capped_pairs = cap_like_production(all_tagged)

                cand_counts = capped_pairs.groupby("s1_id").size()
                row = {
                    "b2_max_cand_token_df": df_cap,
                    "n_b2_pairs": sum(len(v) for v in per_country_b2.values()),
                    "recall_uncapped": recall_of(union_pairs, eval_truth)["recall"],
                    "recall_capped": recall_of(capped_pairs, eval_truth)["recall"],
                    "avg_candidates_per_s1": float(cand_counts.mean()) if len(cand_counts) else 0.0,
                    "p99_candidates_per_s1": float(cand_counts.quantile(0.99)) if len(cand_counts) else 0.0,
                    "max_candidates_per_s1": int(cand_counts.max()) if len(cand_counts) else 0,
                    "seconds": round(time.time() - tc, 1),
                }
                results.append(row)
                print(f"  df_cap={df_cap}: {row}", flush=True)
            except Exception as exc:  # noqa: BLE001
                # An OOM mid-query can leave the connection's internal state unreliable for
                # further queries -- record the failure and stop sweeping rather than risk a
                # silently-wrong result from a half-broken connection on the next (higher, so
                # likely-also-failing) value.
                row = {"b2_max_cand_token_df": df_cap, "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - tc, 1)}
                results.append(row)
                print(f"  df_cap={df_cap}: {row} -- stopping sweep here", flush=True)
                break
    finally:
        blocking.B2_MAX_CAND_TOKEN_DF = orig_df_cap

    con.close()
    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()

    report = {"sizes": sizes, "sweep": results, "build_time_seconds": round(time.time() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "b2_df_sweep.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
