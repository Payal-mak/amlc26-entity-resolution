"""Recall/miss analysis over the real ~45k validation slice already on disk
(scripts/phase3_blocking_report.py's output, Phase 3). Read-only diagnostic:
does not change src/blocking.py or anything the pipeline depends on.

Answers, for a stratified sample of missed true pairs (by country x source):
  1. Was it ever found by ANY block, even before the final cross-block cap?
     (reuses tagged_{country}.parquet -- this IS the "recall_uncapped" miss
     population, ~31,184 pairs, the dominant one; the much smaller ~1,419-pair
     population lost only at the final CANDIDATES_PER_S1_CAP=50 is reported
     separately for completeness but not the focus.)
  2. For the ones never found: would raising a block's own PER_BLOCK_CAP (20
     today), or B2's document-frequency exclusion (B2_MAX_CAND_TOKEN_DF=150
     today), or widening the sorted-neighborhood window (10 today), have
     caught it? Answered by actually RE-RUNNING src/blocking.py's real block
     functions with those constants monkeypatched at runtime (in this
     process only -- the file on disk is untouched), restricted to just the
     sampled S1 ids (the candidate side stays the FULL country pool, since
     document-frequency/rank need the real corpus to mean anything). This
     keeps it fast/light instead of redoing all of Phase 3 at full scale.
  3. For the ones no relaxation catches: a manual read of the actual name/
     address text, categorized (cross-script, typo, word-order/rename, DBA,
     address-only-match, unrelated-looking).

Usage (from code/business_entity_resolution/):
    python -m scripts.miss_analysis
"""

import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blocking, config, io_utils  # noqa: E402
from scripts.phase3_blocking_report import load_eval_truth, load_suffix_sets, load_translit_map  # noqa: E402

VAL_SLICE_DIR = config.PARQUET_DIR / "val_slice"
PHASE3_DIR = config.PARQUET_DIR / "phase3"
OUT_DIR = config.PARQUET_DIR / "miss_analysis"

N_SAMPLE_PER_CELL = 25  # x2 countries x2 sources = 100 total
SEED = config.RANDOM_SEED

# Relaxed-cap configs to test, in increasing order of relaxation. Each is
# (per_block_cap, b2_max_df, neighbor_window). "current" reproduces
# production exactly (sanity check: should recover 0 of the misses, since
# these are misses BY DEFINITION under current settings).
CONFIGS = [
    ("current", blocking.PER_BLOCK_CAP, blocking.B2_MAX_CAND_TOKEN_DF, blocking.SORTED_NEIGHBOR_WINDOW),
    ("cap200", 200, blocking.B2_MAX_CAND_TOKEN_DF, blocking.SORTED_NEIGHBOR_WINDOW),
    ("cap2000", 2000, blocking.B2_MAX_CAND_TOKEN_DF, blocking.SORTED_NEIGHBOR_WINDOW),
    ("cap2000_nodf", 2000, 10_000_000, blocking.SORTED_NEIGHBOR_WINDOW),
    ("cap2000_nodf_window50", 2000, 10_000_000, 50),
]


def compute_miss_universe(con) -> pd.DataFrame:
    """Every eval true pair not found by ANY block post-per-block-cap (the
    "recall_uncapped" miss population). Anti-join runs in DuckDB (streamed),
    only the ~31k miss rows themselves come back into pandas.
    """
    eval_truth = load_eval_truth()
    truth_df = pd.DataFrame(
        [(sid, cid) for sid, ids in eval_truth.items() for cid in ids], columns=["s1_id", "cand_id"]
    )
    con.register("truth_tmp", truth_df)

    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet", columns=["entity_id", "country", "is_eval"])
    s1 = s1.loc[s1["is_eval"], ["entity_id", "country"]]
    con.register("s1country_tmp", s1)

    frames = []
    for country in sorted(s1["country"].unique()):
        tagged_path = (PHASE3_DIR / f"tagged_{country}.parquet").as_posix()
        q = f"""
            SELECT t.s1_id, t.cand_id, '{country}' AS country
            FROM truth_tmp t
            JOIN s1country_tmp c ON c.entity_id = t.s1_id AND c.country = '{country}'
            LEFT JOIN read_parquet('{tagged_path}') f ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id
            WHERE f.s1_id IS NULL
        """
        frames.append(con.execute(q).fetchdf())
    con.unregister("truth_tmp")
    con.unregister("s1country_tmp")
    miss_df = pd.concat(frames, ignore_index=True)
    miss_df["source"] = miss_df["cand_id"].map(lambda c: "S2" if c.startswith("S2-") else "S3")
    return miss_df


def compute_final_cap_miss_count(con) -> dict:
    """The much smaller population: TRUE-MATCH pairs found post-per-block-cap
    (in tagged_{country}) but lost at the final cross-block
    CANDIDATES_PER_S1_CAP (not in capped_{country}) -- reported for
    completeness, not sampled.

    Bug fix (was counting every pair lost to the cap, not just true-match
    pairs among them -- gave an inflated ~218k instead of the correct ~1,419
    that scripts/phase3_blocking_report.py's report.json already implies:
    n_recovered(recall_uncapped) - n_recovered(recall_capped). Restricting the
    anti-join to eval_truth pairs (same pattern as compute_miss_universe)
    fixes it.
    """
    eval_truth = load_eval_truth()
    truth_df = pd.DataFrame(
        [(sid, cid) for sid, ids in eval_truth.items() for cid in ids], columns=["s1_id", "cand_id"]
    )
    con.register("truth_tmp", truth_df)

    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet", columns=["entity_id", "country", "is_eval"])
    s1 = s1.loc[s1["is_eval"], ["entity_id", "country"]]
    con.register("s1country_tmp", s1)

    counts = {}
    for country in ["India", "US"]:
        tagged = (PHASE3_DIR / f"tagged_{country}.parquet").as_posix()
        capped = (PHASE3_DIR / f"capped_{country}.parquet").as_posix()
        q = f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT t.s1_id, t.cand_id
                FROM truth_tmp t
                JOIN s1country_tmp c ON c.entity_id = t.s1_id AND c.country = '{country}'
                JOIN read_parquet('{tagged}') f ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id
            ) tagged_truth
            LEFT JOIN read_parquet('{capped}') c ON c.s1_id = tagged_truth.s1_id AND c.cand_id = tagged_truth.cand_id
            WHERE c.s1_id IS NULL
        """
        counts[country] = con.execute(q).fetchone()[0]
    con.unregister("truth_tmp")
    con.unregister("s1country_tmp")
    return counts


def stratified_sample(miss_df: pd.DataFrame, n_per_cell: int = N_SAMPLE_PER_CELL, seed: int = SEED) -> pd.DataFrame:
    parts = []
    for (country, source), g in miss_df.groupby(["country", "source"]):
        k = min(n_per_cell, len(g))
        parts.append(g.sample(n=k, random_state=seed))
        print(f"  stratum country={country} source={source}: {len(g)} misses available, sampled {k}")
    return pd.concat(parts, ignore_index=True)


def attach_text(con, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Pull business_name/address for the sampled ids only, via a DuckDB
    predicate pushdown (never materializes the full source files in pandas).
    """
    def _fetch(path: Path, ids: set) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame(columns=["entity_id", "business_name", "business_address"])
        id_list = ",".join(f"'{i}'" for i in ids)
        return con.execute(
            f"SELECT entity_id, business_name, business_address FROM read_parquet('{path.as_posix()}') "
            f"WHERE entity_id IN ({id_list})"
        ).fetchdf()

    s1_ids = set(sample_df["s1_id"])
    cand_ids = set(sample_df["cand_id"])
    s1_text = _fetch(VAL_SLICE_DIR / "source1.parquet", s1_ids).set_index("entity_id")
    cand_text = pd.concat(
        [_fetch(VAL_SLICE_DIR / "source2.parquet", cand_ids), _fetch(VAL_SLICE_DIR / "source3.parquet", cand_ids)],
        ignore_index=True,
    ).set_index("entity_id")

    out = sample_df.copy()
    out["s1_name"] = out["s1_id"].map(s1_text["business_name"])
    out["s1_address"] = out["s1_id"].map(s1_text["business_address"])
    out["cand_name"] = out["cand_id"].map(cand_text["business_name"])
    out["cand_address"] = out["cand_id"].map(cand_text["business_address"])
    return out


def load_country_side(con, view_path: Path, country: str, ids: set, suffix_sets: dict, translit_map: dict) -> pd.DataFrame:
    """One country's rows, optionally restricted to `ids` (None = all),
    filtered server-side in DuckDB before normalizing in Python.
    """
    where_ids = ""
    if ids is not None:
        id_list = ",".join(f"'{i}'" for i in ids)
        where_ids = f"AND entity_id IN ({id_list})"
    df = con.execute(
        f"SELECT * FROM read_parquet('{view_path.as_posix()}') WHERE country = '{country}' {where_ids}"
    ).fetchdf()
    return blocking.add_normalized_columns(df, suffix_sets, translit_map)


def run_blocks_relaxed(con, s1_c: pd.DataFrame, cand_c: pd.DataFrame, suffix_sets: dict, per_block_cap, b2_max_df, window) -> dict:
    """Re-run src/blocking.py's REAL block functions with its module-level
    cap constants monkeypatched for this call only, restored afterward no
    matter what. s1_c should already be restricted to the ids of interest;
    cand_c stays the full country pool.
    """
    orig = (blocking.PER_BLOCK_CAP, blocking.B2_MAX_CAND_TOKEN_DF, blocking.SORTED_NEIGHBOR_WINDOW)
    blocking.PER_BLOCK_CAP, blocking.B2_MAX_CAND_TOKEN_DF, blocking.SORTED_NEIGHBOR_WINDOW = per_block_cap, b2_max_df, window
    try:
        blocking.register_suffix_table(con, suffix_sets)
        block_views = blocking.run_all_blocks(con, s1_c, cand_c)
        return {name: con.execute(f"SELECT s1_id, cand_id FROM {view}").fetchdf() for name, view in block_views.items()}
    finally:
        blocking.PER_BLOCK_CAP, blocking.B2_MAX_CAND_TOKEN_DF, blocking.SORTED_NEIGHBOR_WINDOW = orig


def analyze_country(con, country: str, sample_country: pd.DataFrame, suffix_sets: dict, translit_map: dict) -> pd.DataFrame:
    """Run every relaxed-cap config for one country's sampled misses; return
    a long DataFrame (s1_id, cand_id, config, caught_by) marking which
    block(s) catch each pair under each config.
    """
    sample_ids = set(sample_country["s1_id"])
    cand_ids_of_interest = set(sample_country["cand_id"])  # ensure these exist in cand_c regardless of country filter edge cases
    s1_c = load_country_side(con, VAL_SLICE_DIR / "source1.parquet", country, sample_ids, suffix_sets, translit_map)

    s2_c = load_country_side(con, VAL_SLICE_DIR / "source2.parquet", country, None, suffix_sets, translit_map)
    s3_c = load_country_side(con, VAL_SLICE_DIR / "source3.parquet", country, None, suffix_sets, translit_map)
    cand_c = pd.concat([s2_c, s3_c], ignore_index=True)
    del s2_c, s3_c

    rows = []
    for cfg_name, per_block_cap, b2_max_df, window in CONFIGS:
        block_tables = run_blocks_relaxed(con, s1_c, cand_c, suffix_sets, per_block_cap, b2_max_df, window)
        for block_name, df in block_tables.items():
            hit = df.merge(sample_country[["s1_id", "cand_id"]], on=["s1_id", "cand_id"], how="inner")
            for _, r in hit.iterrows():
                rows.append({"s1_id": r["s1_id"], "cand_id": r["cand_id"], "config": cfg_name, "block": block_name})
        print(f"    country={country} config={cfg_name} done", flush=True)
    return pd.DataFrame(rows)


def summarize(sample_df: pd.DataFrame, hits_df: pd.DataFrame, n_total_misses: int) -> dict:
    sample_df = sample_df.copy()
    sample_df["key"] = list(zip(sample_df["s1_id"], sample_df["cand_id"]))
    n_sampled = len(sample_df)

    caught_by_config = {}
    for cfg_name, *_ in CONFIGS:
        keys = set(map(tuple, hits_df.loc[hits_df["config"] == cfg_name, ["s1_id", "cand_id"]].values)) if len(hits_df) else set()
        caught_by_config[cfg_name] = len(keys & set(sample_df["key"]))

    never_caught = sample_df[~sample_df["key"].isin(
        set(map(tuple, hits_df[["s1_id", "cand_id"]].values)) if len(hits_df) else set()
    )]

    caught_by_block_cap2000 = {}
    if len(hits_df):
        sub = hits_df[hits_df["config"] == "cap2000"]
        for block in sub["block"].unique():
            caught_by_block_cap2000[block] = sub.loc[sub["block"] == block, ["s1_id", "cand_id"]].drop_duplicates().shape[0]

    df_cap_specific = 0
    if len(hits_df):
        cap2000_keys = set(map(tuple, hits_df.loc[hits_df["config"] == "cap2000", ["s1_id", "cand_id"]].values))
        nodf_keys = set(map(tuple, hits_df.loc[hits_df["config"] == "cap2000_nodf", ["s1_id", "cand_id"]].values))
        df_cap_specific = len(nodf_keys - cap2000_keys)

    window_specific = 0
    if len(hits_df):
        nodf_keys = set(map(tuple, hits_df.loc[hits_df["config"] == "cap2000_nodf", ["s1_id", "cand_id"]].values))
        window_keys = set(map(tuple, hits_df.loc[hits_df["config"] == "cap2000_nodf_window50", ["s1_id", "cand_id"]].values))
        window_specific = len(window_keys - nodf_keys)

    fraction_sampled = n_sampled / n_total_misses if n_total_misses else 0
    return {
        "n_total_miss_universe": n_total_misses,
        "n_sampled": n_sampled,
        "fraction_sampled": fraction_sampled,
        "caught_by_config": caught_by_config,
        "caught_by_block_at_cap2000": caught_by_block_cap2000,
        "additional_catch_from_lifting_b2_df_cap": df_cap_specific,
        "additional_catch_from_widening_neighbor_window_to_50": window_specific,
        "never_caught_even_fully_relaxed": len(never_caught),
        "never_caught_examples": never_caught.drop(columns=["key"]).to_dict("records"),
    }


def run() -> dict:
    t0 = time.time()
    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()
    con = io_utils.get_connection()

    print("computing full miss universe from tagged_{country}.parquet...", flush=True)
    miss_df = compute_miss_universe(con)
    print(f"  miss universe: {len(miss_df)} pairs ({time.time()-t0:.1f}s)", flush=True)
    final_cap_misses = compute_final_cap_miss_count(con)
    print(f"  (separately: {final_cap_misses} pairs found post-per-block-cap but lost at the final cross-block cap)", flush=True)

    print("stratified sampling...", flush=True)
    sample_df = stratified_sample(miss_df)
    sample_df = attach_text(con, sample_df)

    suffix_sets = load_suffix_sets()
    translit_map = load_translit_map()

    all_hits = []
    for country in sorted(sample_df["country"].unique()):
        print(f"  analyzing country={country}...", flush=True)
        sample_country = sample_df[sample_df["country"] == country]
        hits = analyze_country(con, country, sample_country, suffix_sets, translit_map)
        all_hits.append(hits)
    hits_df = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame(columns=["s1_id", "cand_id", "config", "block"])

    summary = summarize(sample_df, hits_df, len(miss_df))
    summary["final_cap_miss_population"] = final_cap_misses
    summary["build_time_seconds"] = round(time.time() - t0, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(OUT_DIR / "sample.parquet", index=False)
    hits_df.to_parquet(OUT_DIR / "hits.parquet", index=False)
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    con.close()
    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()
    return summary


if __name__ == "__main__":
    summary = run()
    print(json.dumps({k: v for k, v in summary.items() if k != "never_caught_examples"}, indent=2, default=str, ensure_ascii=False))
