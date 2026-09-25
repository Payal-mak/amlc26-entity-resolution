"""Phase 3 measurements: build all blocks over the Phase 1 validation slice,
cap candidates per S1, and report recall ceiling (overall / per-country /
per-source / per-block-marginal), candidate-count stats, and the worst misses.

Processed per country, writing each country's capped candidate pairs to
parquet on D: before moving to the next -- so peak working memory is bounded
by one country's data (at most ~700k candidate rows for India, the larger of
the two), not the full ~1.07M-row pool at once. Combined with every block
capping itself before the union (src/blocking.py), this is the fix for three
separate OOM crashes the unbounded/whole-slice-at-once version hit (see
PROJECT_LOG.md, Phase 3).

Usage (from code/business_entity_resolution/):
    python -m scripts.phase3_blocking_report
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blocking, config, io_utils  # noqa: E402

VAL_SLICE_DIR = config.PARQUET_DIR / "val_slice"
PHASE2_DIR = config.PARQUET_DIR / "phase2"
PHASE3_DIR = config.PARQUET_DIR / "phase3"
BLOCK_NAMES = blocking.BLOCK_NAMES


def load_suffix_sets() -> dict:
    df = pd.read_parquet(PHASE2_DIR / "suffix_token_candidates.parquet")
    return {c: set(g["token"]) for c, g in df.groupby("country")}


def load_translit_map() -> dict:
    path = PHASE2_DIR / "translit_token_map.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["translit_token"], df["latin_token"]))


def load_and_normalize() -> tuple:
    """Load the validation slice and add normalized columns (Python side).

    Output: (s1_df, cand_df) -- cand_df is source2+source3 unioned, tagged
    with a `source` column (S2/S3, derived from the id prefix).
    """
    suffix_sets = load_suffix_sets()
    translit_map = load_translit_map()
    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet")
    s2 = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet")
    s3 = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet")

    s1 = blocking.add_normalized_columns(s1, suffix_sets, translit_map)
    s2 = blocking.add_normalized_columns(s2, suffix_sets, translit_map)
    s3 = blocking.add_normalized_columns(s3, suffix_sets, translit_map)
    s2["source"] = "S2"
    s3["source"] = "S3"
    cand = pd.concat([s2, s3], ignore_index=True)
    return s1, cand


def load_eval_truth() -> dict:
    """{s1_id: set(matched_ids)} restricted to is_eval==True S1 entities."""
    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet", columns=["entity_id", "is_eval"])
    eval_ids = set(s1.loc[s1["is_eval"], "entity_id"])
    gt = pd.read_parquet(VAL_SLICE_DIR / "ground_truth_in_slice.parquet")
    result = {}
    for s1id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if s1id in eval_ids:
            result[s1id] = set(ids.split(",")) if isinstance(ids, str) and ids.strip() else set()
    return result


def run() -> dict:
    t0 = time.time()
    s1, cand = load_and_normalize()
    print(f"loaded+normalized: s1={len(s1)} cand={len(cand)} ({time.time()-t0:.1f}s)")

    suffix_sets = load_suffix_sets()
    eval_truth = load_eval_truth()
    s1_country_map = dict(zip(s1["entity_id"], s1["country"]))

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    countries = sorted(s1["country"].unique())

    all_capped_frames = []
    all_union_frames = []
    raw_pairs_per_block_total = {name: 0 for name in BLOCK_NAMES}

    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()

    for country in countries:
        tc = time.time()
        con = io_utils.get_connection()
        blocking.register_suffix_table(con, suffix_sets)

        s1_c = s1[s1["country"] == country]
        cand_c = cand[cand["country"] == country]
        block_views = blocking.run_all_blocks(con, s1_c, cand_c)

        for name, view in block_views.items():
            n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            raw_pairs_per_block_total[name] += n

        union_view = blocking.union_and_score(con, block_views)
        capped_view = blocking.cap_candidates(con, union_view)

        tagged_df = con.execute("SELECT s1_id, cand_id, block FROM all_pairs_tagged").fetchdf()
        capped_df = con.execute(f"SELECT * FROM {capped_view}").fetchdf()
        tagged_df.to_parquet(PHASE3_DIR / f"tagged_{country}.parquet", index=False)
        capped_df.to_parquet(PHASE3_DIR / f"capped_{country}.parquet", index=False)
        all_union_frames.append(tagged_df[["s1_id", "cand_id"]].drop_duplicates())
        all_capped_frames.append(capped_df[["s1_id", "cand_id"]])

        con.close()
        if config.DUCKDB_PATH.exists():
            config.DUCKDB_PATH.unlink()
        print(f"  country={country}: s1={len(s1_c)} cand={len(cand_c)} union={len(tagged_df)} capped={len(capped_df)} ({time.time()-tc:.1f}s)")

    union_all = pd.concat(all_union_frames, ignore_index=True)
    capped_all = pd.concat(all_capped_frames, ignore_index=True)
    print(f"all countries done ({time.time()-t0:.1f}s)")

    def recall_of(pairs_df: pd.DataFrame) -> dict:
        found_map = pairs_df.groupby("s1_id")["cand_id"].apply(set).to_dict()
        total = recovered = 0
        by_country, by_source = {}, {"S2": [0, 0], "S3": [0, 0]}
        for s1id, truth in eval_truth.items():
            cands = found_map.get(s1id, set())
            country = s1_country_map.get(s1id, "unknown")
            for mid in truth:
                total += 1
                hit = mid in cands
                recovered += hit
                c = by_country.setdefault(country, [0, 0])
                c[0] += hit
                c[1] += 1
                src = "S2" if mid.startswith("S2-") else "S3"
                by_source[src][0] += hit
                by_source[src][1] += 1
        return {
            "recall": recovered / total if total else None,
            "n_total_true_pairs": total,
            "n_recovered": recovered,
            "by_country": {c: (v[0] / v[1] if v[1] else None) for c, v in by_country.items()},
            "by_country_n": {c: v[1] for c, v in by_country.items()},
            "by_source": {s: (v[0] / v[1] if v[1] else None) for s, v in by_source.items()},
        }

    recall_uncapped = recall_of(union_all)
    recall_capped = recall_of(capped_all)

    tagged_all = pd.concat(
        [pd.read_parquet(PHASE3_DIR / f"tagged_{country}.parquet") for country in countries], ignore_index=True
    )
    per_block_recall = {
        name: recall_of(tagged_all.loc[tagged_all["block"] == name, ["s1_id", "cand_id"]])["recall"]
        for name in BLOCK_NAMES
    }
    marginal_recall_without = {
        name: recall_of(tagged_all.loc[tagged_all["block"] != name, ["s1_id", "cand_id"]])["recall"]
        for name in BLOCK_NAMES
    }

    cand_counts = capped_all.groupby("s1_id").size()
    cand_stats = {
        "avg_candidates_per_s1": float(cand_counts.mean()) if len(cand_counts) else 0,
        "p99_candidates_per_s1": float(cand_counts.quantile(0.99)) if len(cand_counts) else 0,
        "max_candidates_per_s1": int(cand_counts.max()) if len(cand_counts) else 0,
        "n_s1_with_zero_candidates": int(len(s1) - len(cand_counts)),
    }

    misses = worst_misses(union_all, eval_truth, limit=30)

    report = {
        "build_time_seconds": round(time.time() - t0, 1),
        "raw_pairs_per_block_total": raw_pairs_per_block_total,
        "n_union_pairs": len(union_all),
        "n_capped_pairs": len(capped_all),
        "candidates_per_s1_cap": blocking.CANDIDATES_PER_S1_CAP,
        "per_block_cap": blocking.PER_BLOCK_CAP,
        "candidate_stats": cand_stats,
        "recall_uncapped": recall_uncapped,
        "recall_capped": recall_capped,
        "recall_by_block_alone": per_block_recall,
        "recall_without_each_block": marginal_recall_without,
        "worst_misses": misses,
    }
    with open(PHASE3_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


def worst_misses(union_all: pd.DataFrame, eval_truth: dict, limit: int = 30) -> list:
    found_map = union_all.groupby("s1_id")["cand_id"].apply(set).to_dict()
    misses = []
    for s1id, truth in eval_truth.items():
        cands = found_map.get(s1id, set())
        for mid in truth:
            if mid not in cands:
                misses.append((s1id, mid))
    misses = misses[:limit]
    if not misses:
        return []

    s1_df = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet", columns=["entity_id", "business_name", "business_address", "country"]).set_index("entity_id")
    s2_df = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet", columns=["entity_id", "business_name", "business_address"])
    s3_df = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet", columns=["entity_id", "business_name", "business_address"])
    cand_map = pd.concat([s2_df, s3_df], ignore_index=True).set_index("entity_id")

    out = []
    for s1id, mid in misses:
        s1row, cdrow = s1_df.loc[s1id], cand_map.loc[mid]
        out.append(
            {
                "s1_id": s1id, "s1_name": s1row["business_name"], "s1_address": s1row["business_address"],
                "country": s1row["country"],
                "cand_id": mid, "cand_name": cdrow["business_name"], "cand_address": cdrow["business_address"],
            }
        )
    return out


if __name__ == "__main__":
    report = run()
    print(json.dumps({k: v for k, v in report.items() if k != "worst_misses"}, indent=2, default=str, ensure_ascii=False))
