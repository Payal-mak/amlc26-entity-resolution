"""Build pairwise features for every Phase 3 capped candidate pair, per
country, writing straight to parquet on D: (never holding both countries'
feature matrices in memory at once -- the same discipline as Phase 3, after
three OOM crashes there).

Rewritten after a 4th crash (see PROJECT_LOG.md): the previous version read
an entire country's capped_{country}.parquet (6.7M rows for India) into one
pandas DataFrame and ran fillna/str.contains over all of it at once, which
failed outright once free system RAM dropped below ~500MB (Chrome/VS Code
memory pressure, not this job's own growth -- it died 32s in).

Uses whatever capped_{country}.parquet Phase 3 last wrote (currently the
B2_MAX_CAND_TOKEN_DF=400 run -- see PROJECT_LOG.md: the 150-vs-400 difference
was measured negligible, and regenerating clean 150 artifacts wasn't worth
another 9-minute run given the time-box).

The actual streaming/memory-safe machinery (chunked batches, the DuckDB
context-feature pass) lives in src/features.py as build_country_features,
shared with scripts/run_pipeline.py's real train/test build -- this script
only handles the validation-slice-specific plumbing (loading the slice,
building the label map from ground_truth_in_slice.parquet).

Usage (from code/business_entity_resolution/):
    python -m scripts.phase4_build_features
"""

import gc
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blocking, config, features  # noqa: E402

VAL_SLICE_DIR = config.PARQUET_DIR / "val_slice"
PHASE2_DIR = config.PARQUET_DIR / "phase2"
PHASE3_DIR = config.PARQUET_DIR / "phase3"
PHASE4_DIR = config.PARQUET_DIR / "phase4"


def load_normalized_slice() -> tuple:
    suffix_df = pd.read_parquet(PHASE2_DIR / "suffix_token_candidates.parquet")
    suffix_sets = {c: set(g["token"]) for c, g in suffix_df.groupby("country")}
    tmap_df = pd.read_parquet(PHASE2_DIR / "translit_token_map.parquet")
    translit_map = dict(zip(tmap_df["translit_token"], tmap_df["latin_token"]))

    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet")
    s2 = pd.read_parquet(VAL_SLICE_DIR / "source2.parquet")
    s3 = pd.read_parquet(VAL_SLICE_DIR / "source3.parquet")
    s1 = blocking.add_normalized_columns(s1, suffix_sets, translit_map)
    s2 = blocking.add_normalized_columns(s2, suffix_sets, translit_map)
    s3 = blocking.add_normalized_columns(s3, suffix_sets, translit_map)
    cand = pd.concat([s2, s3], ignore_index=True)
    return s1, cand


def build_labels() -> dict:
    gt = pd.read_parquet(VAL_SLICE_DIR / "ground_truth_in_slice.parquet")
    result = {}
    for s1id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        result[s1id] = set(ids.split(",")) if isinstance(ids, str) and ids.strip() else set()
    return result


def run() -> None:
    t0 = time.time()
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    s1, cand = load_normalized_slice()
    print(f"loaded+normalized: s1={len(s1)} cand={len(cand)} ({time.time()-t0:.1f}s)", flush=True)

    label_map = build_labels()

    for country in sorted(s1["country"].unique()):
        tc = time.time()
        s1_c = s1[s1["country"] == country]
        cand_c = cand[cand["country"] == country]
        pairs_path = PHASE3_DIR / f"capped_{country}.parquet"
        out_path = PHASE4_DIR / f"features_{country}.parquet"

        n_rows, n_pos = features.build_country_features(pairs_path, s1_c, cand_c, out_path, label_map, country)
        print(
            f"  country={country}: features done, pairs={n_rows} positives={n_pos} "
            f"({time.time()-tc:.1f}s) -> {out_path.name}",
            flush=True,
        )

        del s1_c, cand_c
        gc.collect()

    print(f"done ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    run()
