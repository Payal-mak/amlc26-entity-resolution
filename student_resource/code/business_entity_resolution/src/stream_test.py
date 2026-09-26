"""Streaming test-set inference: never keeps the test features on disk.

Why this exists: the test set has ~94M candidate pairs (France ~13.6M, India ~45M,
US ~35M). Writing 46 float32 features for every pair (~17GB) filled the Kaggle disk,
and building them in one process was too slow for 12 hours. Here each country is
processed in three passes and only SMALL files survive:

  pass 1  per S1 batch (100k S1): block + cap -> (s1_id, cand_id, blocks, n_blocks,
          name_full_ratio) parquet, plus the batch's candidate lists (the rows of
          candidate_pairs.tsv). Optional --test-cap K keeps the top-K per S1.
  pass 2  ONE DuckDB pass over every batch's pairs of the country: rank_by_s1,
          gap_to_best_by_s1, reverse_rank -- computed against ALL S1 of the country
          (validation-slice semantics). Doing it per batch instead made
          reverse_rank ~5x too small (mean 11.5 -> 2.3 on the val slice) and cost
          about 0.4 F0.5 points at threshold 0.60.
  pass 3  per S1 batch: the 41 pairwise features on all cores (features.FeatureWorkers),
          the rank features from pass 2, the saved val_train fold models -> keep only
          pairs with probability >= PROBA_FLOOR as pred_{country}_{batch}.parquet;
          then delete the batch's pairs / rank files and DuckDB temp files.

`merge_predictions` (below) then runs the decision layer over the small prediction
files of one or several countries/notebooks and writes both TSVs.
"""

import gc
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from . import blocking, config, decide, features, io_utils, model, write_outputs

PROBA_FLOOR = 0.005
DEFAULT_BATCH_S1 = 100_000


# ------------------------------------------------------------------ logging
def resources(tag: str, extra_paths=()) -> None:
    """One line: free disk on the work dir (and /kaggle/working, /tmp when present) + process RSS."""
    import tempfile

    paths = [config.WORK_DIR, Path(tempfile.gettempdir()), *[Path(p) for p in extra_paths]]
    if Path("/kaggle/working").exists():
        paths.append(Path("/kaggle/working"))
    seen, parts = set(), []
    for p in paths:
        try:
            du = shutil.disk_usage(p)
        except OSError:
            continue
        key = (du.total, du.free // 10**8)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{p}: {du.free / 1e9:.1f}GB free of {du.total / 1e9:.0f}GB")
    try:
        import psutil

        me = psutil.Process()
        rss = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
        parts.append(f"RSS(all procs)={rss / 1e9:.1f}GB")
        parts.append(f"RAM available={psutil.virtual_memory().available / 1e9:.1f}GB")
    except Exception:
        pass
    print(f"    [resources {tag}] " + " | ".join(parts), flush=True)


def _clean_duckdb_tmp() -> None:
    """DuckDB deletes its spill files on close; this removes anything left after a crash or kill."""
    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()
    shutil.rmtree(config.DUCKDB_TMP_DIR, ignore_errors=True)
    config.DUCKDB_TMP_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ helpers
def load_country_frame(con, view: str, country: str, suffix_sets: dict, translit_map: dict) -> pd.DataFrame:
    df = con.execute(f"SELECT * FROM {view} WHERE country = ?", [country]).fetchdf()
    return blocking.add_normalized_columns(df, suffix_sets, translit_map)


def cap_topk(pairs: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the top-k candidates per S1 by a cheap score: name_full_ratio, then number of
    blocks that found the pair, then id (deterministic)."""
    pairs = pairs.sort_values(
        ["s1_id", "name_full_ratio", "n_blocks", "cand_id"], ascending=[True, False, False, True], kind="stable"
    )
    return pairs[pairs.groupby("s1_id").cumcount() < k]


def _name_ratio(pairs: pd.DataFrame, s1_names: pd.Series, cand_names: pd.Series) -> np.ndarray:
    a = s1_names.loc[pairs["s1_id"].values].tolist()
    b = cand_names.loc[pairs["cand_id"].values].tolist()
    return process.cpdist(a, b, scorer=fuzz.ratio, dtype=np.float32, workers=-1) if a else np.empty(0, np.float32)


def _candidate_lists(pairs: pd.DataFrame) -> pd.DataFrame:
    """(s1_id, cands): each S1's candidate ids, sorted, comma-joined -- one row of candidate_pairs.tsv."""
    ordered = pairs[["s1_id", "cand_id"]].sort_values(["s1_id", "cand_id"], kind="stable")
    return ordered.groupby("s1_id", sort=True)["cand_id"].agg(",".join).rename("cands").reset_index()


# ------------------------------------------------------------------ one country
def stream_country(
    country: str, s1_view: str, s2_view: str, s3_view: str, out_dir: Path, models, suffix_sets: dict,
    translit_map: dict, batch_size: int = DEFAULT_BATCH_S1, test_cap: int = None, n_workers: int = None,
) -> dict:
    """Run passes 1-3 for one country. Writes out_dir/pred/pred_{country}_{b}.parquet and
    out_dir/cands/cands_{country}_{b}.parquet (both small); returns counts for logging."""
    if isinstance(models, dict):
        raise NotImplementedError("streaming inference supports the fold-model ensemble only, not --two-stage")
    pred_dir, cand_dir = out_dir / "pred", out_dir / "cands"
    work = out_dir / "work" / country
    for d in (pred_dir, cand_dir, work):
        d.mkdir(parents=True, exist_ok=True)
    t_country = time.time()

    _clean_duckdb_tmp()
    con = io_utils.get_connection()
    io_utils.register_all_standard_views(con)
    s1_c = load_country_frame(con, s1_view, country, suffix_sets, translit_map)
    s2_c = load_country_frame(con, s2_view, country, suffix_sets, translit_map)
    s3_c = load_country_frame(con, s3_view, country, suffix_sets, translit_map)
    con.close()
    cand_c = pd.concat([s2_c, s3_c], ignore_index=True)
    del s2_c, s3_c
    n_s1 = len(s1_c)
    n_batches = max(1, -(-n_s1 // batch_size))
    print(f"  country={country}: s1={n_s1} cand={len(cand_c)} -> {n_batches} batch(es) of <= {batch_size}"
          + (f", test cap K={test_cap}" if test_cap else ""), flush=True)

    lookup = features.make_lookup(s1_c, cand_c)             # 4 columns per entity; used by pass 1 (name ratio) and pass 3
    s1_names, cand_names = lookup[0]["name_full"], lookup[1]["name_full"]
    resources(f"{country} after load")

    counts = {"union": 0, "capped": 0, "kept_after_cap": 0, "pred_rows": 0, "s1": n_s1}

    # ---- pass 1: blocking per S1 batch
    for b in range(n_batches):
        pairs_path = work / f"pairs_{b:03d}.parquet"
        cand_path = cand_dir / f"cands_{country}_{b:03d}.parquet"
        if (pred_dir / f"pred_{country}_{b:03d}.parquet").exists():
            continue                                         # batch fully done (restart)
        if pairs_path.exists() and cand_path.exists():
            continue
        tb = time.time()
        s1_batch = s1_c.iloc[b * batch_size:(b + 1) * batch_size]
        _clean_duckdb_tmp()
        con = io_utils.get_connection()
        blocking.register_suffix_table(con, suffix_sets)
        capped, _ = blocking.block_and_cap_country(con, s1_batch, cand_c)
        n_union = con.execute("SELECT count(*) FROM all_pairs_scored").fetchone()[0]   # before the per-S1 cap
        con.close()
        _clean_duckdb_tmp()
        capped = capped[["s1_id", "cand_id", "blocks", "n_blocks"]].copy()
        capped["name_full_ratio"] = _name_ratio(capped, s1_names, cand_names)
        n_capped = len(capped)
        if test_cap:
            capped = cap_topk(capped, test_cap)
        capped["batch"] = np.int32(b)
        capped.to_parquet(pairs_path, index=False)
        _candidate_lists(capped).to_parquet(cand_path, index=False)
        counts["union"] += n_union
        counts["capped"] += n_capped
        counts["kept_after_cap"] += len(capped)
        print(f"    {country} batch {b}/{n_batches - 1} blocking: s1={len(s1_batch)} union={n_union:,} -> "
              f"capped(75/S1)={n_capped:,} ({n_capped / max(len(s1_batch), 1):.1f}/S1)"
              + (f" -> top-{test_cap}={len(capped):,}" if test_cap else "") + f" ({time.time() - tb:.0f}s)", flush=True)
        del capped, s1_batch
        gc.collect()
        resources(f"{country} pass1 batch {b}")

    # ---- pass 2: rank features against ALL S1 of the country
    ctx_dir = work / "ctx"
    todo = [b for b in range(n_batches) if not (pred_dir / f"pred_{country}_{b:03d}.parquet").exists()]
    if todo:
        t0 = time.time()
        shutil.rmtree(ctx_dir, ignore_errors=True)
        con = io_utils.get_connection()
        glob = (work / "pairs_*.parquet").as_posix()
        con.execute(
            f"""
            COPY (
                SELECT s1_id, cand_id, batch, n_blocks,
                       ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY name_full_ratio DESC, n_blocks DESC, cand_id) AS rank_by_s1,
                       (MAX(name_full_ratio) OVER (PARTITION BY s1_id) - name_full_ratio) AS gap_to_best_by_s1,
                       ROW_NUMBER() OVER (PARTITION BY cand_id ORDER BY name_full_ratio DESC, n_blocks DESC, s1_id) AS reverse_rank
                FROM read_parquet('{glob}')
            ) TO '{ctx_dir.as_posix()}' (FORMAT PARQUET, PARTITION_BY (batch))
            """
        )
        con.close()
        _clean_duckdb_tmp()
        print(f"  country={country}: rank features over all S1 ({time.time() - t0:.0f}s)", flush=True)
        resources(f"{country} after pass2")

    # ---- pass 3: features + prediction per batch, keep only the small prediction file
    # The per-entity prep (street keys, house numbers, cleaned names, IDF over the whole country) is built only NOW,
    # after DuckDB is closed, and the big frames are freed right after: DuckDB's memory limit and the frames+prep
    # never have to fit in RAM together.
    t0 = time.time()
    prep = features.prepare_entities(s1_c, cand_c)         # once per country
    del s1_c, cand_c
    gc.collect()
    print(f"  country={country}: per-entity prep {time.time() - t0:.0f}s", flush=True)
    resources(f"{country} after prep")
    workers = features.FeatureWorkers(prep.attrs["idf"], n_workers)
    try:
        for b in todo:
            tb = time.time()
            pairs = pd.read_parquet(work / f"pairs_{b:03d}.parquet")
            ctx = pd.read_parquet(ctx_dir / f"batch={b}").drop(columns=["n_blocks"])   # n_blocks only served as a tie-break
            t_feat = time.time()
            feat = features.build_pair_features_parallel(pairs, lookup, prep, workers)
            t_feat = time.time() - t_feat
            feat = feat.merge(ctx, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
            for c in features.CONTEXT_FEATURE_COLUMNS:
                feat[c] = feat[c].astype(np.float32)
            X = feat[features.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            proba = model.predict_with_fold_models(models, X)
            keep = proba >= PROBA_FLOOR
            out = pd.DataFrame({
                "s1_id": feat["s1_id"].values[keep], "cand_id": feat["cand_id"].values[keep],
                "proba": proba[keep].astype(np.float32),
            })
            out["country"] = country
            out.to_parquet(pred_dir / f"pred_{country}_{b:03d}.parquet", index=False)
            counts["pred_rows"] += len(out)
            n_pairs = len(pairs)
            (work / f"pairs_{b:03d}.parquet").unlink(missing_ok=True)
            shutil.rmtree(ctx_dir / f"batch={b}", ignore_errors=True)
            del pairs, ctx, feat, X, proba, out
            gc.collect()
            _clean_duckdb_tmp()
            print(f"    {country} batch {b}/{n_batches - 1}: {n_pairs:,} pairs scored "
                  f"({n_pairs / max(time.time() - tb, 1e-9):,.0f} pairs/s overall, features {n_pairs / max(t_feat, 1e-9):,.0f}/s) "
                  f"-> {counts['pred_rows']:,} kept so far ({time.time() - tb:.0f}s)", flush=True)
            resources(f"{country} pass3 batch {b}")
    finally:
        workers.close()
    shutil.rmtree(work, ignore_errors=True)
    print(f"  country={country}: done in {(time.time() - t_country) / 60:.1f} min", flush=True)
    return counts


# ------------------------------------------------------------------ merge
def default_threshold() -> float:
    """The global threshold val_train selected (from its report), or None if it chose the label-free policy."""
    import json
    import re

    path = config.PARQUET_DIR / "phase4" / "report.json"
    if not path.exists():
        return None
    m = re.search(r"global_threshold\(t=([0-9.]+)\)", json.loads(path.read_text(encoding="utf-8")).get("policy", ""))
    return float(m.group(1)) if m else None


def find_files(dirs, prefix: str) -> list:
    files = []
    for d in dirs:
        files += sorted(Path(d).rglob(f"{prefix}_*.parquet"))
    return files


def merge_predictions(
    dirs, test_ids: set, s1_country: dict, out_dir: Path, threshold: float = None,
) -> dict:
    """Decision layer over the small prediction files, then both TSVs.

    dirs: folders containing pred_*.parquet and cands_*.parquet (searched recursively) --
    one folder per country notebook, or the local stream dir. one-to-one across ALL of
    them, then the global `threshold` (or, if None, the expected-F0.5 policy).
    """
    pred_files, cand_files = find_files(dirs, "pred"), find_files(dirs, "cands")
    if not pred_files or not cand_files:
        raise RuntimeError(f"no pred_*/cands_* parquet files under {list(dirs)}")
    print(f"[merge] {len(pred_files)} prediction files, {len(cand_files)} candidate files", flush=True)
    df = pd.concat([pd.read_parquet(p, columns=["s1_id", "cand_id", "proba"]) for p in pred_files], ignore_index=True)
    print(f"[merge] {len(df):,} scored pairs with proba >= {PROBA_FLOOR}", flush=True)
    resolved = decide.one_to_one(df, score_col="proba")
    if threshold is not None:
        preds = decide.apply_global_threshold(resolved, threshold)
        policy = f"global_threshold(t={threshold:.2f})"
    else:
        preds = decide.expected_f_beta_subset_selection(resolved)
        policy = "expected_f_beta_subset_selection"
    preds = {sid: preds.get(sid, set()) for sid in test_ids}
    out_dir.mkdir(parents=True, exist_ok=True)
    matching_path, candidate_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    write_outputs.write_matching_results(preds, test_ids, matching_path)
    n_with = write_outputs.write_candidate_lists_from_parquet([p.as_posix() for p in cand_files], test_ids, candidate_path)
    code, output = write_outputs.run_validator(matching_path, candidate_path, config.TEST_DIR)
    print(output)
    print("VALIDATOR:", "PASS" if code == 0 else "FAIL", flush=True)
    per_country = {}
    for c in sorted(set(s1_country.values())):
        ids = [i for i, cc in s1_country.items() if cc == c]
        per_country[c] = (len(ids), sum(1 for i in ids if preds.get(i)))
    print(f"[merge] policy={policy}; S1 with a match / total, by country:")
    for c, (tot, m) in per_country.items():
        print(f"   {c:8s} {m:>9,} / {tot:>9,}  = {m / max(tot, 1):6.1%}")
    print(f"[merge] S1 with at least one candidate: {n_with:,} / {len(test_ids):,}", flush=True)
    return {"policy": policy, "validator_exit": code, "per_country": per_country, "n_with_candidates": n_with}
