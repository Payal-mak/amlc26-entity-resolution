"""Scale-realistic validation (opt-in; not wired into --stage all).

Why: the leaderboard score (0.835) is well below the 45k validation slice's OOF F0.5
(0.9492). Hypothesis (PROJECT_LOG.md / branch discussion): the pipeline degrades once
the CANDIDATE POOL reaches real full-country size, not just because of France (which
has no labels at all to check against) -- per-block/per-S1 caps are absolute counts,
so a true match competes against more decoys as the pool grows, and rank features
(reverse_rank etc.) are computed over more competitors. Earlier evidence for exactly
this pattern: blocking pair recall 92.6% on a 2,000-S1 slice vs 79.07% (pre-recall-v2)
on the 45,000-S1 slice, at a similar candidates-per-S1 count -- pool size, not S1
count, degraded it.

What this script does, for ONE country (US or India -- the only two with train labels):
  1. Sample --n-s1 (default 40000) Source-1 ids from train_source1 for that country,
     EXCLUDING any id already used by the val slice (build_validation_split.py) or by
     Kaggle's own val_train run, so the saved val_train model was never trained on them.
  2. Run the SAME code path as the real test run (src.stream_test.stream_country: same
     batching, same country-wide rank features, same per-S1/per-block caps, same
     feature build, same saved fold models) with S1 restricted to the sample but S2/S3
     candidates = the FULL train pool for that country (train_source2 + train_source3,
     not sampled) -- reproducing the real test's candidate density while keeping the
     labeled/scored side small.
  3. Score the result against train_ground_truth.tsv with the exact official metric
     (src.evaluate), plus blocking pair recall and the same loss-bucket breakdown used
     earlier in this branch's error analysis (blocking miss / model miss / FP on
     singletons / FP extra on matched S1).

Cost note (see also PROJECT_LOG.md): unlike the real test run, this needs only ONE S1
batch (40k < the 100k batch size), so blocking runs once against the full candidate
pool instead of 7-9 times -- expect roughly 15-30 minutes on Kaggle per country, NOT
hours. It still touches the full real candidate pool (India: ~4.1M rows), so do not
run this at full scale on an 8GB laptop; use --sample (below) for a cheap local
smoke test of the code path only.

Usage (from code/business_entity_resolution/, after `run_pipeline --stage normalize`
and `--skip-full-train --stage val_train` have produced the phase2 files and the saved
fold models this script reuses):
    python -m scripts.scale_realistic_val --country India --n-s1 40000
    python -m scripts.scale_realistic_val --country US --n-s1 40000 --duckdb-memory 10GB

Local smoke test (tiny data, correctness only -- NOT a real measurement):
    python -m scripts.run_pipeline --sample 3000 --skip-full-train --stage normalize
    python -m scripts.run_pipeline --sample 3000 --skip-full-train --stage val_slice
    python -m scripts.run_pipeline --sample 3000 --skip-full-train --stage val_blocking
    python -m scripts.run_pipeline --sample 3000 --skip-full-train --stage val_features
    python -m scripts.run_pipeline --sample 3000 --skip-full-train --stage val_train
    python -m scripts.scale_realistic_val --sample 3000 --country India --n-s1 200
"""

# lightgbm must be imported before pandas -- see src/model.py's docstring.
import lightgbm  # noqa: F401,E402

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--country", required=True, choices=["US", "India"], help="Train has labels only for these two.")
    p.add_argument("--n-s1", type=int, default=40000, help="Number of Source-1 ids to sample (default 40000).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=None, help="Env: AML_DATA_DIR")
    p.add_argument("--work-dir", default=None, help="Env: AML_WORK_DIR")
    p.add_argument("--duckdb-memory", default=None, help="Env: AML_DUCKDB_MEMORY_LIMIT")
    p.add_argument("--duckdb-threads", type=int, default=None, help="Env: AML_DUCKDB_THREADS")
    p.add_argument("--feature-workers", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None, help="Default: whatever val_train selected (its report.json).")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="LOCAL SMOKE TEST ONLY: use scripts.make_sample_dataset's generated tiny dataset instead of "
                        "the real one (must match the --sample value used for run_pipeline's normalize/val_* stages).")
    args = p.parse_args()
    if args.data_dir:
        import os
        os.environ["AML_DATA_DIR"] = args.data_dir
    if args.work_dir:
        import os
        os.environ["AML_WORK_DIR"] = args.work_dir
    if args.duckdb_memory:
        import os
        os.environ["AML_DUCKDB_MEMORY_LIMIT"] = args.duckdb_memory
    if args.duckdb_threads:
        import os
        os.environ["AML_DUCKDB_THREADS"] = str(args.duckdb_threads)
    return args


_args = _parse_args()

import pandas as pd  # noqa: E402 (re-import after env vars are set; harmless, keeps import order simple)

from src import config, decide, evaluate, features, io_utils, model, stream_test  # noqa: E402

OUT_DIR = config.PARQUET_DIR / "scaleval" / _args.country


def load_suffix_and_translit() -> tuple:
    phase2_dir = config.PARQUET_DIR / "phase2"
    suffix_df = pd.read_parquet(phase2_dir / "suffix_token_candidates.parquet")
    suffix_sets = {c: set(g["token"]) for c, g in suffix_df.groupby("country")}
    tm = phase2_dir / "translit_token_map.parquet"
    translit_map = {}
    if tm.exists():
        tmap_df = pd.read_parquet(tm)
        translit_map = dict(zip(tmap_df["translit_token"], tmap_df["latin_token"]))
    return suffix_sets, translit_map


def excluded_ids() -> set:
    """Every S1 id the saved val_train model could have trained on: the 45k validation
    slice's source1.parquet, EVAL and CONTEXT both. If that file isn't present (a Kaggle
    'carry' folder only keeps the model + report, not the slice itself), this returns an
    empty set and the model_provenance note below records that the check was skipped."""
    p = config.PARQUET_DIR / "val_slice" / "source1.parquet"
    if not p.exists():
        return set()
    return set(pd.read_parquet(p, columns=["entity_id"])["entity_id"])


def sample_s1(country: str, n: int, seed: int, exclude: set) -> pd.DataFrame:
    con = io_utils.get_connection(read_only=False)
    io_utils.register_source_view(con, "train_source1", config.TRAIN_SOURCE1)
    df = con.execute(
        "SELECT entity_id, business_name, business_address, country FROM train_source1 WHERE country = ?", [country]
    ).fetchdf()
    con.close()
    if exclude:
        before = len(df)
        df = df[~df.entity_id.isin(exclude)]
        print(f"[scale_val] excluded {before - len(df)} S1 already in the validation slice (leakage guard)", flush=True)
    if n > len(df):
        raise SystemExit(f"--n-s1 {n} > available {country} train S1 after exclusions ({len(df)})")
    return df.sample(n, random_state=seed).sort_values("entity_id").reset_index(drop=True)


def load_ground_truth(ids: set) -> dict:
    con = io_utils.get_connection(read_only=False)
    io_utils.register_ground_truth_view(con)
    gt = con.execute(
        f"SELECT source1_entity_id, matched_entity_ids FROM train_ground_truth "
        f"WHERE source1_entity_id IN ({','.join('?' * len(ids))})", list(ids)
    ).fetchdf()
    con.close()
    return {
        s1id: (set(v.split(",")) if isinstance(v, str) and v.strip() else set())
        for s1id, v in zip(gt.source1_entity_id, gt.matched_entity_ids)
    }


def default_threshold_from_val_report() -> float:
    t = stream_test.default_threshold()
    return t if t is not None else 0.5


def loss_buckets(preds: dict, truths: dict, capped_by_s1: dict) -> pd.DataFrame:
    """Same accounting as the earlier local error-analysis pass: for each S1, split its
    lost F0.5 into (a) true matches missing from the candidate pool (blocking miss),
    (b) true matches present but not predicted (model/threshold miss), (c) false
    positives, split into singleton-FP and extra-FP on an S1 that does have matches."""
    rows = []
    for s1, truth in truths.items():
        cand = capped_by_s1.get(s1, set())
        pred = preds.get(s1, set())
        f_act = evaluate.f_beta_score(pred, truth)
        ceiling = evaluate.f_beta_score(truth & cand, truth)
        no_fp = evaluate.f_beta_score(pred & truth, truth)
        block_loss = 1 - ceiling
        model_loss = ceiling - no_fp
        fp_loss = no_fp - f_act
        rows.append({
            "s1": s1, "singleton": not truth, "f": f_act,
            "block": block_loss, "model": model_loss,
            "fp_singleton": fp_loss if not truth else 0.0,
            "fp_extra": fp_loss if truth else 0.0,
        })
    return pd.DataFrame(rows)


def run() -> dict:
    t0 = time.time()
    country = _args.country
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[scale_val] country={country} n_s1={_args.n_s1} seed={_args.seed} data_dir={config.DATA_DIR}", flush=True)

    excl = excluded_ids()
    s1_sample = sample_s1(country, _args.n_s1, _args.seed, excl)
    ids = set(s1_sample.entity_id)
    print(f"[scale_val] sampled {len(ids)} {country} S1 (fixed seed {_args.seed})", flush=True)

    suffix_sets, translit_map = load_suffix_and_translit()

    model_path = config.WORK_DIR / "models" / "lgbm_folds.pkl"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found -- run `run_pipeline --skip-full-train --stage val_train` first.")
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    print("[scale_val] loaded the saved val_train fold models (not retrained here)", flush=True)

    stream_test.resources(f"{country} scale_val start")
    counts = stream_test.stream_country(
        country, "train_source1", "train_source2", "train_source3", OUT_DIR, models, suffix_sets, translit_map,
        batch_size=max(_args.n_s1, stream_test.DEFAULT_BATCH_S1), n_workers=_args.feature_workers,
        s1_id_filter=ids,
    )
    print(f"[scale_val] streaming done ({time.time() - t0:.0f}s): {counts}", flush=True)

    truths = load_ground_truth(ids)
    threshold = _args.threshold if _args.threshold is not None else default_threshold_from_val_report()

    pred_files = sorted(OUT_DIR.glob("pred/pred_*.parquet"))
    cand_files = sorted(OUT_DIR.glob("cands/cands_*.parquet"))
    scored = pd.concat([pd.read_parquet(p) for p in pred_files], ignore_index=True)
    resolved = decide.one_to_one(scored, score_col="proba")
    preds = decide.apply_global_threshold(resolved, threshold)
    preds = {sid: preds.get(sid, set()) for sid in ids}

    report = evaluate.score_report(preds, truths)
    report["threshold"] = threshold
    report["country"] = country
    report["n_s1"] = len(ids)

    # blocking pair recall, from the candidate lists (candidate_pairs.tsv equivalent)
    cand_lists = pd.concat([pd.read_parquet(p) for p in cand_files], ignore_index=True)
    capped_by_s1 = {row.s1_id: set(row.cands.split(",")) if row.cands else set() for row in cand_lists.itertuples()}
    n_true_pairs = sum(len(t) for t in truths.values())
    n_recovered = sum(len(t & capped_by_s1.get(s, set())) for s, t in truths.items())
    report["blocking_pair_recall"] = n_recovered / n_true_pairs if n_true_pairs else None
    report["avg_candidates_per_s1"] = cand_lists.cands.map(lambda s: 0 if not s else s.count(",") + 1).mean()

    losses = loss_buckets(preds, truths, capped_by_s1)
    n = len(losses)
    bucket = {
        "blocking_miss_pts": losses["block"].sum() / n * 100,
        "model_miss_pts": losses["model"].sum() / n * 100,
        "fp_singleton_pts": losses["fp_singleton"].sum() / n * 100,
        "fp_extra_pts": losses["fp_extra"].sum() / n * 100,
        "total_loss_pts": (1 - losses["f"]).sum() / n * 100,
    }
    report["loss_buckets"] = bucket

    print("\n" + "=" * 78)
    print(f"SCALE-REALISTIC VALIDATION -- {country}, n_s1={len(ids)}, full real candidate pool")
    print("=" * 78)
    print(f"  macro F0.5: {report['macro_f_beta']:.4f}   micro P: {report['micro_precision']:.4f}   "
          f"micro R: {report['micro_recall']:.4f}   threshold: {threshold}")
    print(f"  blocking pair recall: {report['blocking_pair_recall']:.4f}   "
          f"avg candidates/S1: {report['avg_candidates_per_s1']:.1f}")
    print(f"  loss buckets (points out of 100): {json.dumps(bucket, indent=4)}")
    print("=" * 78, flush=True)

    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"[scale_val] report written to {OUT_DIR / 'report.json'} ({time.time() - t0:.0f}s total)", flush=True)
    return report


if __name__ == "__main__":
    run()
