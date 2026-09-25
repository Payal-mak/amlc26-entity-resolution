"""Single end-to-end entry point: normalize -> blocking -> features -> train
-> predict -> write outputs. Driven entirely by CLI flags / AML_* env vars
(see src/config.py's module docstring) so the identical code runs unmodified
on the local dev machine (Windows, D: drive) and on Kaggle
(/kaggle/input/<dataset>/dataset, /kaggle/working, /kaggle/temp) -- nothing
in this file hardcodes a drive letter or OS-specific path.

Stages (run in order by --stage all, the default):
  normalize      learn suffix tokens + a transliteration token map from the
                 FULL train+test data (country-agnostic, so France gets a
                 suffix list too even with zero training rows).
  train_features block + build labeled features for train_source1 vs
                 train_source2/3, per country, streamed to parquet.
  train          GroupKFold LightGBM over every features_train_*.parquet;
                 reports OOF macro F0.5/precision/recall on the full train
                 set; saves the fold models.
  test_features  block + build UNlabeled features for test_source1 vs
                 test_source2/3, per country (also saves the raw candidate
                 pairs -- needed for candidate_pairs.tsv).
  predict        score test features with the saved fold models; run the
                 decision layer (no ground truth -- always the
                 expected-F0.5 subset-selection policy, see src/decide.py).
  write          write output/matching_results.tsv + candidate_pairs.tsv,
                 run utils/validate_submission.py, archive a versioned copy.

Every stage writes its output to disk before the next one starts, so a
crash partway through only costs the stage it was on -- re-run with
`--stage <name>` to resume instead of redoing everything with `--stage all`.

Usage (from code/business_entity_resolution/):
    # local dev machine (config.py's defaults already point at D:/amazon_ml_work)
    python -m scripts.run_pipeline

    # Kaggle
    python -m scripts.run_pipeline \\
        --data-dir /kaggle/input/<dataset-name>/dataset \\
        --work-dir /kaggle/temp/amazon_ml_work \\
        --output-dir /kaggle/working/output \\
        --duckdb-memory 20GB --duckdb-threads 4

    # resume just one stage (e.g. after fixing a bug in predict/write)
    python -m scripts.run_pipeline --stage predict
    python -m scripts.run_pipeline --stage write
"""

# MUST be the first import in this process, before pandas/duckdb/pyarrow --
# on the dev machine, importing lightgbm AFTER pandas has already loaded its
# native extensions causes a reproducible access violation deep inside
# LightGBM's C API (crashes on any data, in set_label, regardless of row
# count/dtype/contiguity -- isolated by bisecting import order; see
# PROJECT_LOG.md). Importing lightgbm first sidesteps whatever DLL/runtime
# it conflicts with. Every stage in this file eventually trains or predicts
# with LightGBM, so this has to be first even for stages that don't need it
# themselves (e.g. --stage normalize) -- there's no cheap way to defer it
# only to the stages that do.
import lightgbm  # noqa: F401,E402

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="Dataset root (contains train/, test/). Env: AML_DATA_DIR")
    p.add_argument("--work-dir", default=None, help="Large-artifact scratch dir (parquet/duckdb, never committed). Env: AML_WORK_DIR")
    p.add_argument("--output-dir", default=None, help="Final submission output dir. Env: AML_OUTPUT_DIR")
    p.add_argument("--duckdb-memory", default=None, help="DuckDB memory_limit, e.g. 1GB (local) / 20GB (Kaggle). Env: AML_DUCKDB_MEMORY_LIMIT")
    p.add_argument("--duckdb-threads", default=None, type=int, help="DuckDB thread count. Env: AML_DUCKDB_THREADS")
    p.add_argument(
        "--stage", default="all",
        choices=["all", "normalize", "train_features", "train", "test_features", "predict", "write"],
        help="Run one stage only, or 'all' (default) to run every stage in order.",
    )
    return p.parse_args()


def _apply_env(args: argparse.Namespace) -> None:
    """Set AML_* env vars from CLI flags. Must run BEFORE `from src import
    config` anywhere in this process -- config.py reads these at import time
    to build every path constant, so importing it first would bake in the
    wrong defaults.
    """
    if args.data_dir:
        os.environ["AML_DATA_DIR"] = args.data_dir
    if args.work_dir:
        os.environ["AML_WORK_DIR"] = args.work_dir
    if args.output_dir:
        os.environ["AML_OUTPUT_DIR"] = args.output_dir
    if args.duckdb_memory:
        os.environ["AML_DUCKDB_MEMORY_LIMIT"] = args.duckdb_memory
    if args.duckdb_threads:
        os.environ["AML_DUCKDB_THREADS"] = str(args.duckdb_threads)


_args = _parse_args()
_apply_env(_args)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import blocking, config, decide, evaluate, features, io_utils, model, write_outputs  # noqa: E402
from scripts import build_translit_token_map, phase2_normalization_report  # noqa: E402

PIPELINE_DIR = config.PARQUET_DIR / "pipeline"
TRAIN_FEATURES_DIR = PIPELINE_DIR / "train"
TEST_FEATURES_DIR = PIPELINE_DIR / "test"
TEST_CANDIDATES_DIR = TEST_FEATURES_DIR / "candidates"
MODEL_DIR = config.WORK_DIR / "models"
MODEL_PATH = MODEL_DIR / "lgbm_folds.pkl"
TRAIN_REPORT_PATH = PIPELINE_DIR / "train_report.json"
PREDICTIONS_PATH = PIPELINE_DIR / "test_predictions.parquet"


def _reset_duckdb() -> None:
    """Delete the shared DuckDB file between countries/stages -- the fix for
    a real CatalogException crash (CREATE OR REPLACE TABLE on an object that
    was previously a VIEW, left over from an earlier run; see PROJECT_LOG.md,
    Phase 3) and cheap insurance against any other stale-state carryover.
    """
    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()


def load_suffix_and_translit() -> tuple:
    """Load the Phase 2 outputs (learned from FULL train+test, country-
    agnostic) that every normalization step needs.
    """
    phase2_dir = config.PARQUET_DIR / "phase2"
    suffix_df = pd.read_parquet(phase2_dir / "suffix_token_candidates.parquet")
    suffix_sets = {c: set(g["token"]) for c, g in suffix_df.groupby("country")}
    tmap_path = phase2_dir / "translit_token_map.parquet"
    translit_map = {}
    if tmap_path.exists():
        tmap_df = pd.read_parquet(tmap_path)
        translit_map = dict(zip(tmap_df["translit_token"], tmap_df["latin_token"]))
    return suffix_sets, translit_map


def load_country_frame(con, view: str, country: str, suffix_sets: dict, translit_map: dict) -> pd.DataFrame:
    """One country's rows from a registered source view, normalized.

    Filters with WHERE country = ? in DuckDB so only that one country's rows
    are ever materialized in pandas -- never the whole (potentially
    multi-million-row) source file at once. Same per-country discipline as
    Phase 3/4, generalized here to whichever source view the caller wants
    (train_source1/2/3 or test_source1/2/3).
    """
    df = con.execute(f"SELECT * FROM {view} WHERE country = ?", [country]).fetchdf()
    return blocking.add_normalized_columns(df, suffix_sets, translit_map)


def load_ground_truth_map(con) -> dict:
    """{source1_entity_id: set(matched_entity_ids)} over the FULL train
    ground truth (not a validation slice -- this is the real training run).
    """
    gt = con.execute("SELECT source1_entity_id, matched_entity_ids FROM train_ground_truth").fetchdf()
    return {
        s1id: (set(ids.split(",")) if isinstance(ids, str) and ids.strip() else set())
        for s1id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }


def stage_normalize() -> None:
    print("[normalize] learning data-driven suffix tokens (full train+test)...", flush=True)
    phase2_normalization_report.run()
    print("[normalize] learning transliteration token map (full train ground truth)...", flush=True)
    build_translit_token_map.run()


def build_country_pairs_and_features(
    s1_view: str, s2_view: str, s3_view: str, out_dir: Path,
    label_map: dict = None, cand_out_dir: Path = None,
) -> None:
    """Shared block + feature-build loop for both train (labeled) and test
    (unlabeled) data, one country at a time -- so peak memory is bounded by
    one country's data, not the full multi-country pool (the fix for three
    separate OOM crashes in Phase 3; see PROJECT_LOG.md).

    Inputs: registered view names for S1/S2/S3 (already VARCHAR-typed CSV
    views over the raw TSVs), output dir for features_{country}.parquet,
    optional {s1_id: set(true_cand_id)} label map (omit entirely for test --
    no ground truth exists there), optional dir to also persist the raw
    capped candidate pairs (needed for test, to build candidate_pairs.tsv;
    not needed for train).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if cand_out_dir is not None:
        cand_out_dir.mkdir(parents=True, exist_ok=True)

    suffix_sets, translit_map = load_suffix_and_translit()

    # Every connection opened here is fresh (see _reset_duckdb), so the
    # standard source views must be re-registered each time -- they're just
    # view definitions over the raw TSVs (no data loaded yet), so this is
    # cheap to repeat.
    _reset_duckdb()
    con = io_utils.get_connection()
    io_utils.register_all_standard_views(con)
    countries = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM {s1_view} ORDER BY country").fetchall()]
    con.close()
    print(f"  countries: {countries}", flush=True)

    for country in countries:
        tc = time.time()
        _reset_duckdb()
        con = io_utils.get_connection()
        io_utils.register_all_standard_views(con)
        blocking.register_suffix_table(con, suffix_sets)

        s1_c = load_country_frame(con, s1_view, country, suffix_sets, translit_map)
        s2_c = load_country_frame(con, s2_view, country, suffix_sets, translit_map)
        s3_c = load_country_frame(con, s3_view, country, suffix_sets, translit_map)
        cand_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c

        capped_df, _ = blocking.block_and_cap_country(con, s1_c, cand_c)
        pairs_path = out_dir / f"_capped_{country}.parquet"
        capped_df.to_parquet(pairs_path, index=False)
        if cand_out_dir is not None:
            capped_df[["s1_id", "cand_id"]].to_parquet(cand_out_dir / f"capped_{country}.parquet", index=False)
        con.close()

        feat_out = out_dir / f"features_{country}.parquet"
        n_rows, n_pos = features.build_country_features(pairs_path, s1_c, cand_c, feat_out, label_map, country)
        pairs_path.unlink(missing_ok=True)
        print(
            f"  country={country}: s1={len(s1_c)} cand={len(cand_c)} pairs={n_rows} "
            f"positives={n_pos} ({time.time()-tc:.1f}s)", flush=True,
        )
        del s1_c, cand_c, capped_df
        gc.collect()
    _reset_duckdb()


def stage_train_features() -> None:
    print("[train_features] blocking + labeled features over the FULL train set...", flush=True)
    _reset_duckdb()
    con = io_utils.get_connection()
    io_utils.register_all_standard_views(con)
    label_map = load_ground_truth_map(con)
    con.close()
    build_country_pairs_and_features("train_source1", "train_source2", "train_source3", TRAIN_FEATURES_DIR, label_map)


def stage_test_features() -> None:
    print("[test_features] blocking + UNlabeled features over the FULL test set...", flush=True)
    build_country_pairs_and_features(
        "test_source1", "test_source2", "test_source3", TEST_FEATURES_DIR,
        label_map=None, cand_out_dir=TEST_CANDIDATES_DIR,
    )


def stage_train() -> dict:
    print("[train] loading train features...", flush=True)
    frames = [pd.read_parquet(p) for p in sorted(TRAIN_FEATURES_DIR.glob("features_*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[train] {len(df)} rows, {int(df['label'].sum())} positives", flush=True)

    _reset_duckdb()
    con = io_utils.get_connection()
    io_utils.register_all_standard_views(con)
    full_gt = load_ground_truth_map(con)
    s1_country = dict(con.execute("SELECT entity_id, country FROM train_source1").fetchall())
    con.close()
    # Every train S1 must be scored, including ones blocking found zero
    # candidates for (they still count in macro F0.5, usually as a missed
    # match unless they're a true singleton) -- not just the subset that
    # happens to appear in df.
    eval_ids = set(s1_country.keys())
    truths = {sid: full_gt.get(sid, set()) for sid in eval_ids}

    oof_proba, models, _ = model.train_oof(df, features.FEATURE_COLUMNS, n_folds=5, seed=config.RANDOM_SEED)
    df["proba"] = oof_proba
    print(f"[train] trained 5-fold GroupKFold LightGBM", flush=True)
    importance = model.feature_importance_report(models, features.FEATURE_COLUMNS)

    preds, policy = decide.decide(df, eval_ids, truths=truths)
    print(f"[train] decision policy selected: {policy}", flush=True)

    report = evaluate.score_report(preds, truths, group_of=s1_country)
    report["policy"] = policy
    report["n_train_rows"] = len(df)
    report["n_positives"] = int(df["label"].sum())
    report["n_eval_ids"] = len(eval_ids)
    report["top_features"] = importance.head(15).values.tolist()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"[train] macro F0.5={report.get('macro_f_beta')}", flush=True)
    return report


def stage_predict() -> dict:
    print("[predict] loading test features + trained models...", flush=True)
    frames = [pd.read_parquet(p) for p in sorted(TEST_FEATURES_DIR.glob("features_*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    del frames
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)

    df["proba"] = model.predict_with_fold_models(models, df[features.FEATURE_COLUMNS].values)

    _reset_duckdb()
    con = io_utils.get_connection()
    io_utils.register_source_view(con, "test_source1", config.TEST_SOURCE1)
    test_ids = {r[0] for r in con.execute("SELECT entity_id FROM test_source1").fetchall()}
    con.close()

    # No ground truth at real inference time -> decide() always falls back
    # to the expected-F0.5 subset-selection policy (see src/decide.py).
    preds, policy = decide.decide(df, test_ids, truths=None)
    n_matched = sum(1 for v in preds.values() if v)
    print(f"[predict] {n_matched} of {len(preds)} test S1 entities matched (policy={policy})", flush=True)

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_DIR / "predict_report.json", "w", encoding="utf-8") as f:
        json.dump({"policy": policy, "n_test_ids": len(test_ids), "n_matched": n_matched}, f, indent=2)

    pd.DataFrame(
        {
            "source1_entity_id": list(preds.keys()),
            "matched_entity_ids": [",".join(sorted(v)) for v in preds.values()],
        }
    ).to_parquet(PREDICTIONS_PATH, index=False)
    return {"preds": preds, "test_ids": test_ids, "policy": policy}


def stage_write() -> None:
    pred_df = pd.read_parquet(PREDICTIONS_PATH)
    preds = {
        row.source1_entity_id: (set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set())
        for row in pred_df.itertuples()
    }
    required_ids = set(preds.keys())

    candidates = {}
    for p in sorted(TEST_CANDIDATES_DIR.glob("capped_*.parquet")):
        cdf = pd.read_parquet(p)
        for s1id, group in cdf.groupby("s1_id")["cand_id"]:
            candidates[s1id] = set(group)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = config.OUTPUT_DIR / "matching_results.tsv"
    candidate_path = config.OUTPUT_DIR / "candidate_pairs.tsv"
    write_outputs.write_matching_results(preds, required_ids, matching_path)
    write_outputs.write_candidate_pairs(candidates, required_ids, candidate_path)
    print(f"[write] wrote {matching_path} and {candidate_path}", flush=True)

    code, output = write_outputs.run_validator(matching_path, candidate_path, config.TEST_DIR)
    print(output)

    report = json.loads(TRAIN_REPORT_PATH.read_text(encoding="utf-8")) if TRAIN_REPORT_PATH.exists() else None
    dest = write_outputs.save_versioned_copy(matching_path, candidate_path, report)
    print(f"[write] validator exit code {code}; versioned copy saved to {dest}", flush=True)
    if code != 0:
        print("[write] WARNING: validator reported blocking issues -- fix before submitting.", flush=True)


STAGES = {
    "normalize": stage_normalize,
    "train_features": stage_train_features,
    "train": stage_train,
    "test_features": stage_test_features,
    "predict": stage_predict,
    "write": stage_write,
}
STAGE_ORDER = ["normalize", "train_features", "train", "test_features", "predict", "write"]


def main() -> None:
    stages_to_run = STAGE_ORDER if _args.stage == "all" else [_args.stage]
    for name in stages_to_run:
        t0 = time.time()
        print(f"=== stage: {name} ===", flush=True)
        STAGES[name]()
        print(f"=== stage {name} done ({time.time()-t0:.1f}s) ===", flush=True)


if __name__ == "__main__":
    main()
