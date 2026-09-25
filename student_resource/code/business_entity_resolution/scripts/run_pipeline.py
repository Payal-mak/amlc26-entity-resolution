"""Single end-to-end entry point: normalize -> validation-slice OOF report ->
blocking (train + test) -> features -> fit the final model -> test inference
-> decision layer -> write output/matching_results.tsv + candidate_pairs.tsv.
Driven entirely by CLI flags / AML_* env vars (see src/config.py's module
docstring) so the identical code runs unmodified locally and on Kaggle --
nothing in this file hardcodes a drive letter, OS-specific path, or a
laptop-safe compromise (see src/model.py's DEFAULT_PARAMS for why those were
reverted: real full-size runs happen on Kaggle now, not this 8GB machine).

Stages (run in order by --stage all, the default):
  normalize      learn suffix tokens + a transliteration token map from the
                 FULL train+test data (country-agnostic, so France gets a
                 suffix list too even with zero training rows).
  val_slice      build the held-out validation slice from dataset/train/
                 (scripts/build_validation_split.py) -- --val-target-total
                 controls its size; small for a local smoke test.
  val_blocking   block + cap candidates on that slice, report recall
                 (scripts/phase3_blocking_report.py).
  val_features   build labeled features for the slice
                 (scripts/phase4_build_features.py).
  val_train      GroupKFold LightGBM OOF training + decision layer on the
                 slice; prints macro F0.5/precision/recall/singleton
                 accuracy, overall and per country -- THE headline validation
                 number (scripts/phase4_train_and_decide.py).
  train_features block + build labeled features for the FULL train_source1
                 vs train_source2/3, per country, streamed to parquet.
  train          fit the model actually used for test inference: GroupKFold
                 LightGBM over every features_train_*.parquet (also reports
                 its own OOF numbers on the full train set as a sanity
                 check); saves the fold models.
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
    # local dev machine -- SMOKE TEST ONLY, tiny validation slice, no real
    # train/test-scale run (see PROJECT_LOG.md: repeated local OOMs on the
    # real data even after several rounds of memory-safety fixes)
    python -m scripts.run_pipeline --val-target-total 2000 --stage normalize
    python -m scripts.run_pipeline --val-target-total 2000 --stage val_slice
    ... (val_blocking, val_features, val_train to check the wiring end to end)

    # Kaggle -- the real run
    python -m scripts.run_pipeline \\
        --data-dir /kaggle/input/<dataset-name>/dataset \\
        --work-dir /tmp/work \\
        --output-dir /kaggle/working/output \\
        --duckdb-memory 20GB --duckdb-threads 4

    # resume just one stage (e.g. after fixing a bug in predict/write)
    python -m scripts.run_pipeline --stage predict
    python -m scripts.run_pipeline --stage write
"""

# MUST be the first import in this process, before pandas/duckdb/pyarrow --
# on the local dev machine, importing lightgbm AFTER pandas has already
# loaded its native extensions causes a reproducible access violation deep
# inside LightGBM's C API (crashes on any data, in set_label, regardless of
# row count/dtype/contiguity -- isolated by bisecting import order; see
# PROJECT_LOG.md). Importing lightgbm first sidesteps whatever DLL/runtime
# it conflicts with. Not yet confirmed whether Kaggle's environment has the
# same issue; keeping the guard everywhere costs nothing if it doesn't. Every
# stage in this file eventually trains or predicts with LightGBM, so this has
# to be first even for stages that don't need it themselves.
import lightgbm  # noqa: F401,E402

import argparse
import datetime
import gc
import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover -- degrades to no peak-memory logging
    psutil = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="Dataset root (contains train/, test/). Env: AML_DATA_DIR")
    p.add_argument("--work-dir", default=None, help="Large-artifact scratch dir (parquet/duckdb, never committed). Env: AML_WORK_DIR")
    p.add_argument("--output-dir", default=None, help="Final submission output dir. Env: AML_OUTPUT_DIR")
    p.add_argument("--duckdb-memory", default=None, help="DuckDB memory_limit, e.g. 20GB (Kaggle default). Env: AML_DUCKDB_MEMORY_LIMIT")
    p.add_argument("--duckdb-threads", default=None, type=int, help="DuckDB thread count (4 default). Env: AML_DUCKDB_THREADS")
    p.add_argument("--lgbm-threads", default=None, type=int, help="LightGBM n_jobs (4 default). Env: AML_LGBM_THREADS")
    p.add_argument("--lgbm-max-bin", default=None, type=int, help="LightGBM max_bin (255 default -- the library's own default). Env: AML_LGBM_MAX_BIN")
    p.add_argument("--lgbm-two-round", action="store_true", help="Force LightGBM two_round=True (off by default; a laptop-memory compromise, not needed on Kaggle). Env: AML_LGBM_TWO_ROUND")
    p.add_argument("--two-stage", action="store_true", help="Use the two-stage model (stage 2 re-scores with context features rebuilt from stage-1 OOF probabilities; off by default). Env: AML_TWO_STAGE")
    p.add_argument(
        "--train-source", default="full", choices=["full", "val"],
        help="Data the FINAL model (the `train` stage) is fit on. 'full' (default) = every train pair from the "
             "train_features stage (~100M rows on the real data: does not fit in 30GB of RAM). 'val' = the "
             "validation-slice features already built by val_features (EVAL+CONTEXT, ~12M rows at the default "
             "--val-target-total); with 'val', `--stage all` skips train_features.",
    )
    p.add_argument(
        "--sample", default=0, type=int, metavar="N",
        help="LOCAL CRASH TESTING ONLY (off by default): swap the data dir for a generated sample of ~N Source-1 "
             "rows per split (scripts/make_sample_dataset.py) for every stage run in this invocation. Do not use "
             "with val_slice/val_blocking (they need real geographic diversity) and never for a real run.",
    )
    p.add_argument("--val-target-total", default=45000, type=int, help="Validation-slice size (scripts/build_validation_split.py). Use a small value (e.g. 2000) for a local smoke test.")
    p.add_argument(
        "--stage", default="all",
        choices=[
            "all", "normalize", "val_slice", "val_blocking", "val_features", "val_train",
            "train_features", "train", "test_features", "predict", "write",
        ],
        help="Run one stage only, or 'all' (default) to run every stage in order.",
    )
    return p.parse_args()


def _apply_env(args: argparse.Namespace) -> None:
    """Set AML_* env vars from CLI flags. Must run BEFORE `from src import
    config` anywhere in this process -- config.py reads these at import time
    to build every path/param constant, so importing it first would bake in
    the wrong values.
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
    if args.lgbm_threads:
        os.environ["AML_LGBM_THREADS"] = str(args.lgbm_threads)
    if args.lgbm_max_bin:
        os.environ["AML_LGBM_MAX_BIN"] = str(args.lgbm_max_bin)
    if args.lgbm_two_round:
        os.environ["AML_LGBM_TWO_ROUND"] = "true"
    if args.two_stage:
        os.environ["AML_TWO_STAGE"] = "true"
    if args.sample:
        _use_sample_dataset(args)


def _use_sample_dataset(args: argparse.Namespace) -> None:
    """--sample N: build (once) a tiny dataset under <work dir>/sample_dataset and point AML_DATA_DIR at it.
    Runs before src.config is imported, so it works out the real data/work dirs itself."""
    repo_root = Path(__file__).resolve().parents[3]
    real_data = Path(args.data_dir or os.environ.get("AML_DATA_DIR") or repo_root / "dataset")
    default_work = "D:/amazon_ml_work" if Path("D:/").exists() else str(repo_root / ".work")
    work = Path(args.work_dir or os.environ.get("AML_WORK_DIR") or default_work)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import make_sample_dataset

    out = make_sample_dataset.build_sample(real_data, work / "sample_dataset", args.sample)
    os.environ["AML_DATA_DIR"] = str(out)
    print(f"[sample] --sample {args.sample}: using {out} instead of the real dataset", flush=True)


_args = _parse_args()
_apply_env(_args)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src import blocking, config, decide, evaluate, features, io_utils, model, write_outputs  # noqa: E402
from scripts import (  # noqa: E402
    build_translit_token_map,
    build_validation_split,
    phase2_normalization_report,
    phase3_blocking_report,
    phase4_build_features,
    phase4_train_and_decide,
)

PIPELINE_DIR = config.PARQUET_DIR / "pipeline"
TRAIN_FEATURES_DIR = PIPELINE_DIR / "train"
TEST_FEATURES_DIR = PIPELINE_DIR / "test"
TEST_CANDIDATES_DIR = TEST_FEATURES_DIR / "candidates"
MODEL_DIR = config.WORK_DIR / "models"
MODEL_PATH = MODEL_DIR / "lgbm_folds.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
TRAIN_REPORT_PATH = PIPELINE_DIR / "train_report.json"
PREDICTIONS_PATH = PIPELINE_DIR / "test_predictions.parquet"


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _rss_mb():
    """Current process RSS in MB, or None if psutil isn't installed."""
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


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


def _print_metrics_report(report: dict, title: str) -> None:
    """Print a clearly-delimited, human-scannable block for an
    evaluate.score_report() dict -- overall macro F0.5 (the official metric),
    micro precision/recall, singleton accuracy, and the per-country
    breakdown. Meant to stand out in a long Kaggle log.
    """
    print("", flush=True)
    print("=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)
    print(f"  policy: {report.get('policy')}", flush=True)
    print(
        f"  n_entities={report.get('n_entities')}  "
        f"n_train_rows={report.get('n_train_rows')}  n_positives={report.get('n_positives')}",
        flush=True,
    )
    f_beta = report.get("macro_f_beta")
    print(f"  macro F0.5 (OFFICIAL METRIC): {f_beta:.4f}" if f_beta is not None else "  macro F0.5: n/a", flush=True)
    mp, mr = report.get("micro_precision"), report.get("micro_recall")
    print(f"  micro precision: {mp:.4f}  micro recall: {mr:.4f}" if mp is not None else "  micro precision/recall: n/a", flush=True)
    print(f"  singleton accuracy: {report.get('singleton_accuracy')}", flush=True)
    by_group = report.get("by_group") or {}
    if by_group:
        print("  by country:", flush=True)
        for country in sorted(by_group):
            g = by_group[country]
            print(
                f"    {country:12s} n={g['n']:>7}  F0.5={g['f_beta']:.4f}  "
                f"precision={g['precision']:.4f}  recall={g['recall']:.4f}",
                flush=True,
            )
    print("=" * 78, flush=True)
    print("", flush=True)


def stage_normalize() -> None:
    print("[normalize] learning data-driven suffix tokens (full train+test)...", flush=True)
    phase2_normalization_report.run()
    print("[normalize] learning transliteration token map (full train ground truth)...", flush=True)
    build_translit_token_map.run()


def stage_val_slice() -> None:
    print(f"[val_slice] building validation slice, target_total={_args.val_target_total}...", flush=True)
    build_validation_split.build(_args.val_target_total)


def stage_val_blocking() -> None:
    print("[val_blocking] blocking + capping candidates on the validation slice...", flush=True)
    phase3_blocking_report.run()


def stage_val_features() -> None:
    print("[val_features] building labeled features on the validation slice...", flush=True)
    phase4_build_features.run()


def stage_val_train() -> dict:
    print("[val_train] GroupKFold LightGBM OOF training + decision layer on the validation slice...", flush=True)
    report = phase4_train_and_decide.run()
    _print_metrics_report(report, "VALIDATION-SLICE OOF RESULTS (scripts/phase4_train_and_decide.py)")
    return report


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

        # Same three steps as blocking.block_and_cap_country, but the capped pairs are
        # written to parquet BY DUCKDB instead of being fetched into a pandas frame first:
        # at full scale one country is 40M+ pairs (several GB of Python strings).
        block_views = blocking.run_all_blocks(con, s1_c, cand_c)
        capped_view = blocking.cap_candidates(con, blocking.union_and_score(con, block_views))
        pairs_path = out_dir / f"_capped_{country}.parquet"
        io_utils.export_parquet(con, f"SELECT * FROM {capped_view}", pairs_path)
        if cand_out_dir is not None:
            io_utils.export_parquet(con, f"SELECT s1_id, cand_id FROM {capped_view}", cand_out_dir / f"capped_{country}.parquet")
        con.close()

        feat_out = out_dir / f"features_{country}.parquet"
        n_rows, n_pos = features.build_country_features(pairs_path, s1_c, cand_c, feat_out, label_map, country)
        pairs_path.unlink(missing_ok=True)
        print(
            f"  country={country}: s1={len(s1_c)} cand={len(cand_c)} pairs={n_rows} "
            f"positives={n_pos} ({time.time()-tc:.1f}s)", flush=True,
        )
        del s1_c, cand_c
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


def _save_feature_columns() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(features.FEATURE_COLUMNS), encoding="utf-8")


def _check_feature_columns() -> None:
    """The model was fit on a specific ordered column list; scoring with any other
    list/order would silently feed features to the wrong trees."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise RuntimeError(f"{FEATURE_COLUMNS_PATH} missing -- run the `train` stage (with this code version) first.")
    trained_on = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    if trained_on != features.FEATURE_COLUMNS:
        raise RuntimeError(
            "FEATURE_COLUMNS changed since the model was trained: "
            f"missing now={sorted(set(trained_on) - set(features.FEATURE_COLUMNS))}, "
            f"new={sorted(set(features.FEATURE_COLUMNS) - set(trained_on))}, or the order differs. Re-run `train`."
        )


def stage_train() -> dict:
    """Fit the model actually used for test inference. Also reports its own OOF
    numbers as a sanity check.

    --train-source full (default): every pair from `train_features` (the FULL train
    set). --train-source val: the validation-slice features from `val_features`
    (EVAL+CONTEXT design, realistic decoy density) -- what a 30GB machine can
    actually hold; see the flag's help text.
    """
    if _args.train_source == "val":
        print("[train] --train-source val: loading the validation-slice features...", flush=True)
        frames = [pd.read_parquet(p) for p in sorted(phase4_train_and_decide.PHASE4_DIR.glob("features_*.parquet"))]
        if not frames:
            raise RuntimeError("no validation-slice features found -- run the val_features stage first")
        eval_ids, truths, s1_country = phase4_train_and_decide.load_eval_ids_and_truth_and_country()
    else:
        print("[train] loading full-train features...", flush=True)
        frames = [pd.read_parquet(p) for p in sorted(TRAIN_FEATURES_DIR.glob("features_*.parquet"))]
        _reset_duckdb()
        con = io_utils.get_connection()
        io_utils.register_all_standard_views(con)
        full_gt = load_ground_truth_map(con)
        s1_country = dict(con.execute("SELECT entity_id, country FROM train_source1").fetchall())
        con.close()
        # Every train S1 must be scored, including ones blocking found zero candidates for
        # (they still count in macro F0.5) -- not just the subset that appears in df.
        eval_ids = set(s1_country.keys())
        truths = {sid: full_gt.get(sid, set()) for sid in eval_ids}
    df = pd.concat(frames, ignore_index=True)
    del frames
    n_rows, n_positives = len(df), int(df["label"].sum())
    print(f"[train] {n_rows} rows, {n_positives} positives", flush=True)

    X, y, groups = model.build_xy(df, features.FEATURE_COLUMNS)
    s1_ids, cand_ids = df["s1_id"].values, df["cand_id"].values
    # Free the full features dataframe before training -- see src/model.py's
    # build_xy docstring / PROJECT_LOG.md for the real OOM this avoids.
    del df
    gc.collect()

    if config.TWO_STAGE:
        oof_proba, info = model.train_two_stage_oof(
            X, y, groups, s1_ids, cand_ids, features.FEATURE_COLUMNS, n_folds=5, seed=config.RANDOM_SEED
        )
        importance = info["importance"]
        print("[train] trained 2-stage 5-fold GroupKFold LightGBM (OOF); fitting the full-train stage-1/stage-2 models predict uses", flush=True)
        models = model.fit_two_stage_final(X, y, s1_ids, cand_ids, info["stage1_oof"], seed=config.RANDOM_SEED)
        del info
    else:
        oof_proba, models, _ = model.train_oof(X, y, groups, n_folds=5, seed=config.RANDOM_SEED)
        print("[train] trained 5-fold GroupKFold LightGBM -- these are the models predict/write use", flush=True)
        importance = model.feature_importance_report(models, features.FEATURE_COLUMNS)
    del X, y, groups
    gc.collect()

    df = pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids, "proba": oof_proba})
    preds, policy = decide.decide(df, eval_ids, truths=truths)

    report = evaluate.score_report(preds, truths, group_of=s1_country)
    report["policy"] = policy
    report["train_source"] = _args.train_source
    report["two_stage"] = config.TWO_STAGE
    report["n_train_rows"] = n_rows
    report["n_positives"] = n_positives
    report["n_eval_ids"] = len(eval_ids)
    report["top_features"] = importance.head(15).values.tolist()
    _print_metrics_report(report, "TRAIN-STAGE OOF RESULTS (--train-source %s)" % _args.train_source)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    _save_feature_columns()

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


# Pairs the model scores below this are dropped BEFORE the decision layer: at full
# scale the scored test table is ~75M pairs (far too big for pandas), and a pair under
# 0.5% can neither be picked nor change one-to-one assignment. Checked on the
# validation slice: the decision layer's F0.5 is unchanged by it (see the commit message).
PREDICT_PROBA_FLOOR = 0.005
PREDICT_BATCH_ROWS = 2_000_000


def _score_country(models, path: Path) -> pd.DataFrame:
    """Score one country's test features file. Returns (s1_id, cand_id, proba)
    for pairs at or above PREDICT_PROBA_FLOOR. Fold-model ensembles are scored
    in 2M-row batches; the two-stage model needs every row of the country at
    once (its context features are per-S1 / per-candidate), so it reads the
    whole file -- fine for a slice, not for a 40M-row country."""
    cols = ["s1_id", "cand_id"] + features.FEATURE_COLUMNS
    parts = []
    if isinstance(models, dict) and models.get("kind") == "two_stage":
        chunk = pd.read_parquet(path, columns=cols)
        proba = model.predict_two_stage_final(
            models, chunk[features.FEATURE_COLUMNS].to_numpy(dtype="float32"), chunk["s1_id"].values, chunk["cand_id"].values
        )
        keep = proba >= PREDICT_PROBA_FLOOR
        return pd.DataFrame({"s1_id": chunk["s1_id"].values[keep], "cand_id": chunk["cand_id"].values[keep], "proba": proba[keep]})
    for batch in pq.ParquetFile(path).iter_batches(batch_size=PREDICT_BATCH_ROWS, columns=cols):
        chunk = batch.to_pandas()
        # to_numpy(dtype=...), not .values -- see src/model.py's build_xy docstring.
        proba = model.predict_with_fold_models(models, chunk[features.FEATURE_COLUMNS].to_numpy(dtype="float32"))
        keep = proba >= PREDICT_PROBA_FLOOR
        parts.append(pd.DataFrame({"s1_id": chunk["s1_id"].values[keep], "cand_id": chunk["cand_id"].values[keep], "proba": proba[keep]}))
        del chunk, proba
    if not parts:
        return pd.DataFrame({"s1_id": [], "cand_id": [], "proba": []})
    return pd.concat(parts, ignore_index=True)


def stage_predict() -> dict:
    print("[predict] loading trained models...", flush=True)
    _check_feature_columns()
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)

    _reset_duckdb()
    con = io_utils.get_connection()
    io_utils.register_source_view(con, "test_source1", config.TEST_SOURCE1)
    ids_by_country = {}
    for eid, country in con.execute("SELECT entity_id, country FROM test_source1").fetchall():
        ids_by_country.setdefault(country, set()).add(eid)
    con.close()
    test_ids = set().union(*ids_by_country.values()) if ids_by_country else set()

    # One country at a time: blocking is within-country, so candidates, one-to-one
    # assignment and the per-S1 decision never cross countries. Peak memory is one
    # country's (pruned) scored pairs instead of the whole test set.
    preds = {}
    policy = "expected_f_beta_subset_selection"
    for path in sorted(TEST_FEATURES_DIR.glob("features_*.parquet")):
        country = path.stem[len("features_"):]
        t0 = time.time()
        scored = _score_country(models, path)
        ids = ids_by_country.get(country, set())
        # No ground truth at real inference time -> decide() always uses the
        # expected-F0.5 subset-selection policy (see src/decide.py).
        country_preds, policy = decide.decide(scored, ids, truths=None)
        preds.update(country_preds)
        print(
            f"[predict] {country}: {len(ids)} S1, {len(scored)} pairs >= {PREDICT_PROBA_FLOOR}, "
            f"{sum(1 for v in country_preds.values() if v)} matched ({time.time()-t0:.0f}s)", flush=True,
        )
        del scored, country_preds
        gc.collect()
    for eid in test_ids:
        preds.setdefault(eid, set())  # a country with no feature file: predicted singleton
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

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = config.OUTPUT_DIR / "matching_results.tsv"
    candidate_path = config.OUTPUT_DIR / "candidate_pairs.tsv"
    write_outputs.write_matching_results(preds, required_ids, matching_path)
    n_with_candidates = write_outputs.write_candidate_pairs_from_parquet(
        (TEST_CANDIDATES_DIR / "capped_*.parquet").as_posix(), required_ids, candidate_path
    )
    print(f"[write] {n_with_candidates} of {len(required_ids)} S1 have at least one candidate", flush=True)
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
    "val_slice": stage_val_slice,
    "val_blocking": stage_val_blocking,
    "val_features": stage_val_features,
    "val_train": stage_val_train,
    "train_features": stage_train_features,
    "train": stage_train,
    "test_features": stage_test_features,
    "predict": stage_predict,
    "write": stage_write,
}
STAGE_ORDER = [
    "normalize", "val_slice", "val_blocking", "val_features", "val_train",
    "train_features", "train", "test_features", "predict", "write",
]


def _run_stage_with_logging(name: str, fn) -> None:
    """Run one stage with a timestamped start/end line and the stage's peak
    RSS in between (polled every 2s on a background thread -- there's no
    single cross-platform "peak memory" call that works identically on
    Windows and Kaggle's Linux, so this approximates it, which is enough for
    reading off a log).
    """
    start_rss = _rss_mb()
    start_note = f" (rss={start_rss:.0f}MB)" if start_rss is not None else ""
    print(f"[{_now()}] === stage: {name} start{start_note} ===", flush=True)

    peak = {"mb": start_rss or 0.0}
    stop = threading.Event()

    def _poll() -> None:
        while not stop.wait(2):
            m = _rss_mb()
            if m is not None and m > peak["mb"]:
                peak["mb"] = m

    poller = None
    if psutil is not None:
        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()

    t0 = time.time()
    try:
        fn()
    finally:
        if poller is not None:
            stop.set()
            poller.join(timeout=3)

    dt = time.time() - t0
    peak_note = f", peak_rss={peak['mb']:.0f}MB" if psutil is not None else ""
    print(f"[{_now()}] === stage: {name} done ({dt:.1f}s{peak_note}) ===", flush=True)


def main() -> None:
    stages_to_run = STAGE_ORDER if _args.stage == "all" else [_args.stage]
    if _args.stage == "all" and _args.train_source == "val":
        stages_to_run = [n for n in stages_to_run if n != "train_features"]  # unused when training on the val slice
    for name in stages_to_run:
        _run_stage_with_logging(name, STAGES[name])


if __name__ == "__main__":
    main()
