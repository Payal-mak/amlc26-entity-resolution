"""Phase 4: train LightGBM under GroupKFold on the Phase 3 candidate pairs,
run the decision layer (one-to-one + best-of{global threshold, expected-F0.5
subset selection}), and report val F0.5 / precision / recall / singleton
accuracy, overall and per country.

Usage (from code/business_entity_resolution/):
    python -m scripts.phase4_train_and_decide
"""

# MUST be the first import in this process, before pandas/duckdb/pyarrow --
# on this dev machine, importing lightgbm AFTER pandas has already loaded
# its native extensions causes a reproducible access violation deep inside
# LightGBM's C API (crashes on literally any data, in set_label, regardless
# of row count/dtype/contiguity -- isolated by bisecting import order, see
# PROJECT_LOG.md). Importing lightgbm first sidesteps whatever DLL/runtime
# it conflicts with.
import lightgbm  # noqa: F401,E402

import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, decide, evaluate, features, model  # noqa: E402

VAL_SLICE_DIR = config.PARQUET_DIR / "val_slice"
PHASE4_DIR = config.PARQUET_DIR / "phase4"


def load_features() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(PHASE4_DIR.glob("features_*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return df


def load_eval_ids_and_truth_and_country() -> tuple:
    s1 = pd.read_parquet(VAL_SLICE_DIR / "source1.parquet", columns=["entity_id", "is_eval", "country"])
    eval_ids = set(s1.loc[s1["is_eval"], "entity_id"])
    s1_country = dict(zip(s1["entity_id"], s1["country"]))
    gt = pd.read_parquet(VAL_SLICE_DIR / "ground_truth_in_slice.parquet")
    truths = {}
    for s1id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if s1id in eval_ids:
            truths[s1id] = set(ids.split(",")) if isinstance(ids, str) and ids.strip() else set()
    return eval_ids, truths, s1_country


def run() -> dict:
    t0 = time.time()
    df = load_features()
    n_rows, n_positives = len(df), int(df["label"].sum())
    print(f"loaded features: {n_rows} rows, {n_positives} positives ({time.time()-t0:.1f}s)")

    eval_ids, truths, s1_country = load_eval_ids_and_truth_and_country()

    X, y, groups = model.build_xy(df, features.FEATURE_COLUMNS)
    s1_ids, cand_ids = df["s1_id"].values, df["cand_id"].values
    # Free the full mixed-dtype features dataframe (feature columns + string
    # ids) BEFORE training starts, not after -- holding both it and X/y alive
    # for the whole GroupKFold loop was real, measured peak-memory waste that
    # contributed to a `bad allocation` crash inside LightGBM's own native
    # Dataset construction at ~11.6M rows (see PROJECT_LOG.md).
    del df
    gc.collect()

    oof_proba, models, fold_id = model.train_oof(X, y, groups, n_folds=5, seed=config.RANDOM_SEED)
    print(f"trained 5-fold GroupKFold LightGBM ({time.time()-t0:.1f}s)")
    importance = model.feature_importance_report(models, features.FEATURE_COLUMNS)
    del X, y, groups
    gc.collect()

    df = pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids, "proba": oof_proba})
    preds, policy = decide.decide(df, eval_ids, truths=truths)
    print(f"decision policy selected: {policy} ({time.time()-t0:.1f}s)")

    report = evaluate.score_report(preds, truths, group_of=s1_country)
    report["policy"] = policy
    report["n_train_rows"] = n_rows
    report["n_positives"] = n_positives
    report["build_time_seconds"] = round(time.time() - t0, 1)
    report["top_features"] = importance.head(15).values.tolist()

    with open(PHASE4_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    write_gt_style(preds, PHASE4_DIR / "val_predictions.parquet")
    return report


def write_gt_style(preds: dict, path: Path) -> None:
    d = pd.DataFrame(
        {
            "source1_entity_id": list(preds.keys()),
            "matched_entity_ids": [",".join(sorted(v)) for v in preds.values()],
        }
    )
    d.to_parquet(path, index=False)


if __name__ == "__main__":
    report = run()
    print(json.dumps({k: v for k, v in report.items() if k != "by_group"}, indent=2, default=str, ensure_ascii=False))
    print(json.dumps({"by_group": report.get("by_group")}, indent=2, default=str, ensure_ascii=False))
