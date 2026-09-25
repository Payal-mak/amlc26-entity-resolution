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
    return pd.concat(frames, ignore_index=True)


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
    print(f"loaded features: {len(df)} rows, {df['label'].sum()} positives ({time.time()-t0:.1f}s)")

    eval_ids, truths, s1_country = load_eval_ids_and_truth_and_country()

    oof_proba, models, fold_id = model.train_oof(df, features.FEATURE_COLUMNS, n_folds=5, seed=config.RANDOM_SEED)
    df["proba"] = oof_proba
    print(f"trained 5-fold GroupKFold LightGBM ({time.time()-t0:.1f}s)")

    importance = model.feature_importance_report(models, features.FEATURE_COLUMNS)

    preds, policy = decide.decide(df, eval_ids, truths=truths)
    print(f"decision policy selected: {policy} ({time.time()-t0:.1f}s)")

    report = evaluate.score_report(preds, truths, group_of=s1_country)
    report["policy"] = policy
    report["n_train_rows"] = len(df)
    report["n_positives"] = int(df["label"].sum())
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
