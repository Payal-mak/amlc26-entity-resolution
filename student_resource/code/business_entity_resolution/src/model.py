"""LightGBM pairwise match/no-match classifier.

GroupKFold by source1_entity_id (never split one S1's own candidate pairs
across folds -- that would leak the context features in features.py, which
are computed relative to "this S1's other candidates" and "this candidate's
other S1s"). Produces out-of-fold (OOF) probabilities for every row: each
row is scored by a model that never saw its own S1 group during training,
so OOF probabilities are a valid, leak-free stand-in for held-out
performance -- this is what evaluate.py / decide.py tune against.

LightGBM (MIT license) chosen over a neural classifier specifically because
its histogram-binned training keeps memory low even at several-million-row
scale, which matters on this 8GB dev machine (see PROJECT_LOG.md's hardware
audit) -- most of today's crashes were in blocking's raw SQL joins, not
anything model-related.

Import order matters on the dev machine: `lightgbm` must be imported before
`pandas` anywhere in the process, or LightGBM's native Dataset construction
segfaults (access violation in set_label) on any data at all. Importing it
first here only protects a caller that imports src.model before anything
else touches pandas; every real entry point (scripts/phase4_train_and_decide,
scripts/run_pipeline) also imports lightgbm as its own first line for that
reason -- see their docstrings and PROJECT_LOG.md.
"""

from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DEFAULT_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary",
    n_jobs=2,
    verbosity=-1,
)


def train_oof(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label",
    group_col: str = "s1_id",
    n_folds: int = 5,
    seed: int = 42,
    params: dict = None,
) -> tuple:
    """Train n_folds LightGBM models under GroupKFold and return OOF probabilities.

    Inputs: df with feature_cols + label_col + group_col; number of folds;
    random seed; optional param overrides (merged over DEFAULT_PARAMS).
    Output: (oof_proba: np.ndarray aligned to df's row order, models: list of
    n_folds fitted LGBMClassifier, fold_id: np.ndarray of each row's held-out
    fold index).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["random_state"] = seed

    groups = df[group_col].values
    # ascontiguousarray is required, not cosmetic: df[cols].values on a frame
    # concatenated from parquet files with mixed dtypes (float32 features,
    # int64 rank columns) can hand back a non-C-contiguous array, and
    # LightGBM's native library segfaults (access violation) reading the
    # label field when fed one -- reproduced directly on this machine.
    X = np.ascontiguousarray(df[feature_cols].values, dtype=np.float64)
    y = np.ascontiguousarray(df[label_col].values, dtype=np.float64)

    gkf = GroupKFold(n_splits=n_folds)
    oof_proba = np.zeros(len(df), dtype=np.float64)
    fold_id = np.full(len(df), -1, dtype=np.int32)
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = LGBMClassifier(**p)
        model.fit(X[train_idx], y[train_idx])
        oof_proba[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        fold_id[val_idx] = fold
        models.append(model)

    return oof_proba, models, fold_id


def predict_with_fold_models(models: list, X: np.ndarray) -> np.ndarray:
    """Average predictions across all fold models (for scoring rows that
    weren't part of GroupKFold's row set, e.g. context-entity-only pairs
    scored after the fact, or full-test inference in Phase 5).

    Inputs: list of fitted models (from train_oof), feature matrix.
    Output: mean predicted probability of the positive class across models.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=1)


def feature_importance_report(models: list, feature_cols: list) -> pd.DataFrame:
    """Average LightGBM gain-based feature importance across fold models.

    Inputs: fitted models, feature column names (same order used in training).
    Output: DataFrame sorted by importance descending.
    """
    importances = np.column_stack([m.booster_.feature_importance(importance_type="gain") for m in models])
    return (
        pd.DataFrame({"feature": feature_cols, "gain": importances.mean(axis=1)})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
