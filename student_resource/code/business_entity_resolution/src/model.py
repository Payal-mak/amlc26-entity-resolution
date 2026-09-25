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
scale -- real full-size training runs on Kaggle (31GB RAM, 4 CPUs); the local
dev machine (8GB) is dev-only smoke tests on tiny samples after repeated OOMs
even post-memory-safety-work (see PROJECT_LOG.md, 2026-09-25).

Import order matters on the local dev machine: `lightgbm` must be imported
before `pandas` anywhere in the process, or LightGBM's native Dataset
construction segfaults (access violation in set_label) on any data at all.
Importing it first here only protects a caller that imports src.model before
anything else touches pandas; every real entry point
(scripts/phase4_train_and_decide, scripts/run_pipeline) also imports
lightgbm as its own first line for that reason -- see their docstrings and
PROJECT_LOG.md. Not yet confirmed whether Kaggle's environment has the same
issue; keeping the import-order guard everywhere costs nothing if it doesn't.
"""

from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config

DEFAULT_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary",
    verbosity=-1,
    # Real values, not laptop-safe compromises -- see config.py's
    # LGBM_NUM_THREADS/LGBM_MAX_BIN/LGBM_TWO_ROUND docstring. Read at import
    # time (like every other AML_* setting), so scripts/run_pipeline.py's
    # CLI flags/env vars still control these if a local run ever needs the
    # laptop-safe values back (AML_LGBM_MAX_BIN=63 AML_LGBM_TWO_ROUND=true
    # AML_LGBM_THREADS=1).
    n_jobs=config.LGBM_NUM_THREADS,
    max_bin=config.LGBM_MAX_BIN,
    two_round=config.LGBM_TWO_ROUND,
)


def build_xy(df: pd.DataFrame, feature_cols: list, label_col: str = "label", group_col: str = "s1_id") -> tuple:
    """Extract contiguous float32 X/y/groups arrays from a features dataframe.

    Split out from train_oof on purpose: the caller should `del` (and
    gc.collect) the original dataframe right after calling this, before
    training starts -- holding both the full mixed-dtype dataframe (with its
    string id/country columns) AND X/y alive for the whole GroupKFold loop
    was real, measured peak-memory waste that contributed to an OOM at
    several-million-row scale (see PROJECT_LOG.md).

    Inputs: df with feature_cols + label_col + group_col.
    Output: (X, y, groups) -- X/y as C-contiguous float32 arrays, groups as
    whatever dtype group_col already is (strings here).
    """
    # to_numpy(dtype=...), NOT .values then cast: a mixed-dtype selection
    # (float32 features + int64 rank columns here) makes plain `.values`
    # build a float64-promoted array FIRST (pandas' block-manager
    # interleave picks the common dtype internally) and only then would a
    # separate np.ascontiguousarray(..., dtype=np.float32) downcast it --
    # by then the oversized float64 intermediate has already been
    # allocated. Reproduced directly: 25 cols x 11.56M rows needed 2.15GiB
    # as float64 and failed outright under memory pressure; to_numpy's
    # dtype argument is threaded through to the initial allocation, so it's
    # built at the target size (~1.08GiB) from the start. ascontiguousarray
    # is then a cheap no-op safety net (to_numpy already returns C-order).
    X = np.ascontiguousarray(df[feature_cols].to_numpy(dtype=np.float32))
    y = np.ascontiguousarray(df[label_col].to_numpy(dtype=np.float32))
    groups = df[group_col].values
    return X, y, groups


def train_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = 5, seed: int = 42, params: dict = None) -> tuple:
    """Train n_folds LightGBM models under GroupKFold and return OOF probabilities.

    Inputs: X/y/groups from build_xy (or any array of matching shape); number
    of folds; random seed; optional param overrides (merged over
    DEFAULT_PARAMS).
    Output: (oof_proba: np.ndarray aligned to X's row order, models: list of
    n_folds fitted LGBMClassifier, fold_id: np.ndarray of each row's held-out
    fold index).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["random_state"] = seed

    gkf = GroupKFold(n_splits=n_folds)
    oof_proba = np.zeros(len(X), dtype=np.float64)
    fold_id = np.full(len(X), -1, dtype=np.int32)
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
    X = np.ascontiguousarray(X, dtype=np.float32)
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
