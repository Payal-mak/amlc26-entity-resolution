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


# ---------------------------------------------------------------------------
# Two-stage model
# ---------------------------------------------------------------------------
# Stage 1 is the ordinary model. Stage 2 sees the same features PLUS context
# rebuilt from stage-1 probabilities: how a pair's probability ranks among its
# S1's other candidates, among the other S1s competing for the same candidate,
# and how many of its S1's candidates look like matches. The existing
# rank_by_s1 / gap_to_best_by_s1 / reverse_rank features are computed from a
# crude name-string ratio; these use the model's own opinion instead.
#
# No leakage: every stage-2 input for a row comes from stage-1 OUT-OF-FOLD
# probabilities (a model that never saw that row's S1 group), and both stages
# use the same deterministic GroupKFold-by-S1 splits. (Second-order effect,
# standard in stacking: a training row's stage-1 OOF probability came from a
# model that did see the stage-2 validation fold. Checked against a strictly
# nested run (stage 1 refit inside every outer fold, exactly the test-time
# recipe): 0.9399 nested vs 0.9371 here on the 2000-entity val slice, so the
# shortcut does not inflate the score.)
STAGE2_EXTRA_COLUMNS = ["s2_rank_by_s1", "s2_gap_to_best", "s2_reverse_rank", "s2_n_above_05"]
STAGE2_ABOVE_THRESHOLD = 0.5


def stage2_context_features(s1_ids, cand_ids, p1: np.ndarray, threshold: float = STAGE2_ABOVE_THRESHOLD) -> np.ndarray:
    """(n, 4) float32 matrix, columns = STAGE2_EXTRA_COLUMNS, from stage-1
    probabilities `p1` (one per pair; must be OOF for training rows). The raw
    probability is deliberately NOT a column: tried, stage 2 then took 90% of
    its gain from it and became a copy of stage 1.

    s2_rank_by_s1: 1 = this S1's best candidate.
    s2_gap_to_best: best probability among this S1's candidates minus this one.
    s2_reverse_rank: 1 = the highest-probability S1 among those that list this
    candidate. s2_n_above_05: how many of this S1's candidates have p1 > 0.5.
    Ranks break ties by row order, so results are deterministic.
    """
    p1 = np.asarray(p1, dtype=np.float32)
    df = pd.DataFrame({"s": pd.factorize(np.asarray(s1_ids))[0], "c": pd.factorize(np.asarray(cand_ids))[0], "p": p1})
    by_s = df.groupby("s")["p"]
    rank = by_s.rank(method="first", ascending=False)
    gap = by_s.transform("max") - df["p"]
    rrank = df.groupby("c")["p"].rank(method="first", ascending=False)
    n_above = (df["p"] > threshold).astype(np.float32).groupby(df["s"]).transform("sum")
    return np.ascontiguousarray(np.column_stack([rank, gap, rrank, n_above]), dtype=np.float32)


def _with_stage2(X: np.ndarray, s1_ids, cand_ids, p1: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.hstack([X, stage2_context_features(s1_ids, cand_ids, p1)]), dtype=np.float32)


def train_two_stage_oof(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, s1_ids, cand_ids, feature_cols: list,
    n_folds: int = 5, seed: int = 42, params: dict = None,
) -> tuple:
    """Out-of-fold probabilities from the two-stage model.

    Inputs: X/y/groups from build_xy, the pair ids (aligned to X), feature
    column names, folds/seed/params as in train_oof.
    Output: (oof_stage2 probabilities, info dict with stage1_oof, models1,
    models2, and importance = stage-2 gain report over feature_cols +
    STAGE2_EXTRA_COLUMNS).
    """
    p1, models1, _ = train_oof(X, y, groups, n_folds=n_folds, seed=seed, params=params)
    X2 = _with_stage2(X, s1_ids, cand_ids, p1)
    p2, models2, _ = train_oof(X2, y, groups, n_folds=n_folds, seed=seed, params=params)
    importance = feature_importance_report(models2, list(feature_cols) + STAGE2_EXTRA_COLUMNS)
    return p2, {"stage1_oof": p1, "models1": models1, "models2": models2, "importance": importance}


def fit_two_stage_final(X, y, s1_ids, cand_ids, stage1_oof: np.ndarray, seed: int = 42, params: dict = None) -> dict:
    """The models used for test inference: stage 1 fit on ALL train rows, and
    stage 2 fit on all train rows with features built from the stage-1 OOF
    probabilities (never from the in-sample fit above).

    Output: {"kind": "two_stage", "stage1": model, "stage2": model}.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["random_state"] = seed
    X = np.ascontiguousarray(X, dtype=np.float32)
    stage1 = LGBMClassifier(**p).fit(X, y)
    stage2 = LGBMClassifier(**p).fit(_with_stage2(X, s1_ids, cand_ids, stage1_oof), y)
    return {"kind": "two_stage", "stage1": stage1, "stage2": stage2}


def predict_two_stage_final(final: dict, X: np.ndarray, s1_ids, cand_ids) -> np.ndarray:
    """Score unseen rows (e.g. the test set): stage-1 probabilities from the
    full-train stage-1 model, stage-2 features rebuilt from those, then the
    stage-2 model. Needs ALL of the rows' candidates at once (context features
    are per-S1 / per-candidate), so pass the whole test feature table."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    p1 = final["stage1"].predict_proba(X)[:, 1]
    return final["stage2"].predict_proba(_with_stage2(X, s1_ids, cand_ids, p1))[:, 1]
