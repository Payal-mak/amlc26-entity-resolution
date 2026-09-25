"""Decision layer: probabilities -> final match sets per S1.

Order matters and is fixed: one-to-one assignment ALWAYS runs first, then a
per-entity policy (global threshold, or expected-F0.5 subset selection)
decides how many of the SURVIVING candidates each S1 keeps. Running it in
the other order would let an S1 "keep" a candidate via a low per-entity
threshold that then gets stolen by a higher-probability competitor anyway,
so pruning losers first is strictly more informative for the per-entity step.

One-to-one is a hard constraint, not a heuristic: PROJECT_LOG.md's Phase 1
EDA found 0 violations of "each S2/S3 record belongs to at most one S1"
across ~3.7M/3.9M checked ground-truth references. Greedy conflict
resolution (each candidate id goes to whichever S1 scored it highest) is
used instead of an exact assignment algorithm (Hungarian) because this runs
at millions-of-pairs scale, where an exact algorithm is not viable.

Both per-entity policies are inference-time-only functions of the model's
own probabilities (no ground truth needed), so the SAME code path used to
tune/validate here is what Phase 5 will call for real. `evaluate.py`'s exact
metric is used only to CHOOSE between them on the labeled validation slice,
never inside the policies themselves.
"""

import numpy as np
import pandas as pd

from . import config, evaluate


def one_to_one(df: pd.DataFrame, score_col: str = "proba", cand_col: str = "cand_id") -> pd.DataFrame:
    """Keep, for each candidate id, only the row with the highest score.

    Inputs: a pairs dataframe with cand_col and score_col (and whatever else
    the caller needs preserved -- s1_id, etc.).
    Output: a filtered copy, one row per candidate id (its best-scoring S1).
    """
    idx = df.groupby(cand_col)[score_col].idxmax()
    return df.loc[idx].reset_index(drop=True)


def apply_global_threshold(df: pd.DataFrame, threshold: float, s1_col: str = "s1_id", score_col: str = "proba") -> dict:
    """Predict every candidate at or above `threshold`, per S1.

    Inputs: post-one-to-one pairs dataframe, a probability threshold.
    Output: {s1_id: set(cand_id)} -- entities with no surviving candidate
    above threshold get an empty set (a predicted singleton).
    """
    kept = df[df[score_col] >= threshold]
    return {s1id: set(g["cand_id"]) for s1id, g in kept.groupby(s1_col)}


def tune_global_threshold(
    df: pd.DataFrame, truths: dict, thresholds=None, s1_col: str = "s1_id", score_col: str = "proba"
) -> tuple:
    """Grid-search a single global probability threshold against the exact
    macro F0.5 metric.

    Inputs: post-one-to-one pairs dataframe restricted to scored S1 entities,
    {s1_id: set(true_cand_id)} to score against, candidate threshold grid.
    Output: (best_threshold, best_macro_f_beta, all_s1_ids_considered) --
    all_s1_ids_considered lets the caller add back S1s with zero candidates
    (who are absent from `df` entirely) as empty predictions.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        preds = apply_global_threshold(df, t, s1_col, score_col)
        f = evaluate.macro_f_beta(preds, truths)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def expected_f_beta_subset_selection(
    df: pd.DataFrame, s1_col: str = "s1_id", cand_col: str = "cand_id", score_col: str = "proba",
    beta: float = config.F_BETA,
) -> dict:
    """Per-entity adaptive cutoff: for each S1, sort its surviving candidates
    by probability, and pick the count k that maximizes an EXPECTED F_beta
    computed from the probabilities themselves (no ground truth used -- this
    is the function Phase 5 test-time inference calls too, not just
    validation tuning).

    Approximation (documented, not the exact Ye et al. optimal-F-measure DP,
    given the time-box): treats each candidate's true-match status as an
    independent Bernoulli(p_i). Expected precision@k = mean(top-k probs);
    expected recall@k = sum(top-k probs) / T, where T = sum of ALL this S1's
    candidate probabilities (an estimate of its expected true-match count).
    k=0 (predict empty) uses P(no true match among any candidate) =
    prod(1 - p_i), the independence-assumption analogue of the official
    "empty pred, empty truth -> 1.0" rule.

    Inputs: post-one-to-one pairs dataframe.
    Output: {s1_id: set(cand_id)}.
    """
    beta2 = beta * beta
    result = {}
    for s1id, group in df.groupby(s1_col):
        g = group.sort_values(score_col, ascending=False)
        probs = g[score_col].to_numpy(dtype=np.float64)
        cids = g[cand_col].tolist()
        n = len(probs)

        best_k, best_score = 0, float(np.prod(1 - probs))
        cumsum = np.cumsum(probs)
        total = cumsum[-1] if n else 0.0
        for k in range(1, n + 1):
            precision = cumsum[k - 1] / k
            recall = cumsum[k - 1] / total if total > 0 else 0.0
            denom = beta2 * precision + recall
            score = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
            if score > best_score:
                best_k, best_score = k, score

        result[s1id] = set(cids[:best_k])
    return result


def decide(
    df: pd.DataFrame,
    eval_ids: set,
    truths: dict = None,
    s1_col: str = "s1_id",
    cand_col: str = "cand_id",
    score_col: str = "proba",
) -> dict:
    """Full decision pipeline: one-to-one, then whichever per-entity policy
    wins on `truths` (if given) -- else defaults to expected-F0.5 subset
    selection (the policy usable with no ground truth, for real inference).

    Inputs: full scored pairs dataframe (ALL S1 -- eval and context both, so
    one-to-one sees true competition), eval_ids (S1s to actually return
    predictions for), truths (optional, {s1_id: set(true_cand_id)} for
    policy comparison -- validation only), column names.
    Output: {s1_id: set(cand_id)} for every id in eval_ids (empty set if it
    has no surviving candidates at all).
    """
    resolved = one_to_one(df, score_col, cand_col)

    subset_preds = expected_f_beta_subset_selection(resolved, s1_col, cand_col, score_col)
    policy = "expected_f_beta_subset_selection"

    if truths is not None:
        eval_subset_preds = {k: v for k, v in subset_preds.items() if k in eval_ids}
        f_subset = evaluate.macro_f_beta(eval_subset_preds, truths)

        eval_only = resolved[resolved[s1_col].isin(eval_ids)]
        best_t, f_threshold = tune_global_threshold(eval_only, truths, s1_col=s1_col, score_col=score_col)
        if f_threshold > f_subset:
            subset_preds = apply_global_threshold(resolved, best_t, s1_col, score_col)
            policy = f"global_threshold(t={best_t:.2f})"

    return {eid: subset_preds.get(eid, set()) for eid in eval_ids}, policy
