"""Exact reimplementation of the official scoring metric (macro F0.5).

This is the single source of truth for "is change X actually an improvement."
Every other stage (blocking recall-ceiling checks aside) should be judged
through this module, on out-of-fold or held-out predictions, using the exact
same formula the leaderboard uses -- not a proxy metric -- so local numbers
and leaderboard numbers move together. If they ever diverge, the validation
slice construction is the first thing to suspect (see build_validation_split.py).

Official rule (from the problem statement), per Source-1 entity:
  - empty prediction, empty truth       -> 1.0  (singleton correctly identified)
  - non-empty prediction, empty truth   -> 0.0  (false merge on a singleton)
  - empty prediction, non-empty truth   -> 0.0  (missed every true match)
  - otherwise                           -> standard F_beta(precision, recall)
The final score is the unweighted mean of the per-entity score across every
Source-1 entity in the evaluation set (macro average).
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Set

from . import config

IdSet = Set[str]
IdMap = Mapping[str, IdSet]


def f_beta_score(pred: IdSet, truth: IdSet, beta: float = config.F_BETA) -> float:
    """Per-entity F_beta score for one Source-1 entity.

    Inputs: pred (predicted S2/S3 id set), truth (ground-truth id set), beta.
    Output: float score in [0, 1], per the official edge-case rules above.
    """
    if not truth and not pred:
        return 1.0
    if not pred or not truth:
        return 0.0
    intersection = len(pred & truth)
    if intersection == 0:
        return 0.0
    precision = intersection / len(pred)
    recall = intersection / len(truth)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(
    preds: IdMap, truths: IdMap, beta: float = config.F_BETA
) -> float:
    """Macro-averaged F_beta across every entity key present in `truths`.

    Inputs: preds and truths, both {source1_entity_id: set(matched_ids)}.
    Missing keys in `preds` are treated as an empty prediction (no match).
    Output: the mean per-entity F_beta score -- this is THE official metric.
    """
    if not truths:
        return 0.0
    total = 0.0
    for s1, truth_set in truths.items():
        pred_set = preds.get(s1, set())
        total += f_beta_score(pred_set, truth_set, beta)
    return total / len(truths)


def micro_precision_recall(preds: IdMap, truths: IdMap) -> Dict[str, float]:
    """Aggregate (non-official) precision/recall, for diagnostics only.

    Pools true positives / predicted / actual counts across ALL entities
    before dividing, rather than averaging per-entity ratios. Useful to see
    "of everything we predicted, what fraction was right" at a glance; the
    leaderboard score itself always comes from macro_f_beta, never from this.
    Inputs/Output: same id-map shapes as macro_f_beta; returns
    {"precision": .., "recall": .., "tp": .., "n_pred": .., "n_truth": ..}.
    """
    tp = n_pred = n_truth = 0
    for s1, truth_set in truths.items():
        pred_set = preds.get(s1, set())
        tp += len(pred_set & truth_set)
        n_pred += len(pred_set)
        n_truth += len(truth_set)
    precision = tp / n_pred if n_pred else float("nan")
    recall = tp / n_truth if n_truth else float("nan")
    return {"precision": precision, "recall": recall, "tp": tp, "n_pred": n_pred, "n_truth": n_truth}


def singleton_accuracy(preds: IdMap, truths: IdMap) -> Optional[float]:
    """Fraction of true singletons (empty truth) correctly predicted empty.

    Inputs/Output: same shapes as macro_f_beta; returns None if there are no
    singletons in `truths` (undefined rather than misleadingly 0/0 -> 0).
    """
    singleton_keys = [s1 for s1, t in truths.items() if not t]
    if not singleton_keys:
        return None
    correct = sum(1 for s1 in singleton_keys if not preds.get(s1, set()))
    return correct / len(singleton_keys)


def per_group_macro_f_beta(
    preds: IdMap,
    truths: IdMap,
    group_of: Mapping[str, str],
    beta: float = config.F_BETA,
) -> Dict[str, Dict[str, float]]:
    """Macro F_beta broken out per group (e.g. per country, per source mix).

    Inputs: preds, truths as above; group_of maps source1_entity_id -> group
    label (e.g. the S1 record's country). Entities missing from group_of are
    reported under group "unknown".
    Output: {group_label: {"f_beta", "precision", "recall", "n"}} (precision/
    recall are the same micro/pooled diagnostic as micro_precision_recall,
    scoped to that group -- not the official metric, but what "precision/
    recall per country" in a log means in practice), one entry per group.
    """
    buckets: Dict[str, Dict[str, IdSet]] = {}
    for s1, truth_set in truths.items():
        g = group_of.get(s1, "unknown")
        buckets.setdefault(g, {})[s1] = truth_set

    report = {}
    for g, group_truths in buckets.items():
        pr = micro_precision_recall(preds, group_truths)
        report[g] = {
            "f_beta": macro_f_beta(preds, group_truths, beta),
            "precision": pr["precision"],
            "recall": pr["recall"],
            "n": len(group_truths),
        }
    return report


def load_id_list_tsv(path: Path, id_col: str, list_col: str) -> Dict[str, IdSet]:
    """Load a two-column TSV of {id_col}\\t{comma list} into an id->set map.

    Inputs: path to a .tsv with a header row; id_col/list_col column names.
    Output: {row_id: set(list_ids)}, empty set when the list cell is blank.
    Mirrors the format used by matching_results.tsv / candidate_pairs.tsv /
    train_ground_truth.tsv so it can load any of the three.
    """
    result: Dict[str, IdSet] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw = row[list_col]
            ids = set(raw.split(",")) if raw and raw.strip() else set()
            result[row[id_col]] = ids
    return result


def load_ground_truth(path: Path = config.TRAIN_GROUND_TRUTH) -> Dict[str, IdSet]:
    """Convenience wrapper: load train_ground_truth.tsv as an id->set map."""
    return load_id_list_tsv(path, config.COL_SOURCE1_ENTITY_ID, config.COL_MATCHED_ENTITY_IDS)


def score_report(
    preds: IdMap,
    truths: IdMap,
    group_of: Optional[Mapping[str, str]] = None,
    beta: float = config.F_BETA,
) -> Dict:
    """Bundle the full diagnostic report used for phase write-ups / the log.

    Inputs: preds, truths (id-set maps); optional group_of (e.g. S1 -> country)
    for a per-group breakdown; beta.
    Output: dict with overall macro F_beta (the official number), micro P/R,
    singleton accuracy, entity count, and (if group_of given) a per-group
    breakdown -- exactly the shape Phase 4 reporting asks for.
    """
    report = {
        "n_entities": len(truths),
        "macro_f_beta": macro_f_beta(preds, truths, beta),
        "singleton_accuracy": singleton_accuracy(preds, truths),
    }
    report.update({f"micro_{k}": v for k, v in micro_precision_recall(preds, truths).items()})
    if group_of is not None:
        report["by_group"] = per_group_macro_f_beta(preds, truths, group_of, beta)
    return report
