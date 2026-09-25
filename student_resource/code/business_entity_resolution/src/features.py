"""Pairwise feature engineering for candidate (S1, S2/S3) pairs.

v1 scope cut, deliberate: the plan (PROJECT_LOG.md) listed TF-IDF char-3gram
cosine on name as a feature. Skipped here to keep Phase 4 fast given the 9 PM
time-box -- rapidfuzz's ratio/token_set_ratio/partial_ratio/Jaro-Winkler
already give strong, overlapping string-similarity signal, and computing a
proper per-pair TF-IDF cosine (not a full cross-product matrix, which would
be tens of GB at this row count) needs chunked sparse-matrix gathering that
isn't worth the implementation/compute risk today. Straightforward to add in
a v2 pass.

Country is never a feature (confirmed absent below) -- it would not
generalize to France, which has zero training rows.

All string-similarity functions come from rapidfuzz (MIT license, C++
backed) -- fast enough to run row-wise in Python at millions-of-pairs scale
without a vectorization trick.
"""

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from . import normalize

_DIGIT_RUN_RE = re.compile(r"\d+")

BLOCK_NAMES = ["b1_exact_core", "b2_rare_token", "b3_postal_minor", "b_sorted_neighborhood", "bgeo_address_token"]

FEATURE_COLUMNS = [
    "name_full_ratio", "name_full_token_set_ratio", "name_full_partial_ratio", "name_full_jaro_winkler",
    "name_core_ratio", "name_core_token_set_ratio", "name_core_partial_ratio", "name_core_jaro_winkler",
    "address_ratio", "address_token_set_ratio", "digit_jaccard",
    "postal_equal", "postal_conflict", "postal_missing",
    "n_blocks", *[f"block_{b}" for b in BLOCK_NAMES],
    "source_is_s2", "name_full_len_diff",
    "rank_by_s1", "gap_to_best_by_s1", "reverse_rank",
]


def _digit_tokens(address: str) -> set:
    """Extract the set of digit runs (house numbers, PINs, unit numbers, ...)
    from an address string. Input: raw address (or None). Output: set of
    digit-string tokens, e.g. {"123", "560001"}.
    """
    return set(_DIGIT_RUN_RE.findall(address or ""))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets; 1.0 if both empty (vacuously equal
    "no digits" case), 0.0 if only one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_pair_features(pairs: pd.DataFrame, s1_df: pd.DataFrame, cand_df: pd.DataFrame) -> pd.DataFrame:
    """Compute every pairwise + context feature for one country's pairs.

    Inputs: pairs (s1_id, cand_id, blocks[comma list], n_blocks -- from
    src.blocking's tagged/capped output), s1_df / cand_df (already carrying
    name_full/name_core from blocking.add_normalized_columns, plus
    business_address, entity_id, and a postal_code column).
    Output: a new DataFrame, same row order as `pairs`, with FEATURE_COLUMNS
    plus the original s1_id/cand_id for joining back.
    """
    s1_idx = s1_df.set_index("entity_id")
    cand_idx = cand_df.set_index("entity_id")

    s1_rows = s1_idx.loc[pairs["s1_id"]].reset_index(drop=True)
    cand_rows = cand_idx.loc[pairs["cand_id"]].reset_index(drop=True)

    n = len(pairs)
    out = {}

    name_full_ratio = np.empty(n, dtype=np.float32)
    name_full_tsr = np.empty(n, dtype=np.float32)
    name_full_pr = np.empty(n, dtype=np.float32)
    name_full_jw = np.empty(n, dtype=np.float32)
    name_core_ratio = np.empty(n, dtype=np.float32)
    name_core_tsr = np.empty(n, dtype=np.float32)
    name_core_pr = np.empty(n, dtype=np.float32)
    name_core_jw = np.empty(n, dtype=np.float32)
    addr_ratio = np.empty(n, dtype=np.float32)
    addr_tsr = np.empty(n, dtype=np.float32)
    digit_jac = np.empty(n, dtype=np.float32)
    len_diff = np.empty(n, dtype=np.float32)

    s1_addr_clean = s1_rows["business_address"].map(normalize.basic_clean).tolist()
    cand_addr_clean = cand_rows["business_address"].map(normalize.basic_clean).tolist()
    s1_digits = s1_rows["business_address"].map(_digit_tokens).tolist()
    cand_digits = cand_rows["business_address"].map(_digit_tokens).tolist()

    nf1, nf2 = s1_rows["name_full"].tolist(), cand_rows["name_full"].tolist()
    nc1, nc2 = s1_rows["name_core"].tolist(), cand_rows["name_core"].tolist()

    for i, (a, b, ca, cb, aa, ab, da, db) in enumerate(
        zip(nf1, nf2, nc1, nc2, s1_addr_clean, cand_addr_clean, s1_digits, cand_digits)
    ):
        name_full_ratio[i] = fuzz.ratio(a, b)
        name_full_tsr[i] = fuzz.token_set_ratio(a, b)
        name_full_pr[i] = fuzz.partial_ratio(a, b)
        name_full_jw[i] = JaroWinkler.normalized_similarity(a, b) * 100

        name_core_ratio[i] = fuzz.ratio(ca, cb)
        name_core_tsr[i] = fuzz.token_set_ratio(ca, cb)
        name_core_pr[i] = fuzz.partial_ratio(ca, cb)
        name_core_jw[i] = JaroWinkler.normalized_similarity(ca, cb) * 100

        addr_ratio[i] = fuzz.ratio(aa, ab)
        addr_tsr[i] = fuzz.token_set_ratio(aa, ab)

        digit_jac[i] = _jaccard(da, db)
        len_diff[i] = abs(len(a) - len(b))

    out["name_full_ratio"] = name_full_ratio
    out["name_full_token_set_ratio"] = name_full_tsr
    out["name_full_partial_ratio"] = name_full_pr
    out["name_full_jaro_winkler"] = name_full_jw
    out["name_core_ratio"] = name_core_ratio
    out["name_core_token_set_ratio"] = name_core_tsr
    out["name_core_partial_ratio"] = name_core_pr
    out["name_core_jaro_winkler"] = name_core_jw
    out["address_ratio"] = addr_ratio
    out["address_token_set_ratio"] = addr_tsr
    out["digit_jaccard"] = digit_jac
    out["name_full_len_diff"] = len_diff

    s1_postal = s1_rows["postal_code"].values
    cand_postal = cand_rows["postal_code"].values
    both_present = pd.notna(s1_postal) & pd.notna(cand_postal)
    out["postal_equal"] = (both_present & (s1_postal == cand_postal)).astype(np.float32)
    out["postal_conflict"] = (both_present & (s1_postal != cand_postal)).astype(np.float32)
    out["postal_missing"] = (~both_present).astype(np.float32)

    out["n_blocks"] = pairs["n_blocks"].astype(np.float32).values
    blocks_str = pairs["blocks"].fillna("")
    for b in BLOCK_NAMES:
        out[f"block_{b}"] = blocks_str.str.contains(b, regex=False).astype(np.float32).values

    out["source_is_s2"] = pairs["cand_id"].str.startswith("S2-").astype(np.float32).values

    result = pd.DataFrame(out)
    result["s1_id"] = pairs["s1_id"].values
    result["cand_id"] = pairs["cand_id"].values

    result["rank_by_s1"] = (
        result.groupby("s1_id")["name_full_ratio"].rank(method="first", ascending=False).astype(np.float32)
    )
    best_by_s1 = result.groupby("s1_id")["name_full_ratio"].transform("max")
    result["gap_to_best_by_s1"] = (best_by_s1 - result["name_full_ratio"]).astype(np.float32)
    result["reverse_rank"] = (
        result.groupby("cand_id")["name_full_ratio"].rank(method="first", ascending=False).astype(np.float32)
    )

    return result
