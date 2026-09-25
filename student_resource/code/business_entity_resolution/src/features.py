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

import gc
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import io_utils, normalize

_DIGIT_RUN_RE = re.compile(r"\d+")

# Rows per in-memory pandas batch while streaming base features (see
# build_base_features_streaming). Sized so a handful of python-string
# intermediate lists per batch stay well under 1GB -- this pipeline has run
# on an 8GB machine seen as low as ~250MB free with nothing but the OS +
# browser/editor running (PROJECT_LOG.md, Phase 4).
STREAMING_CHUNK_SIZE = 100_000

BLOCK_NAMES = ["b1_exact_core", "b2_rare_token", "b3_postal_minor", "b_sorted_neighborhood", "bgeo_address_token"]

FEATURE_COLUMNS = [
    "name_full_ratio", "name_full_token_set_ratio", "name_full_partial_ratio", "name_full_jaro_winkler",
    "name_core_ratio", "name_core_token_set_ratio", "name_core_partial_ratio", "name_core_jaro_winkler",
    "address_ratio", "address_token_set_ratio", "digit_jaccard",
    "postal_equal", "postal_conflict", "postal_missing",
    "n_blocks", *[f"block_{b}" for b in BLOCK_NAMES],
    "source_is_s2", "name_full_len_diff",
    # Address (features-address-name branch, commit A): house-number
    # compatibility one-hots + street similarity with numbers removed.
    "hn_equal", "hn_prefix_suffix", "hn_one_edit", "hn_one_missing", "hn_conflict", "hn_both_missing",
    "hn_small_diff", "hn_min_abs_diff_log", "hn_suffix_differs",
    "street_ratio", "street_token_set_ratio", "street_token_sort_ratio",
    # Name cleanup + containment (commit B).
    "name_clean_full_ratio", "name_clean_core_ratio", "name_nospace_ratio", "name_nospace_partial_ratio",
    "name_core_containment",
    # Rarity (commit C).
    "name_idf_jaccard", "name_idf_containment", "name_max_shared_idf", "same_addr_low_name",
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


SAME_ADDR_STREET_MIN = 90   # street_token_set_ratio at or above this counts as "same street text"
LOW_NAME_IDF_MAX = 0.2      # name_idf_jaccard below this = no meaningful rare-token overlap
LOW_NAME_NOSPACE_MAX = 60   # name_nospace_ratio below this = names look unrelated

HN_BOTH_MISSING, HN_EQUAL, HN_PREFIX_SUFFIX, HN_ONE_EDIT, HN_ONE_MISSING, HN_CONFLICT = range(6)


def _one_edit_apart(x: str, y: str) -> bool:
    """True if x and y (digit strings, len >= 3) differ by exactly one
    substitution, insertion or deletion."""
    if abs(len(x) - len(y)) > 1 or min(len(x), len(y)) < 3 or x == y:
        return False
    if len(x) == len(y):
        return sum(1 for a, b in zip(x, y) if a != b) == 1
    short, long_ = (x, y) if len(x) < len(y) else (y, x)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def house_number_relation(a: tuple, b: tuple) -> int:
    """Compatibility of two addresses' house numbers (tuples of RAW digit
    strings, leading zeros kept). Returns one of HN_* codes, checked in this
    order: both empty / one number in common once leading zeros are ignored
    ("03153" = "3153") / one number is the other with leading or trailing
    digits dropped -- a prefix or suffix, min 2 digits, checked on the raw
    strings so "302" vs "02" counts ("6104" vs "104", "16549" vs "1654") / one
    edit apart ("7800" vs "7802") / exactly one side has no number at all / no
    number in common (conflict).
    """
    if not a and not b:
        return HN_BOTH_MISSING
    if not a or not b:
        return HN_ONE_MISSING
    if {x.lstrip("0") or "0" for x in a} & {y.lstrip("0") or "0" for y in b}:
        return HN_EQUAL
    for x in a:
        for y in b:
            short, long_ = (x, y) if len(x) <= len(y) else (y, x)
            if len(short) >= 2 and (long_.startswith(short) or long_.endswith(short)):
                return HN_PREFIX_SUFFIX
    for x in a:
        for y in b:
            if _one_edit_apart(x.lstrip("0"), y.lstrip("0")):
                return HN_ONE_EDIT
    return HN_CONFLICT


HN_SMALL_DIFF_MAX = 10


def house_number_signals(a: tuple, b: tuple) -> tuple:
    """All house-number pair signals in one pass.

    Inputs: two tuples of (raw digits, suffix) from normalize.house_number_parts.
    Output: (relation code, small_diff, min_abs_diff_log10, suffix_differs):
      relation      house_number_relation on the digit strings;
      small_diff    1.0 if two DIFFERENT numbers are within HN_SMALL_DIFF_MAX of
                    each other (7800 vs 7802, 169 vs 171) -- neighbours on one
                    street, or a slightly mistyped number;
      min_abs_diff_log10  log10(1 + smallest |difference| over all number pairs),
                    NaN if either side has no number;
      suffix_differs  1.0 if the same number appears with two different
                    suffixes ("12A" vs "12B", "12 bis" vs "12 ter"); a suffix on
                    one side only is compatible.
    """
    rel = house_number_relation(tuple(d for d, _ in a), tuple(d for d, _ in b))
    if not a or not b:
        return rel, 0.0, np.nan, 0.0
    min_diff = None
    sfx_differs = 0.0
    for x, sx in a:
        vx = int(x[:12])
        for y, sy in b:
            d = abs(vx - int(y[:12]))
            if min_diff is None or d < min_diff:
                min_diff = d
            if d == 0 and sx and sy and sx != sy:
                sfx_differs = 1.0
    small = 1.0 if any(0 < abs(int(x[:12]) - int(y[:12])) <= HN_SMALL_DIFF_MAX for x, _ in a for y, _ in b) else 0.0
    return rel, small, float(np.log10(1 + min_diff)), sfx_differs


def prepare_entities(s1_df: pd.DataFrame, cand_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ENTITY derived fields for one country, computed once (not per pair
    -- pairs outnumber entities ~37:1) and indexed by entity_id.

    Inputs: that country's s1 and candidate frames (entity_id, business_name,
    business_address, postal_code, name_full, name_core, country).
    Output: DataFrame indexed by entity_id with the columns the pair-level
    features gather from.
    """
    both = pd.concat([s1_df, cand_df], ignore_index=True)
    prep = pd.DataFrame(index=pd.Index(both["entity_id"].values, name="entity_id"))
    prep["street"] = both["business_address"].map(normalize.street_key).values
    prep["hn"] = [
        normalize.house_number_parts(a, pc if isinstance(pc, str) else None)
        for a, pc in zip(both["business_address"], both["postal_code"])
    ]

    # Names with decorations stripped / digit-for-letter swaps fixed. Only
    # names that actually change are re-normalized; everything else keeps the
    # frame's existing name_full/name_core (which already include the
    # Devanagari transliteration map, unavailable here).
    raw = [n or "" for n in both["business_name"]]
    cleaned = [normalize.clean_name_for_matching(n) for n in raw]
    name_full = both["name_full"].tolist()
    name_core = both["name_core"].tolist()
    suffixes = _recover_suffix_set(name_full, name_core)
    for i, (r, c) in enumerate(zip(raw, cleaned)):
        if c != " ".join(r.split()):
            name_full[i] = normalize.normalize_full(c)
            name_core[i] = normalize.name_core(name_full[i], suffixes)
    prep["name_full_c"] = name_full
    prep["name_core_c"] = name_core
    prep["name_nospace"] = [n.replace(" ", "") for n in name_full]
    prep["core_tokens"] = [frozenset(n.split()) for n in name_core]

    # Token rarity: IDF over THIS country's S1+S2+S3 names (every entity in
    # both frames counts once; no labels). Scaled by ln(N+1) so weights sit in
    # [0, 1) whatever the corpus size -- the validation slice, the full train
    # set and the test set then produce comparable overlap ratios.
    n_entities = len(prep)
    doc_freq = Counter()
    for toks in prep["core_tokens"]:
        doc_freq.update(toks)
    scale = math.log(n_entities + 1) or 1.0
    idf = {t: math.log((n_entities + 1) / (c + 1)) / scale for t, c in doc_freq.items()}
    prep["idf_sum"] = [sum(idf[t] for t in toks) for toks in prep["core_tokens"]]
    prep.attrs["idf"] = idf
    return prep


def _recover_suffix_set(name_full: list, name_core: list) -> set:
    """The suffix/stop tokens name_core stripped, recovered from the data itself
    (every token present in a name_full but absent from its name_core). The
    per-country suffix sets aren't passed down to the feature builder, and a
    country's frame contains every token that set ever removed."""
    removed = set()
    for f, c in zip(name_full, name_core):
        if f != c:
            removed.update(set(f.split()) - set(c.split()))
    return removed


def _containment(a: frozenset, b: frozenset) -> float:
    """Share of the shorter token set found in the longer one (1.0 = one name's
    core is entirely contained in the other's); 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _pairwise(scorer, xs: list, ys: list) -> np.ndarray:
    """Element-wise rapidfuzz similarity (0-100) of xs[i] vs ys[i], vectorized in C++."""
    if not xs:
        return np.empty(0, dtype=np.float32)
    return process.cpdist(xs, ys, scorer=scorer, dtype=np.float32, workers=1)


CONTEXT_FEATURE_COLUMNS = ["rank_by_s1", "gap_to_best_by_s1", "reverse_rank"]
BASE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CONTEXT_FEATURE_COLUMNS]


def build_pair_features_base(
    pairs: pd.DataFrame, s1_df: pd.DataFrame, cand_df: pd.DataFrame, prep: pd.DataFrame = None
) -> pd.DataFrame:
    """Compute every PAIRWISE (non-context) feature for one chunk of pairs.

    Split out from the context features (rank_by_s1 / gap_to_best_by_s1 /
    reverse_rank) on purpose: those need the FULL candidate pool for a given
    s1_id/cand_id to be meaningful, so they can't be computed correctly on an
    arbitrary row-chunk (see add_context_features). This function is safe to
    call chunk-by-chunk to bound peak memory on large countries.

    Inputs: pairs (s1_id, cand_id, blocks[comma list], n_blocks -- from
    src.blocking's tagged/capped output), s1_df / cand_df (already carrying
    name_full/name_core from blocking.add_normalized_columns, plus
    business_address, entity_id, and a postal_code column).
    `prep` is prepare_entities(s1_df, cand_df); pass it when calling chunk-by-
    chunk so per-entity work is done once per country, not once per chunk
    (built here if omitted).
    Output: a new DataFrame, same row order as `pairs`, with
    BASE_FEATURE_COLUMNS plus s1_id/cand_id for joining back.
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

    if prep is None:
        prep = prepare_entities(s1_df, cand_df)
    p1 = prep.loc[pairs["s1_id"].values]
    p2 = prep.loc[pairs["cand_id"].values]

    sig = [house_number_signals(a, b) for a, b in zip(p1["hn"], p2["hn"])]
    rel = np.fromiter((s[0] for s in sig), dtype=np.int8, count=n)
    out["hn_small_diff"] = np.fromiter((s[1] for s in sig), dtype=np.float32, count=n)
    out["hn_min_abs_diff_log"] = np.fromiter((s[2] for s in sig), dtype=np.float32, count=n)
    out["hn_suffix_differs"] = np.fromiter((s[3] for s in sig), dtype=np.float32, count=n)
    out["hn_equal"] = (rel == HN_EQUAL).astype(np.float32)
    out["hn_prefix_suffix"] = (rel == HN_PREFIX_SUFFIX).astype(np.float32)
    out["hn_one_edit"] = (rel == HN_ONE_EDIT).astype(np.float32)
    out["hn_one_missing"] = (rel == HN_ONE_MISSING).astype(np.float32)
    out["hn_conflict"] = (rel == HN_CONFLICT).astype(np.float32)
    out["hn_both_missing"] = (rel == HN_BOTH_MISSING).astype(np.float32)

    st1, st2 = p1["street"].tolist(), p2["street"].tolist()
    either_empty = np.array([(not a) or (not b) for a, b in zip(st1, st2)])
    for name, scorer in (("street_ratio", fuzz.ratio), ("street_token_set_ratio", fuzz.token_set_ratio),
                         ("street_token_sort_ratio", fuzz.token_sort_ratio)):
        v = _pairwise(scorer, st1, st2)
        v[either_empty] = np.nan  # a blank/number-only address carries no street signal
        out[name] = v

    nf1, nf2 = p1["name_full_c"].tolist(), p2["name_full_c"].tolist()
    out["name_clean_full_ratio"] = _pairwise(fuzz.ratio, nf1, nf2)
    out["name_clean_core_ratio"] = _pairwise(fuzz.ratio, p1["name_core_c"].tolist(), p2["name_core_c"].tolist())
    ns1, ns2 = p1["name_nospace"].tolist(), p2["name_nospace"].tolist()
    out["name_nospace_ratio"] = _pairwise(fuzz.ratio, ns1, ns2)
    out["name_nospace_partial_ratio"] = _pairwise(fuzz.partial_ratio, ns1, ns2)
    out["name_core_containment"] = np.fromiter(
        (_containment(a, b) for a, b in zip(p1["core_tokens"], p2["core_tokens"])), dtype=np.float32, count=n
    )

    idf = prep.attrs["idf"]
    jac = np.zeros(n, dtype=np.float32)
    cont = np.zeros(n, dtype=np.float32)
    max_shared = np.zeros(n, dtype=np.float32)
    for i, (ta, tb, sa, sb) in enumerate(zip(p1["core_tokens"], p2["core_tokens"], p1["idf_sum"], p2["idf_sum"])):
        shared = ta & tb
        if not shared:
            continue
        weights = [idf[t] for t in shared]
        inter = sum(weights)
        union = sa + sb - inter
        jac[i] = inter / union if union > 0 else 0.0
        smaller = min(sa, sb)
        cont[i] = inter / smaller if smaller > 0 else 0.0
        max_shared[i] = max(weights)
    out["name_idf_jaccard"] = jac
    out["name_idf_containment"] = cont
    out["name_max_shared_idf"] = max_shared

    # Same place, different-looking name: street text matches and the house
    # numbers don't disagree, yet the names share no rare token and the
    # spaces-removed names are far apart (a renamed/garbled name at a shared
    # address -- or two unrelated businesses at one address; the model decides).
    same_addr = (out["street_token_set_ratio"] >= SAME_ADDR_STREET_MIN) & np.isin(
        rel, (HN_EQUAL, HN_PREFIX_SUFFIX, HN_ONE_EDIT)
    )
    low_name = (jac < LOW_NAME_IDF_MAX) & (out["name_nospace_ratio"] < LOW_NAME_NOSPACE_MAX)
    out["same_addr_low_name"] = (same_addr & low_name).astype(np.float32)

    result = pd.DataFrame(out)
    result["s1_id"] = pairs["s1_id"].values
    result["cand_id"] = pairs["cand_id"].values
    return result


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank_by_s1 / gap_to_best_by_s1 / reverse_rank to a FULL (unchunked)
    pairs+base-features dataframe for one country.

    Must see every pair for every s1_id and every cand_id at once -- these
    features measure a candidate's competition, so splitting an id's rows
    across calls would silently corrupt them. Callers with a large country
    should prefer running this via DuckDB window functions over the on-disk
    base-features parquet instead (see scripts/phase4_build_features.py),
    which spills to disk instead of holding it all in pandas.

    Inputs: dataframe with s1_id, cand_id, name_full_ratio columns.
    Output: same dataframe (copy) with CONTEXT_FEATURE_COLUMNS added.
    """
    df = df.copy()
    df["rank_by_s1"] = (
        df.groupby("s1_id")["name_full_ratio"].rank(method="first", ascending=False).astype(np.float32)
    )
    best_by_s1 = df.groupby("s1_id")["name_full_ratio"].transform("max")
    df["gap_to_best_by_s1"] = (best_by_s1 - df["name_full_ratio"]).astype(np.float32)
    df["reverse_rank"] = (
        df.groupby("cand_id")["name_full_ratio"].rank(method="first", ascending=False).astype(np.float32)
    )
    return df


def build_pair_features(pairs: pd.DataFrame, s1_df: pd.DataFrame, cand_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: base + context features in one call, for small/
    already-in-memory inputs (unit tests, small ad-hoc checks). Real
    country-scale runs use build_base_features_streaming +
    add_context_features_via_duckdb instead (see below), used by both the
    Phase 4 validation-slice build and scripts/run_pipeline.py's real
    train/test build.
    """
    return add_context_features(build_pair_features_base(pairs, s1_df, cand_df))


def build_base_features_streaming(
    pairs_path: Path, s1_c: pd.DataFrame, cand_c: pd.DataFrame, out_path: Path,
    label_map: dict = None, country: str = None, chunk_size: int = STREAMING_CHUNK_SIZE,
) -> tuple:
    """Stream one country's candidate pairs from parquet in `chunk_size`-row
    batches, computing base (non-context) features per batch and appending
    straight to out_path -- never materializes the whole country's pairs
    table in pandas at once (the fix for a real OOM crash on 6.7M India
    rows once free system RAM dropped under ~500MB; see PROJECT_LOG.md).

    Inputs: path to a capped-candidate-pairs parquet (s1_id, cand_id, blocks,
    n_blocks columns), that country's normalized s1/candidate frames,
    destination parquet path for the base features, an optional
    {s1_id: set(true_cand_id)} label map (omit entirely -- no "label" column
    written -- when there is no ground truth, e.g. real test-set inference),
    optional country tag to stamp on every row, batch size override.
    Output: (n_rows, n_positives_or_None) written.
    """
    pf = pq.ParquetFile(pairs_path)
    prep = prepare_entities(s1_c, cand_c)
    writer = None
    n_rows = 0
    n_pos = 0 if label_map is not None else None
    try:
        for batch in pf.iter_batches(batch_size=chunk_size, columns=["s1_id", "cand_id", "blocks", "n_blocks"]):
            chunk = batch.to_pandas()
            feat = build_pair_features_base(chunk, s1_c, cand_c, prep)
            if label_map is not None:
                feat["label"] = [
                    1 if cid in label_map.get(sid, ()) else 0 for sid, cid in zip(feat["s1_id"], feat["cand_id"])
                ]
                n_pos += int(feat["label"].sum())
            if country is not None:
                feat["country"] = country

            table = pa.Table.from_pandas(feat, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            n_rows += len(feat)
            del chunk, feat, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    return n_rows, n_pos


def add_context_features_via_duckdb(base_path: Path, out_path: Path) -> None:
    """Second pass: rank_by_s1 / gap_to_best_by_s1 / reverse_rank as DuckDB
    window functions over the on-disk base-features file for one country.

    DuckDB spills past its memory_limit to DUCKDB_TMP_DIR rather than holding
    the whole (potentially several-million-row) file in RAM, unlike the
    pandas groupby add_context_features uses. Reads/writes parquet directly;
    the Python process only holds the query plan, not the data.
    """
    con = io_utils.get_connection()
    try:
        con.execute(
            f"""
            COPY (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY name_full_ratio DESC) AS rank_by_s1,
                    (MAX(name_full_ratio) OVER (PARTITION BY s1_id) - name_full_ratio) AS gap_to_best_by_s1,
                    ROW_NUMBER() OVER (PARTITION BY cand_id ORDER BY name_full_ratio DESC) AS reverse_rank
                FROM read_parquet('{base_path.as_posix()}')
            ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def build_country_features(
    pairs_path: Path, s1_c: pd.DataFrame, cand_c: pd.DataFrame, out_path: Path,
    label_map: dict = None, country: str = None, chunk_size: int = STREAMING_CHUNK_SIZE,
) -> tuple:
    """One country's full feature build: stream base features to a temp file,
    then add context features via DuckDB, writing the final `out_path`.

    This is the memory-safe entry point real callers should use (see
    scripts/phase4_build_features.py and scripts/run_pipeline.py) instead of
    calling the two steps above directly.

    Output: (n_rows, n_positives_or_None), same as build_base_features_streaming.
    """
    base_path = out_path.with_name(f"_basefeat_{out_path.name}")
    n_rows, n_pos = build_base_features_streaming(pairs_path, s1_c, cand_c, base_path, label_map, country, chunk_size)
    add_context_features_via_duckdb(base_path, out_path)
    base_path.unlink(missing_ok=True)
    return n_rows, n_pos
