"""Candidate generation / blocking.

Country-string equality is the first, mandatory partition for every block
below (verified safe in Phase 1 EDA: 0 mismatches out of 182,932 sampled
true-match pairs). Generic string equality only -- never a hardcoded
{US, India} set, since France has zero training examples.

Redesign note (Phase 3, after three separate OOM crashes -- PROJECT_LOG.md):
the original version built each block as one big unbounded join, then tried
to cap the union afterward. That's backwards -- an unbounded join can exhaust
memory/disk before a cap ever gets applied downstream. Every block here now
caps ITSELF, inside its own SQL, before its result ever reaches a union:
  - Token-based blocks (B2) exclude any token whose candidate-side document
    frequency exceeds B2_MAX_CAND_TOKEN_DF -- a token shared by tens of
    thousands of records has unbounded join fan-out regardless of whether
    it's a "suffix" by Phase 2's 2%-frequency definition.
  - Every block ends with QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id
    ORDER BY score DESC) <= PER_BLOCK_CAP, so no single block can ever
    contribute more than PER_BLOCK_CAP candidates for one S1, independent of
    how common its matching key is.
  - The old B4 (name-prefix join) is replaced by `block_b_sorted_neighborhood`:
    sort S1+candidates together by name_core within country and compare only
    within a fixed window, computed via LAG/LEAD window functions (O(n) per
    offset) rather than a range self-join (which risks the query planner
    falling back to a nested-loop join at this row count).

Blocks:
  B1: exact match on normalized name_core, within country.
  B2: shared non-suffix, non-too-common name token, within country.
  B3_postal: same postal code + >=1 shared name token, within country --
      DEMOTED to a minor/supplementary block: postal codes are present in
      only ~7.9% of US / ~0.2% of India addresses (PROJECT_LOG.md), so this
      covers almost nothing on its own.
  B_sorted_neighborhood: replaces the original prefix-based B4.
  B_geo: shared rarest in-country address token (the ~100%-coverage
      geographic signal built for scripts/build_validation_split.py) --
      PROMOTED to the primary geographic block over B3_postal.
  B5 ("transliterated field version of B1/B2 for India") is deliberately NOT
      a separate block: src.normalize.normalize_full already transliterates
      Devanagari (with schwa deletion + a learned token map -- see
      PROJECT_LOG.md) internally, so B1/B2 are already script-bridging.

The per-S1 CAP applied after unioning all (already individually-capped)
blocks is a second, smaller safety net (see `cap_candidates`), not the
primary control.
"""

import re

import duckdb
import pandas as pd

from . import normalize

POSTAL_EXPR = """
COALESCE(
    NULLIF(regexp_extract(business_address, '(\\d{6})\\D*$', 1), ''),
    NULLIF(regexp_extract(business_address, '(\\d{5})(?:-\\d{4})?\\D*$', 1), '')
)
"""

# Python-side equivalent of POSTAL_EXPR above (same two regexes -- 6-digit
# India PIN, then 5(+4) US ZIP -- kept identical on purpose so a pandas-side
# postal_code column always agrees with the SQL-side one blocking uses).
_POSTAL_RE_6 = re.compile(r"(\d{6})\D*$")
_POSTAL_RE_5 = re.compile(r"(\d{5})(?:-\d{4})?\D*$")


def extract_postal(address) -> str:
    """Extract a postal code from a raw address string, or None.

    Inputs: raw business_address (or None/NaN). Output: the matched digit
    string, or None if neither pattern matches.
    """
    address = address or ""
    m = _POSTAL_RE_6.search(address)
    if m:
        return m.group(1)
    m = _POSTAL_RE_5.search(address)
    return m.group(1) if m else None

MIN_TOKEN_LEN = 3
MIN_GEO_TOKEN_DF = 3
CANDIDATES_PER_S1_CAP = 50
PER_BLOCK_CAP = 20
# A token shared by more than this many candidate-side records (within
# country) is excluded from B2 entirely -- the Phase 3 diagnostic distribution
# had p99=114, so 150 keeps ~99% of real tokens while cutting off the
# unbounded-fan-out tail that caused an OOM (spilled to 43.7GB) the first time
# this block had no cap at all. Tried raising to 400 (hypothesis: excluding
# common-but-useful shared words like "vijay"+"ventures"/"bright"/"golden" was
# costing real recall) -- measured effect was negligible (uncapped recall
# +0.38pp, capped recall flat-to-worse: 0.7907->0.7905), so reverted to 150,
# which is also cheaper. See PROJECT_LOG.md, Phase 3.
B2_MAX_CAND_TOKEN_DF = 150
SORTED_NEIGHBOR_WINDOW = 10


def add_normalized_columns(df: pd.DataFrame, suffix_sets: dict, translit_map: dict = None) -> pd.DataFrame:
    """Add name_full / name_core / postal_code columns to a source dataframe.

    Inputs: dataframe with entity_id/business_name/business_address/country
    columns (as loaded from a source parquet/tsv), suffix_sets from
    scripts/phase2_normalization_report.py's suffix_tokens_by_country, and
    optionally translit_map from scripts/build_translit_token_map.py (pushes
    Devanagari India token-overlap from 83% to 96% on top of schwa deletion
    alone -- see PROJECT_LOG.md).
    Output: a copy of df with name_full, name_core, postal_code added.
    Runs in Python (not SQL) because normalize_full needs the real
    normalize.py logic (accent stripping, abbreviation expansion,
    Devanagari transliteration), not a SQL approximation.
    """
    df = df.copy()
    df["name_full"] = df["business_name"].map(lambda t: normalize.normalize_full(t, translit_map))
    df["name_core"] = [
        normalize.name_core(nf, suffix_sets.get(c, set()))
        for nf, c in zip(df["name_full"], df["country"])
    ]
    df["postal_code"] = df["business_address"].map(extract_postal)
    return df


def register_postal_code(con: duckdb.DuckDBPyConnection, view: str, out_view: str) -> None:
    """Add a postal_code column (regex-extracted, mostly NULL -- see module
    docstring) to `view`, registered as `out_view`.
    """
    con.execute(f"CREATE OR REPLACE TABLE {out_view} AS SELECT *, {POSTAL_EXPR} AS postal_code FROM {view}")


def register_geo_tokens(con: duckdb.DuckDBPyConnection, view: str, out_view: str) -> None:
    """Assign each row its rarest in-country address token (B_geo's key).

    Inputs: connection, source view (entity_id/business_address/country),
    output view name.
    Output: none; registers `{out_view}` with (entity_id, country, geo_token,
    geo_df) -- geo_df (the token's document frequency) is kept for scoring.
    Same method as scripts/build_validation_split.py's build_s1_geo, but
    generic over any table (there it was train_source1-only).
    """
    tok_view = f"{out_view}_tok"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {tok_view} AS
        SELECT entity_id, country, token
        FROM (
            SELECT entity_id, country,
                   UNNEST(regexp_split_to_array(
                       regexp_replace(lower(business_address), '[^a-z0-9]+', ' ', 'g'), ' '
                   )) AS token
            FROM {view}
        )
        WHERE length(token) >= {MIN_TOKEN_LEN} AND NOT regexp_matches(token, '^[0-9]+$')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out_view}_df AS
        SELECT country, token, COUNT(*) AS df FROM {tok_view} GROUP BY country, token
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out_view} AS
        SELECT t.entity_id, t.country, t.token AS geo_token, d.df AS geo_df
        FROM {tok_view} t JOIN {out_view}_df d USING (country, token)
        WHERE d.df >= {MIN_GEO_TOKEN_DF}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY d.df ASC, t.token ASC) = 1
        """
    )


def register_name_tokens(con: duckdb.DuckDBPyConnection, view: str, out_view: str) -> None:
    """Explode name_full into (entity_id, country, token), length-filtered.

    Inputs: connection, view with a name_full column, output view name.
    Output: none; registers `{out_view}` (entity_id, country, token). Used by
    B2 (join on any shared token) -- suffix filtering happens at query time
    in `block_b2_rare_token` via an anti-join against the suffix table, not
    here, so this view is reusable for other token-level analysis too.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out_view} AS
        SELECT entity_id, country, token
        FROM (
            SELECT entity_id, country, UNNEST(string_split(name_full, ' ')) AS token
            FROM {view}
        )
        WHERE length(token) >= {MIN_TOKEN_LEN}
        """
    )


def block_b1_exact_core(con: duckdb.DuckDBPyConnection, s1_view: str, cand_view: str) -> str:
    """B1: exact match on name_core, within country, capped at PER_BLOCK_CAP
    per S1 (score = constant; exact-name collisions are rare enough that the
    cap should almost never bind, but every block caps itself regardless).
    Returns the output table name.
    """
    out = "b1_pairs"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out} AS
        SELECT s1_id, cand_id FROM (
            SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY c.entity_id) AS rn
            FROM {s1_view} s JOIN {cand_view} c
              ON s.country = c.country AND s.name_core = c.name_core
            WHERE s.name_core != ''
        )
        WHERE rn <= {PER_BLOCK_CAP}
        """
    )
    return out


def block_b2_rare_token(
    con: duckdb.DuckDBPyConnection, s1_tok_view: str, cand_tok_view: str, suffix_table: str
) -> str:
    """B2: shared non-suffix, non-too-common name token, within country.
    Score = sum of 1/df over shared qualifying tokens (rarer shared tokens
    count for more); capped at PER_BLOCK_CAP per S1. Returns the output
    table name.

    Two exclusions, not one: the suffix table (>=2% country frequency, Phase
    2's name_core threshold) removes obvious legal-suffix/stop words, but a
    token just under that -- still shared by tens of thousands of records --
    has unbounded join fan-out on its own; B2_MAX_CAND_TOKEN_DF bounds that
    directly.
    """
    out = "b2_pairs"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE b2_cand_token_df AS
        SELECT country, token, COUNT(*) AS df FROM {cand_tok_view} GROUP BY country, token
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out} AS
        SELECT s1_id, cand_id FROM (
            SELECT s1_id, cand_id,
                   ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY score DESC, cand_id) AS rn
            FROM (
                SELECT s.entity_id AS s1_id, c.entity_id AS cand_id, SUM(1.0 / d.df) AS score
                FROM {s1_tok_view} s
                JOIN b2_cand_token_df d
                  ON d.country = s.country AND d.token = s.token AND d.df <= {B2_MAX_CAND_TOKEN_DF}
                JOIN {cand_tok_view} c ON s.country = c.country AND s.token = c.token
                WHERE NOT EXISTS (
                    SELECT 1 FROM {suffix_table} sf WHERE sf.country = s.country AND sf.token = s.token
                )
                GROUP BY s.entity_id, c.entity_id
            )
        )
        WHERE rn <= {PER_BLOCK_CAP}
        """
    )
    return out


def block_b3_postal(
    con: duckdb.DuckDBPyConnection, s1_view: str, cand_view: str, s1_tok_view: str, cand_tok_view: str
) -> str:
    """B3 (minor): same postal code + >=1 shared name token, within country,
    capped at PER_BLOCK_CAP per S1. Reuses the same name-token views B2
    builds. Returns the output table name.
    """
    out = "b3_pairs"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out} AS
        SELECT s1_id, cand_id FROM (
            SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY c.entity_id) AS rn
            FROM {s1_view} s
            JOIN {cand_view} c ON s.country = c.country AND s.postal_code = c.postal_code
            JOIN {s1_tok_view} st ON st.entity_id = s.entity_id
            JOIN {cand_tok_view} ct ON ct.entity_id = c.entity_id AND ct.country = st.country AND ct.token = st.token
            WHERE s.postal_code IS NOT NULL
        )
        WHERE rn <= {PER_BLOCK_CAP}
        """
    )
    return out


def block_b_sorted_neighborhood(
    con: duckdb.DuckDBPyConnection, s1_view: str, cand_view: str, window: int = SORTED_NEIGHBOR_WINDOW
) -> str:
    """Sorted-neighborhood block: sort S1+candidates together by name_core
    within country, and pair each S1 with candidates within `window`
    positions of it in that sorted order.

    Replaces the original prefix-based B4 (measured at 251M raw pairs on this
    slice -- a 4-char prefix is far too coarse a key with a 1M+ candidate
    pool). Implemented with LAG/LEAD window functions rather than a range
    self-join (`ABS(rn_a - rn_b) <= window`): a range join on ~1.4M rows risks
    the query planner falling back to a nested-loop join, whereas 2*window
    LAG/LEAD passes are each O(n) and DuckDB computes them together in one
    pass over the sorted data. Output is bounded by construction (at most
    2*window neighbors per S1), so this needs no separate DF-style cap.

    Inputs: connection, S1 view, candidate view (both need name_core,
    country, entity_id), window (positions each side).
    Output: output table name (s1_id, cand_id).
    """
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sorted_combined AS
        SELECT entity_id, country, name_core, 's1' AS typ FROM {s1_view}
        UNION ALL
        SELECT entity_id, country, name_core, 'cand' AS typ FROM {cand_view}
        """
    )
    lag_cols = ", ".join(
        f"LAG(entity_id, {k}) OVER w AS lag{k}_id, LAG(typ, {k}) OVER w AS lag{k}_typ" for k in range(1, window + 1)
    )
    lead_cols = ", ".join(
        f"LEAD(entity_id, {k}) OVER w AS lead{k}_id, LEAD(typ, {k}) OVER w AS lead{k}_typ" for k in range(1, window + 1)
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sorted_with_neighbors AS
        SELECT entity_id, country, typ, {lag_cols}, {lead_cols}
        FROM sorted_combined
        WINDOW w AS (PARTITION BY country ORDER BY name_core, entity_id)
        """
    )
    branches = [
        f"SELECT entity_id AS s1_id, lag{k}_id AS cand_id, lag{k}_typ AS cand_typ, {k} AS dist FROM sorted_with_neighbors WHERE typ = 's1'"
        for k in range(1, window + 1)
    ] + [
        f"SELECT entity_id AS s1_id, lead{k}_id AS cand_id, lead{k}_typ AS cand_typ, {k} AS dist FROM sorted_with_neighbors WHERE typ = 's1'"
        for k in range(1, window + 1)
    ]
    union_sql = " UNION ALL ".join(branches)
    out = "bneighbor_pairs"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out} AS
        SELECT s1_id, cand_id FROM (
            SELECT s1_id, cand_id,
                   ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY dist ASC, cand_id) AS rn
            FROM ({union_sql})
            WHERE cand_typ = 'cand' AND cand_id IS NOT NULL
        )
        WHERE rn <= {PER_BLOCK_CAP}
        """
    )
    return out


def block_b_geo(con: duckdb.DuckDBPyConnection, s1_geo_view: str, cand_geo_view: str) -> str:
    """B_geo: shared rarest in-country address token, capped at PER_BLOCK_CAP
    per S1 (score = 1/geo_df, so a shared rare locality token ranks above a
    shared common one). Returns the output table name.
    """
    out = "bgeo_pairs"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out} AS
        SELECT s1_id, cand_id FROM (
            SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY (1.0 / c.geo_df) DESC, c.entity_id) AS rn
            FROM {s1_geo_view} s JOIN {cand_geo_view} c
              ON s.country = c.country AND s.geo_token = c.geo_token
        )
        WHERE rn <= {PER_BLOCK_CAP}
        """
    )
    return out


def union_and_score(con: duckdb.DuckDBPyConnection, block_views: dict) -> str:
    """Union every (already per-block-capped) block's pairs, tagging which
    block(s) found each pair and a cheap score (number of distinct blocks
    that fired).

    Inputs: connection, {block_name: table_name} for every block already
    registered.
    Output: table name `all_pairs_scored` with (s1_id, cand_id, blocks,
    n_blocks). Since every input is already <= PER_BLOCK_CAP per S1, this
    union is bounded at <= PER_BLOCK_CAP * n_blocks distinct pairs per S1
    before the final cap_candidates step.
    """
    parts = [
        f"SELECT s1_id, cand_id, '{name}' AS block FROM {view}" for name, view in block_views.items()
    ]
    union_sql = " UNION ALL ".join(parts)
    con.execute(f"CREATE OR REPLACE TABLE all_pairs_tagged AS {union_sql}")
    con.execute(
        """
        CREATE OR REPLACE TABLE all_pairs_scored AS
        SELECT s1_id, cand_id,
               string_agg(DISTINCT block, ',') AS blocks,
               COUNT(DISTINCT block) AS n_blocks
        FROM all_pairs_tagged
        GROUP BY s1_id, cand_id
        """
    )
    return "all_pairs_scored"


def cap_candidates(con: duckdb.DuckDBPyConnection, scored_view: str, cap: int = CANDIDATES_PER_S1_CAP) -> str:
    """Final cap across the (already bounded) union, to CANDIDATES_PER_S1_CAP
    per S1 by score (n_blocks that fired, ties broken by cand_id).

    Inputs: connection, table from union_and_score, cap.
    Output: table name `capped_pairs` (s1_id, cand_id, blocks, n_blocks, rank).
    """
    out = "capped_pairs"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out} AS
        SELECT *, ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY n_blocks DESC, cand_id) AS rank
        FROM {scored_view}
        QUALIFY rank <= {cap}
        """
    )
    return out


BLOCK_NAMES = ["b1_exact_core", "b2_rare_token", "b3_postal_minor", "b_sorted_neighborhood", "bgeo_address_token"]


def register_suffix_table(con: duckdb.DuckDBPyConnection, suffix_sets: dict) -> None:
    """Register {country: set(tokens)} as a DuckDB table for B2's anti-join
    exclusion (see block_b2_rare_token).
    """
    rows = [(country, token) for country, toks in suffix_sets.items() for token in toks]
    df = pd.DataFrame(rows, columns=["country", "token"])
    con.register("suffix_tmp", df)
    con.execute("CREATE OR REPLACE TABLE suffix_table AS SELECT * FROM suffix_tmp")
    con.unregister("suffix_tmp")


def run_all_blocks(con: duckdb.DuckDBPyConnection, s1_country: pd.DataFrame, cand_country: pd.DataFrame) -> dict:
    """Register base views and run every block for ONE country's already-
    normalized (name_full/name_core/postal_code present) S1/candidate frames.

    Requires register_suffix_table to have already been called on `con`.
    Inputs: connection, s1/cand dataframes already filtered to one country.
    Output: {block_name: table_name}, same keys/order as BLOCK_NAMES.
    """
    con.register("s1_raw", s1_country)
    con.register("cand_raw", cand_country)
    con.execute("CREATE OR REPLACE TABLE s1_norm AS SELECT * FROM s1_raw")
    con.execute("CREATE OR REPLACE TABLE cand_norm AS SELECT * FROM cand_raw")
    con.unregister("s1_raw")
    con.unregister("cand_raw")

    register_postal_code(con, "s1_norm", "s1_pc")
    register_postal_code(con, "cand_norm", "cand_pc")
    register_geo_tokens(con, "s1_norm", "s1_geo")
    register_geo_tokens(con, "cand_norm", "cand_geo")
    register_name_tokens(con, "s1_norm", "s1_tok")
    register_name_tokens(con, "cand_norm", "cand_tok")

    b1 = block_b1_exact_core(con, "s1_pc", "cand_pc")
    b2 = block_b2_rare_token(con, "s1_tok", "cand_tok", "suffix_table")
    b3 = block_b3_postal(con, "s1_pc", "cand_pc", "s1_tok", "cand_tok")
    bneighbor = block_b_sorted_neighborhood(con, "s1_pc", "cand_pc")
    bgeo = block_b_geo(con, "s1_geo", "cand_geo")
    return dict(zip(BLOCK_NAMES, [b1, b2, b3, bneighbor, bgeo]))


def block_and_cap_country(con: duckdb.DuckDBPyConnection, s1_country: pd.DataFrame, cand_country: pd.DataFrame) -> tuple:
    """Full per-country blocking pipeline in one call: every block, union +
    score, then the final cap. Used by scripts/run_pipeline.py's real
    train/test build; scripts/phase3_blocking_report.py calls the pieces
    (run_all_blocks / union_and_score / cap_candidates) directly instead
    since it also needs the raw per-block tagging for recall reporting.

    Requires register_suffix_table to have already been called on `con`.
    Output: (capped_df, block_views) -- capped_df has columns (s1_id,
    cand_id, blocks, n_blocks, rank); block_views is {block_name: table_name}.
    """
    block_views = run_all_blocks(con, s1_country, cand_country)
    union_view = union_and_score(con, block_views)
    capped_view = cap_candidates(con, union_view)
    capped_df = con.execute(f"SELECT * FROM {capped_view}").fetchdf()
    return capped_df, block_views
