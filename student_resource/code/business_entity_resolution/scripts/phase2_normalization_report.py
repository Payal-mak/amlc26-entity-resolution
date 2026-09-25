"""Phase 2 measurements: data-driven suffix tokens, Devanagari script-mismatch
rate among India true pairs, transliteration's effect on token overlap, and
15 before/after normalization examples across US/India/France.

Everything that touches a full source file goes through DuckDB (country
buckets first) rather than pandas, per the standing memory-safety rule. Only
the (much smaller) subset of India true-match pairs with a confirmed script
mismatch is pulled into Python, since `indic_transliteration` is a per-string
Python call with no SQL equivalent.

Usage (from code/business_entity_resolution/):
    python -m scripts.phase2_normalization_report
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, io_utils, normalize  # noqa: E402

SUFFIX_MIN_DF_FRACTION = 0.02  # a token counts as "suffix/stop" if it's in >=2% of a country's names
DEVANAGARI_RANGE = "[ऀ-ॿ]"

_ABBR_CASE_SQL = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in normalize.ABBREVIATIONS.items())


def _name_token_expr(view: str) -> str:
    """SQL for (entity_id, country, token) from a view's business_name.

    Approximates src.normalize.normalize_full in SQL (lowercase, '&'->'and',
    punctuation removal, whole-token abbreviation expansion) for full-corpus
    frequency statistics -- close enough for identifying suffix/stop tokens;
    the authoritative per-record normalization used for actual features is
    always the Python normalize.py functions, not this approximation.
    """
    return f"""
        SELECT entity_id, country,
               CASE token {_ABBR_CASE_SQL} ELSE token END AS token
        FROM (
            SELECT entity_id, country,
                   UNNEST(regexp_split_to_array(
                       regexp_replace(
                           regexp_replace(lower(business_name), '&', ' and ', 'g'),
                           '[^a-z0-9]+', ' ', 'g'
                       ), ' '
                   )) AS token
            FROM {view}
        )
        WHERE length(token) >= 1
    """


def build_name_token_df(con: duckdb.DuckDBPyConnection) -> None:
    """Per-(country, token) document frequency over ALL train+test names.

    Inputs: connection with all 6 standard source views registered.
    Output: none; registers table `name_token_df(country, token, df)` and
    `country_name_counts(country, n_names)`, combining train+test (allowed:
    unsupervised frequency counting over provided data, not an external
    lookup), so the suffix list is defined for France too even though France
    has zero training rows.
    """
    views = ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]
    union_sql = " UNION ALL ".join(_name_token_expr(v) for v in views)
    con.execute(f"CREATE OR REPLACE TABLE all_name_tokens AS {union_sql}")
    con.execute(
        """
        CREATE OR REPLACE TABLE name_token_df AS
        SELECT country, token, COUNT(*) AS df
        FROM all_name_tokens
        GROUP BY country, token
        """
    )
    views_union_names = " UNION ALL ".join(f"SELECT entity_id, country FROM {v}" for v in views)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE country_name_counts AS
        SELECT country, COUNT(*) AS n_names FROM ({views_union_names}) GROUP BY country
        """
    )


def suffix_tokens_by_country(con: duckdb.DuckDBPyConnection, min_fraction: float = SUFFIX_MIN_DF_FRACTION) -> dict:
    """Learn the suffix/stop-token set per country from measured frequency.

    Inputs: connection (name_token_df / country_name_counts registered),
    minimum document-frequency fraction to count as "suffix/stop".
    Output: {country: set(tokens)}. Purely data-driven -- no hardcoded list,
    so it's defined the same way for France as for US/India.
    """
    df = con.execute(
        """
        SELECT d.country, d.token, d.df, c.n_names, d.df * 1.0 / c.n_names AS frac
        FROM name_token_df d JOIN country_name_counts c USING (country)
        WHERE d.df * 1.0 / c.n_names >= ?
        ORDER BY country, frac DESC
        """,
        [min_fraction],
    ).fetchdf()
    result = {}
    for country, group in df.groupby("country"):
        result[country] = set(group["token"])
    return result, df


def measure_devanagari_mismatch(con: duckdb.DuckDBPyConnection) -> tuple:
    """Fraction of India true pairs with a Latin<->Devanagari name-script mismatch.

    Inputs: connection (train views + train_ground_truth registered).
    Output: (summary dict, DataFrame of the mismatched pairs: s1_id, s1_name,
    matched_id, matched_name) for reuse by the transliteration measurement.
    """
    con.execute(
        """
        CREATE OR REPLACE VIEW s23_names AS
        SELECT entity_id, business_name, country FROM train_source2
        UNION ALL
        SELECT entity_id, business_name, country FROM train_source3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW gt_exploded_pairs AS
        SELECT source1_entity_id AS s1_id, UNNEST(string_split(matched_entity_ids, ',')) AS matched_id
        FROM train_ground_truth WHERE matched_entity_ids != ''
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW india_pairs AS
        SELECT p.s1_id, s1.business_name AS s1_name, p.matched_id, m2.business_name AS matched_name
        FROM gt_exploded_pairs p
        JOIN train_source1 s1 ON s1.entity_id = p.s1_id
        JOIN s23_names m2 ON m2.entity_id = p.matched_id
        WHERE s1.country = 'India'
        """
    )
    counts = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_pairs,
            SUM(CASE WHEN regexp_matches(s1_name, '{DEVANAGARI_RANGE}')
                          != regexp_matches(matched_name, '{DEVANAGARI_RANGE}')
                     THEN 1 ELSE 0 END) AS n_mismatch
        FROM india_pairs
        """
    ).fetchdf()
    n_pairs, n_mismatch = int(counts["n_pairs"][0]), int(counts["n_mismatch"][0])

    mismatched = con.execute(
        f"""
        SELECT s1_id, s1_name, matched_id, matched_name
        FROM india_pairs
        WHERE regexp_matches(s1_name, '{DEVANAGARI_RANGE}')
              != regexp_matches(matched_name, '{DEVANAGARI_RANGE}')
        """
    ).fetchdf()
    summary = {"n_india_true_pairs": n_pairs, "n_script_mismatch": n_mismatch, "pct_script_mismatch": n_mismatch / n_pairs if n_pairs else None}

    MAX_FOR_TRANSLIT_CHECK = 50_000
    sampled = False
    if len(mismatched) > MAX_FOR_TRANSLIT_CHECK:
        mismatched = mismatched.sample(n=MAX_FOR_TRANSLIT_CHECK, random_state=config.RANDOM_SEED)
        sampled = True
    summary["transliteration_check_sampled"] = sampled
    summary["transliteration_check_n"] = len(mismatched)
    return summary, mismatched


def measure_transliteration_gain(mismatched: pd.DataFrame) -> dict:
    """For mismatched pairs, does transliterating the Devanagari side create
    token overlap that wasn't there before?

    Inputs: DataFrame from measure_devanagari_mismatch (s1_id, s1_name,
    matched_id, matched_name).
    Output: summary dict with before/after shared-token counts.
    """
    # "Before" bypasses normalize.normalize_full's transliteration step on
    # purpose (that step exists precisely BECAUSE of what this measurement
    # found -- see PROJECT_LOG.md) to isolate what transliteration itself is
    # buying us; it is not what production code should call.
    def _pre_translit_normalize(text: str) -> str:
        return normalize.expand_abbreviations(normalize.basic_clean(text))

    n_before_overlap = 0
    n_after_overlap = 0
    examples = []
    for _, row in mismatched.iterrows():
        s1_tokens = set(_pre_translit_normalize(row["s1_name"]).split())
        m_tokens = set(_pre_translit_normalize(row["matched_name"]).split())
        before_overlap = bool(s1_tokens & m_tokens)
        if before_overlap:
            n_before_overlap += 1

        s1_translit = normalize.transliterate_devanagari(row["s1_name"]) if normalize.has_devanagari(row["s1_name"]) else row["s1_name"]
        m_translit = normalize.transliterate_devanagari(row["matched_name"]) if normalize.has_devanagari(row["matched_name"]) else row["matched_name"]
        s1_tokens2 = set(_pre_translit_normalize(s1_translit).split())
        m_tokens2 = set(_pre_translit_normalize(m_translit).split())
        after_overlap = bool(s1_tokens2 & m_tokens2)
        if after_overlap:
            n_after_overlap += 1
        if (not before_overlap) and after_overlap and len(examples) < 8:
            examples.append(
                {
                    "s1_name": row["s1_name"], "matched_name": row["matched_name"],
                    "s1_translit": s1_translit, "matched_translit": m_translit,
                }
            )

    n = len(mismatched)
    return {
        "n_mismatched_pairs_checked": n,
        "n_with_token_overlap_before_translit": n_before_overlap,
        "n_with_token_overlap_after_translit": n_after_overlap,
        "newly_bridged_pairs": n_after_overlap - n_before_overlap,
        "examples_newly_bridged": examples,
    }


def build_examples(con: duckdb.DuckDBPyConnection, suffix_sets: dict) -> list:
    """15 before/after normalize_full/name_core examples: US, India (incl. a
    Devanagari case), and test-set France rows.

    Inputs: connection, suffix_sets from suffix_tokens_by_country.
    Output: list of dicts with country/raw/normalized_full/name_core.
    """
    us_rows = con.execute(
        "SELECT business_name, country FROM train_source1 WHERE country='US' USING SAMPLE 5"
    ).fetchdf()
    india_rows = con.execute(
        "SELECT business_name, country FROM train_source1 WHERE country='India' USING SAMPLE 4"
    ).fetchdf()
    devanagari_row = con.execute(
        f"SELECT business_name, country FROM train_source2 WHERE country='India' AND regexp_matches(business_name, '{DEVANAGARI_RANGE}') LIMIT 1"
    ).fetchdf()
    france_rows = con.execute(
        "SELECT business_name, country FROM test_source1 WHERE country='France' USING SAMPLE 5"
    ).fetchdf()

    all_rows = pd.concat([us_rows, india_rows, devanagari_row, france_rows], ignore_index=True)
    examples = []
    for _, row in all_rows.iterrows():
        raw = row["business_name"]
        country = row["country"]
        full = normalize.normalize_full(raw)
        core = normalize.name_core(full, suffix_sets.get(country, set()))
        entry = {"country": country, "raw": raw, "normalized_full": full, "name_core": core}
        if normalize.has_devanagari(raw):
            entry["transliterated"] = normalize.transliterate_devanagari(raw)
            entry["normalized_full_after_translit"] = normalize.normalize_full(entry["transliterated"])
        examples.append(entry)
    return examples


def run() -> dict:
    t0 = time.time()
    con = io_utils.get_connection()
    io_utils.register_all_standard_views(con)

    build_name_token_df(con)
    suffix_sets, suffix_df = suffix_tokens_by_country(con)

    mismatch_summary, mismatched = measure_devanagari_mismatch(con)
    translit_summary = measure_transliteration_gain(mismatched)

    examples = build_examples(con, suffix_sets)
    con.close()

    out_dir = config.PARQUET_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_df.to_parquet(out_dir / "suffix_token_candidates.parquet", index=False)

    report = {
        "build_time_seconds": round(time.time() - t0, 1),
        "suffix_min_df_fraction": SUFFIX_MIN_DF_FRACTION,
        "suffix_tokens_top20_by_country": {
            c: suffix_df[suffix_df["country"] == c].head(20)[["token", "frac"]].values.tolist()
            for c in suffix_df["country"].unique()
        },
        "devanagari_mismatch": mismatch_summary,
        "transliteration_gain": translit_summary,
        "examples": examples,
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
