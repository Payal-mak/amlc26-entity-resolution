"""Build a geographic validation slice out of train/, with realistic decoy density.

Why not just random-sample S1 rows and pull their true matches? Because that
throws away distractors: a locally-sampled "validation set" of only true pairs
plus nothing else would let a model look far more precise than it really is,
since real inference always has to pick the right candidate out of a crowd of
similar nearby businesses. Instead we pick whole neighborhoods and take every
S1/S2/S3 record that falls in one, matched or not, so the decoy density in the
slice matches the full data.

v1 used a regex-extracted postal code as the neighborhood key: rejected after
measuring only ~7.9% US / ~0.2% India address coverage, which silently dropped
~30% of true matches. v2 switched to each S1's rarest in-country address token
(data-driven, no gazetteer): ~100% coverage, but its EVAL-only pool (geo
neighbors of the sampled S1s, plus their true-match partners) had an inflated
decoy rate (~81-84% vs a genuine ~26%). That inflation is a real distortion,
not just "conservative": most of those extra "decoys" are S2/S3 records whose
true S1 partner exists in the full 2.2M-row train_source1 but simply wasn't
one of the ~45k S1 rows sampled into the slice. On the real test set that
partner *is* present, so one-to-one assignment and reverse-rank features would
correctly route the record to it -- in the old slice they couldn't, which
would have (a) trained context features on a distribution that won't exist at
test time, (b) left one-to-one assignment undertested, and (c) tuned the
decision threshold against ~4x the real decoy density, costing recall on test.

v3 (this version) fixes that with an EVAL/CONTEXT split instead of a single
flat S1 set:
  - EVAL entities: the ~target_total S1s actually sampled (geo-bucketed as in
    v2). These are the only ones evaluate.py scores (see `is_eval` below).
  - CONTEXT entities: every S1 (from the full 2.2M) whose true match already
    landed in the EVAL pool. They're added -- with their real name/address/
    country row and their (pool-intersected) ground truth -- so the pipeline
    can route contested candidates to their real owner during one-to-one
    assignment and reverse-rank feature computation, exactly as it would on
    test, instead of that record dangling as an unownable phantom decoy.

    First attempt also pulled each new context entity's OWN geo-token
    neighborhood into the pool (reasoning: they should "participate in
    blocking" like a real query entity, which for EVAL meant contributing
    decoys too). That crashed the dev machine: round 1 alone found 219,416
    context entities against the ~45k EVAL sample, and pulling all of their
    neighborhoods -- several thousand of which have a rarest-token document
    frequency in the tens or hundreds of thousands (max observed: 273,616) --
    pushed available system RAM down to ~300MB before it was killed. Measured
    before the fix, not assumed: see PROJECT_LOG.md.

    Fixed by *not* re-pulling context entities' own neighborhoods at all: a
    context entity's true-match record is, by definition, already in the
    pool (that's the only reason it was found), and for Phase 3 blocking to
    later route that record to its rightful context owner instead of a
    competing EVAL entity, blocking just needs the context entity's real
    address text available (it is -- context rows are written to
    source1.parquet like any other) so a real blocking pass can independently
    rediscover the link. No pre-computed decoy neighborhood is needed for
    that. With the pool held fixed, closure completes in exactly one round by
    construction: a second owners_of() pass over the same pool cannot find
    any owner the first pass didn't already find. That second pass is still
    run and reported, as a cheap, honest confirmation of closure rather than
    an assumption of it.
  - Only EVAL entities are scored; CONTEXT entities exist purely so EVAL's
    candidates have real competition. `source1.parquet` carries an `is_eval`
    column so evaluate.py (or any Phase 4 code) can filter to the scored set.

Country allocation is proportional to each country's real share of
train_source1 (data-driven, not hardcoded to {US, India}).

The France-vs-{US,India} generalization check ("train on US only, validate on
India") needs no separate artifact: it's just this same slice's ground truth,
filtered by the S1 record's own country column and by is_eval, at
model-evaluation time.

Usage (from code/business_entity_resolution/):
    python -m scripts.build_validation_split --target-total 45000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, io_utils  # noqa: E402

MIN_TOKEN_LEN = 3
MIN_TOKEN_DF = 3  # drop rarer-than-this tokens: more likely a typo than a place name
MAX_BUCKET_N = 2000  # cap a single neighborhood's S1 count so a few huge cities
# (Brooklyn, Houston, Delhi-area localities -- all real, all measured with df in the
# hundreds of thousands for the most extreme cases) can't single-handedly satisfy the
# whole target and produce a slice dominated by a handful of mega-city decoy pools.
CLOSURE_ITERATIONS = 2  # find context entities, expand pool, repeat once more, then stop


def _tokenize_view(con: duckdb.DuckDBPyConnection, src_view: str, out_view: str) -> None:
    """Create a view of (entity_id, country, token) from a source view's address.

    Inputs: connection, source view name (train/test source1/2/3), output view name.
    Output: none; registers out_view. Tokens are lowercased, non-alnum-split,
    length >= MIN_TOKEN_LEN, and purely-numeric tokens (house numbers, PINs)
    are dropped since those aren't place names.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {out_view} AS
        SELECT entity_id, country, token
        FROM (
            SELECT entity_id, country,
                   UNNEST(regexp_split_to_array(
                       regexp_replace(lower(business_address), '[^a-z0-9]+', ' ', 'g'), ' '
                   )) AS token
            FROM {src_view}
        )
        WHERE length(token) >= {MIN_TOKEN_LEN}
          AND NOT regexp_matches(token, '^[0-9]+$')
        """
    )


def build_s1_geo(con: duckdb.DuckDBPyConnection) -> None:
    """Assign each train_source1 row its rarest-in-country address token.

    Inputs: connection (train_source1 view already registered).
    Output: none; registers view `s1_geo(entity_id, country, geo_token, df)`,
    covering ALL of train_source1 (not just the EVAL sample) so context
    entities' geo tokens are available with no extra computation later.
    """
    _tokenize_view(con, "train_source1", "train_source1_tok")
    con.execute(
        """
        CREATE OR REPLACE TABLE s1_token_df AS
        SELECT country, token, COUNT(*) AS df
        FROM train_source1_tok
        GROUP BY country, token
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW s1_geo AS
        SELECT t.entity_id, t.country, t.token AS geo_token, d.df
        FROM train_source1_tok t
        JOIN s1_token_df d USING (country, token)
        WHERE d.df >= {MIN_TOKEN_DF}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY d.df ASC, t.token ASC) = 1
        """
    )


def select_eval_buckets(con: duckdb.DuckDBPyConnection, target_total: int) -> list:
    """Randomly pick many small/medium (country, geo_token) buckets from
    s1_geo to hit ~target_total S1 rows, proportioned by each country's real
    share of train_source1 (data-driven, not a hardcoded country list).

    Buckets are capped at MAX_BUCKET_N and shuffled (fixed seed) before greedy
    accumulation -- picking largest-first was tried and rejected: it hit
    target with only 51 buckets, all mega-city tokens, which made the pool far
    larger than needed. Many small neighborhoods instead keeps pool size
    reasonable and gives better geographic diversity.

    Inputs: connection (with s1_geo registered), target row count.
    Output: list of (country, geo_token) tuples selected.
    """
    country_shares = con.execute(
        "SELECT country, COUNT(*) AS n FROM train_source1 GROUP BY country"
    ).fetchdf()
    country_shares["target"] = (
        country_shares["n"] / country_shares["n"].sum() * target_total
    ).round().astype(int)

    bucket_counts = con.execute(
        f"""
        SELECT country, geo_token, COUNT(*) AS n
        FROM s1_geo
        GROUP BY country, geo_token
        HAVING COUNT(*) <= {MAX_BUCKET_N}
        """
    ).fetchdf()

    selected = []
    for _, row in country_shares.iterrows():
        country, target = row["country"], row["target"]
        sub = bucket_counts[bucket_counts["country"] == country].sample(
            frac=1.0, random_state=config.RANDOM_SEED
        )
        cum = 0
        for _, b in sub.iterrows():
            if cum >= target:
                break
            selected.append((b["country"], b["geo_token"]))
            cum += b["n"]
    return selected


def geo_pool_ids(con: duckdb.DuckDBPyConnection, s1_ids: set) -> tuple:
    """S2/S3 ids sharing a geo token with any of the given S1 entities.

    Inputs: connection (s1_geo, train_source2_tok, train_source3_tok
    registered), a set of source1_entity_id strings.
    Output: (s2_ids, s3_ids) sets. Used both for the initial EVAL pool and,
    per round, for each new batch of context entities' own neighborhoods.
    """
    if not s1_ids:
        return set(), set()
    con.register("tmp_geo_s1", pd.DataFrame({"entity_id": list(s1_ids)}))
    tokens = con.execute(
        "SELECT DISTINCT g.country, g.geo_token FROM s1_geo g JOIN tmp_geo_s1 t USING (entity_id)"
    ).fetchdf()
    con.unregister("tmp_geo_s1")
    if tokens.empty:
        return set(), set()
    con.register("tmp_geo_tokens", tokens)
    s2 = set(
        con.execute(
            "SELECT DISTINCT tk.entity_id FROM train_source2_tok tk "
            "JOIN tmp_geo_tokens b ON tk.country = b.country AND tk.token = b.geo_token"
        ).fetchdf()["entity_id"]
    )
    s3 = set(
        con.execute(
            "SELECT DISTINCT tk.entity_id FROM train_source3_tok tk "
            "JOIN tmp_geo_tokens b ON tk.country = b.country AND tk.token = b.geo_token"
        ).fetchdf()["entity_id"]
    )
    con.unregister("tmp_geo_tokens")
    return s2, s3


def register_gt_exploded(con: duckdb.DuckDBPyConnection) -> None:
    """One row per (source1_entity_id, matched_id) from the full ground truth.

    Inputs: connection (train_ground_truth view already registered).
    Output: none; registers view `gt_exploded`, used as a reverse index
    (matched_id -> owning S1) for finding context entities.
    """
    con.execute(
        """
        CREATE OR REPLACE VIEW gt_exploded AS
        SELECT source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) AS matched_id
        FROM train_ground_truth
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
        """
    )


def owners_of(con: duckdb.DuckDBPyConnection, pool_ids: set, exclude: set) -> set:
    """S1 ids (anywhere in the full 2.2M train_source1) owning any of pool_ids.

    Inputs: connection (gt_exploded registered), the S2/S3 ids to look up,
    and a set of S1 ids to exclude from the result (already in EVAL/CONTEXT).
    Output: set of new owner S1 ids not already in `exclude`.
    """
    if not pool_ids:
        return set()
    con.register("tmp_owner_pool", pd.DataFrame({"matched_id": list(pool_ids)}))
    owners = set(
        con.execute(
            "SELECT DISTINCT g.source1_entity_id FROM gt_exploded g "
            "JOIN tmp_owner_pool p USING (matched_id)"
        ).fetchdf()["source1_entity_id"]
    )
    con.unregister("tmp_owner_pool")
    return owners - exclude


def load_full_ground_truth(con: duckdb.DuckDBPyConnection) -> dict:
    """Load the entire train ground truth once as {s1_id: set(matched_ids)}.

    Inputs: connection (train_ground_truth view already registered).
    Output: full mapping, ~2.2M entries. Loaded once and sliced in Python
    afterward (dict lookups) instead of re-querying DuckDB per S1 subset.
    """
    df = con.execute("SELECT source1_entity_id, matched_entity_ids FROM train_ground_truth").fetchdf()
    result = {}
    for s1, ids in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        result[s1] = set(ids.split(",")) if isinstance(ids, str) and ids.strip() else set()
    return result


def write_gt_parquet(gt: dict, path: Path) -> None:
    """Write a {s1: set(ids)} mapping out as a two-column parquet file.

    Inputs: gt mapping, destination path.
    Output: none; writes source1_entity_id / matched_entity_ids (comma-joined).
    """
    df = pd.DataFrame(
        {
            "source1_entity_id": list(gt.keys()),
            "matched_entity_ids": [",".join(sorted(v)) for v in gt.values()],
        }
    )
    df.to_parquet(path, index=False)


def build(target_total: int) -> dict:
    """Build the EVAL/CONTEXT validation slice end to end; write parquet + report.

    Inputs: target_total EVAL S1 row count.
    Output: the stats report dict (also written to disk as JSON and printed).
    """
    t_start = time.time()
    out_dir = config.PARQUET_DIR / "val_slice"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = io_utils.get_connection()
    io_utils.register_all_standard_views(con)

    build_s1_geo(con)
    coverage = con.execute(
        """
        SELECT s.country, COUNT(*) AS n_total, COUNT(g.entity_id) AS n_covered
        FROM train_source1 s LEFT JOIN s1_geo g USING (entity_id)
        GROUP BY s.country
        """
    ).fetchdf()

    buckets = select_eval_buckets(con, target_total)
    con.execute("CREATE OR REPLACE TABLE selected_buckets (country VARCHAR, geo_token VARCHAR)")
    con.executemany("INSERT INTO selected_buckets VALUES (?, ?)", buckets)
    eval_ids = set(
        con.execute(
            "SELECT g.entity_id FROM s1_geo g JOIN selected_buckets b USING (country, geo_token)"
        ).fetchdf()["entity_id"]
    )

    _tokenize_view(con, "train_source2", "train_source2_tok")
    _tokenize_view(con, "train_source3", "train_source3_tok")
    register_gt_exploded(con)

    full_gt_all = load_full_ground_truth(con)

    def true_match_ids(ids_subset: set, prefix: str) -> set:
        return {i for s1 in ids_subset for i in full_gt_all.get(s1, set()) if i.startswith(prefix)}

    s2_geo_eval, s3_geo_eval = geo_pool_ids(con, eval_ids)
    pool_s2 = s2_geo_eval | true_match_ids(eval_ids, config.SOURCE2_PREFIX)
    pool_s3 = s3_geo_eval | true_match_ids(eval_ids, config.SOURCE3_PREFIX)

    s1_set = set(eval_ids)
    context_ids = set()
    all_gt = {s1: full_gt_all.get(s1, set()) for s1 in eval_ids}

    # Pool is held FIXED here (see module docstring: expanding each context
    # entity's own geo-neighborhood is what crashed the first attempt). With
    # a fixed pool, one owners_of() pass finds every context entity there is
    # to find; a second pass is run purely to confirm that -- it is
    # guaranteed empty by construction, not a further expansion step.
    closure_log = []
    current_pool = pool_s2 | pool_s3
    for it in range(1, CLOSURE_ITERATIONS + 1):
        new_owners = owners_of(con, current_pool, exclude=s1_set)
        closure_log.append(
            {"iteration": it, "pool_size": len(current_pool), "new_context_entities": len(new_owners)}
        )
        if not new_owners:
            break
        context_ids |= new_owners
        s1_set |= new_owners
        for s1 in new_owners:
            all_gt[s1] = full_gt_all.get(s1, set())
        # deliberately NOT expanding pool_s2/pool_s3 via new_owners' own geo
        # tokens -- see module docstring.

    residual_orphans = len(owners_of(con, current_pool, exclude=s1_set))

    s2_ids, s3_ids = pool_s2, pool_s3
    in_slice_ids = s2_ids | s3_ids

    con.register(
        "s1_ids_tbl",
        pd.DataFrame({"entity_id": list(s1_set), "is_eval": [i in eval_ids for i in s1_set]}),
    )
    con.register("s2_ids_tbl", pd.DataFrame({"entity_id": list(s2_ids)}))
    con.register("s3_ids_tbl", pd.DataFrame({"entity_id": list(s3_ids)}))

    io_utils.export_parquet(
        con,
        "SELECT s.*, t.is_eval FROM train_source1 s JOIN s1_ids_tbl t USING (entity_id)",
        out_dir / "source1.parquet",
    )
    io_utils.export_parquet(
        con, "SELECT s.* FROM train_source2 s JOIN s2_ids_tbl USING (entity_id)", out_dir / "source2.parquet"
    )
    io_utils.export_parquet(
        con, "SELECT s.* FROM train_source3 s JOIN s3_ids_tbl USING (entity_id)", out_dir / "source3.parquet"
    )

    s1_country = dict(
        zip(
            *con.execute("SELECT entity_id, country FROM train_source1 s JOIN s1_ids_tbl t USING (entity_id)")
            .fetchdf()
            .values.T
        )
    )

    # Effective decoy rate: a pool record counts as "owned" only if its true
    # owner is inside s1_set (EVAL + CONTEXT); everything else -- genuine
    # decoys plus any still-unresolved orphans -- is a decoy as the pipeline
    # will actually see it.
    con.register("tmp_final_pool", pd.DataFrame({"matched_id": list(in_slice_ids)}))
    owner_rows = con.execute(
        "SELECT p.matched_id, g.source1_entity_id FROM tmp_final_pool p "
        "LEFT JOIN gt_exploded g USING (matched_id)"
    ).fetchdf()
    con.unregister("tmp_final_pool")
    owner_map = dict(zip(owner_rows["matched_id"], owner_rows["source1_entity_id"]))
    owned_by_s1_set = sum(1 for owner in owner_map.values() if owner in s1_set)
    con.close()

    gt_in_slice = {s1: (ids & in_slice_ids) for s1, ids in all_gt.items()}
    matches_dropped = sum(len(all_gt[s1]) - len(gt_in_slice[s1]) for s1 in all_gt)
    # Broken out separately because a combined number is misleading: EVAL's
    # matches are protected by the true-match safety net (should be exactly
    # 0). CONTEXT entities are only pulled in for the ONE match that landed
    # in the pool -- their OTHER, unrelated matches elsewhere in the full
    # 2.2M-row dataset are expected to show up as "dropped" here and are
    # harmless, since context entities are never scored and Phase 3 blocking
    # was never going to need to recover those unrelated matches anyway.
    eval_matches_dropped = sum(len(all_gt[s1]) - len(gt_in_slice[s1]) for s1 in eval_ids)
    context_matches_dropped = matches_dropped - eval_matches_dropped

    write_gt_parquet(gt_in_slice, out_dir / "ground_truth_in_slice.parquet")
    write_gt_parquet(all_gt, out_dir / "ground_truth_full_for_slice.parquet")

    eval_gt = {s1: gt_in_slice[s1] for s1 in eval_ids}
    n_singleton_eval = sum(1 for ids in eval_gt.values() if not ids)

    eval_country_counts, context_country_counts = {}, {}
    for eid, c in s1_country.items():
        bucket = eval_country_counts if eid in eval_ids else context_country_counts
        bucket[c] = bucket.get(c, 0) + 1

    naive_decoy_rate = (
        len(in_slice_ids - {i for ids in eval_gt.values() for i in ids}) / len(in_slice_ids)
        if in_slice_ids
        else float("nan")
    )
    effective_decoy_rate = 1 - owned_by_s1_set / len(in_slice_ids) if in_slice_ids else float("nan")

    report = {
        "target_total": target_total,
        "build_time_seconds": round(time.time() - t_start, 1),
        "s1_geo_token_coverage_by_country": coverage.assign(
            pct=lambda d: d["n_covered"] / d["n_total"]
        ).to_dict(orient="records"),
        "n_eval": len(eval_ids),
        "n_context": len(context_ids),
        "n_s1_total": len(s1_set),
        "eval_country_counts": eval_country_counts,
        "context_country_counts": context_country_counts,
        "closure_log": closure_log,
        "residual_orphans_after_closure": residual_orphans,
        "n_s2": len(s2_ids),
        "n_s3": len(s3_ids),
        "pool_size_total": len(in_slice_ids),
        "singleton_rate_eval": n_singleton_eval / len(eval_gt) if eval_gt else None,
        "full_train_singleton_rate_baseline": 0.0558,
        "matches_dropped_by_slice_boundary": matches_dropped,
        "eval_matches_dropped_by_slice_boundary": eval_matches_dropped,
        "context_matches_dropped_by_slice_boundary": context_matches_dropped,
        "matches_total": sum(len(v) for v in all_gt.values()),
        "naive_decoy_rate_pool": naive_decoy_rate,
        "effective_decoy_rate_pool": effective_decoy_rate,
        "full_train_decoy_rate_baseline_s2": 0.266,
        "full_train_decoy_rate_baseline_s3": 0.254,
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=45000)
    args = parser.parse_args()

    report = build(args.target_total)
    print(json.dumps(report, indent=2, default=str))
