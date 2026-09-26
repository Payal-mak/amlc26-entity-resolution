"""Rules-only baseline: exact / near-exact keys, no ML model. A fast safety-net
submission that is completely separate from the ML pipeline (its own work
sub-directory and its own output folder; nothing the ML pipeline reads or writes
is touched).

Pipeline
  1. Entity keys (per record, streamed in chunks so RAM stays small), all from the
     EXISTING normalize.py / blocking.py helpers:
       name_full  = normalize_full(clean_name_for_matching(name), translit_map)
       name_core  = normalize.name_core(name_full, data-learned suffix set of the country)
       street     = normalize.street_key(address)         (numbers removed, abbreviations expanded)
       postal     = blocking.extract_postal(address)
       hn         = raw house-number digit runs (normalize.house_number_parts)
  2. Candidates: three cheap exact joins, always within the same country (a generic
     string equality, never a country list):
       K1 same name_core
       K2 same postal code + same rarest name token
       K3 same street + same first name token
       K4 same street + same house number        K5 same street + same rarest name token
       K6 same name tokens in any order          (K4-K6 added because K1-K3 alone reach only ~53% recall)
     A key shared by more than KEY_CAP candidate records is skipped (it cannot identify
     anything) and each key keeps at most BLOCK_CAP candidates per S1. This union IS
     candidate_pairs.tsv.
  3. Rules (all thresholds tuned on the validation slice with evaluate.py):
       R1  name_core equal AND (postal equal OR (street equal AND house numbers compatible))
       R2  token_set_ratio(cleaned names) >= X AND street equal AND house numbers compatible
       R3  name_core equal AND no conflicting postal AND the S1 name has a rare token (IDF >= T)
  4. One-to-one: every S2/S3 id goes to its best-scoring S1 only.

Usage (from code/business_entity_resolution/), work dir must hold the Phase-2 files
(run `run_pipeline --stage normalize` once) and, for --split val, the validation slice:
    python -m scripts.rules_baseline --split val
    python -m scripts.rules_baseline --split test --duckdb-memory 2GB
"""

import argparse
import json
import os
import sys
import threading
import time
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

from src import blocking, config, evaluate, features, io_utils, normalize, write_outputs  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

KEY_CAP = 100        # a candidate-side key shared by more records than this identifies nothing
BLOCK_CAP = 20       # candidates kept per S1 per key type
MIN_STREET_LEN = 6
CHUNK_ROWS = 200_000
PAIR_BATCH_ROWS = 500_000

HN_OK_MODES = {
    "strict": {features.HN_EQUAL, features.HN_PREFIX_SUFFIX},
    "near": {features.HN_EQUAL, features.HN_PREFIX_SUFFIX, features.HN_ONE_EDIT},
    "lenient": {features.HN_EQUAL, features.HN_PREFIX_SUFFIX, features.HN_ONE_EDIT,
                features.HN_ONE_MISSING, features.HN_BOTH_MISSING},
}
RULE_SETS = [("R1",), ("R2",), ("R3",), ("R1", "R2"), ("R1", "R3"), ("R2", "R3"), ("R1", "R2", "R3")]
DEFAULT_CFG = {"rules": ["R1", "R2", "R3"], "x": 90, "hn_mode": "near", "idf_min": 0.5, "street_min": 100, "r3_street": 0, "sim": "name_sim"}


# ----------------------------------------------------------------------------- resources / helpers
def load_resources() -> tuple:
    p2 = config.PARQUET_DIR / "phase2"
    suffix_df = pd.read_parquet(p2 / "suffix_token_candidates.parquet")
    suffix_sets = {c: set(g["token"]) for c, g in suffix_df.groupby("country")}
    tm = p2 / "translit_token_map.parquet"
    translit = dict(zip(*(pd.read_parquet(tm)[c] for c in ("translit_token", "latin_token")))) if tm.exists() else {}
    return suffix_sets, translit


def entity_keys(df: pd.DataFrame, suffix_sets: dict, translit_map: dict) -> pd.DataFrame:
    """Per-record keys for one chunk (columns entity_id, business_name, business_address, country)."""
    names = df["business_name"].fillna("").tolist()
    addrs = df["business_address"].fillna("").tolist()
    countries = df["country"].tolist()
    full = [normalize.normalize_full(normalize.clean_name_for_matching(n), translit_map) for n in names]
    core = [normalize.name_core(f, suffix_sets.get(c, set())) for f, c in zip(full, countries)]
    postal = [blocking.extract_postal(a) for a in addrs]
    hn = [",".join(d for d, _ in normalize.house_number_parts(a, p)) for a, p in zip(addrs, postal)]
    return pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "src": [e[1] for e in df["entity_id"]],       # "S1-..." -> "1"
        "country": countries,
        "name_core": core,
        "name_full": full,
        "first_tok": [c.split(" ")[0] if c else "" for c in core],
        "name_sorted": [" ".join(sorted(c.split(" "))) if c else "" for c in core],
        "nospace": [c.replace(" ", "") for c in core],
        "hn1": [(h.split(",")[0].lstrip("0") or "0") if h else "" for h in hn],
        "street": [normalize.street_key(a) for a in addrs],
        "postal": postal,
        "hn": hn,
    })


def write_entities(batches, out_path: Path, suffix_sets: dict, translit_map: dict) -> int:
    """Stream (pandas) record chunks through entity_keys into one parquet file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for df in batches:
        keys = entity_keys(df, suffix_sets, translit_map)
        table = pa.Table.from_pandas(keys, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        n += len(keys)
    if writer is not None:
        writer.close()
    return n


def _new_connection(memory: str, threads: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory}'")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET temp_directory = '{config.DUCKDB_TMP_DIR.as_posix()}'")
    con.execute("SET max_temp_directory_size = '200GB'")
    con.execute("SET preserve_insertion_order = false")
    return con


# ----------------------------------------------------------------------------- candidates (DuckDB)
def build_candidates(con, ent_path: Path, cand_path: Path) -> dict:
    """Materialize the entity table with IDF-based rare-token info, run the three exact joins,
    write the deduplicated (s1_id, cand_id, k1, k2, k3) union to cand_path. Returns counts."""
    con.execute(f"CREATE OR REPLACE TABLE ent0 AS SELECT * FROM read_parquet('{ent_path.as_posix()}')")
    con.execute(
        """
        CREATE OR REPLACE TABLE tok AS
        SELECT DISTINCT entity_id, country, tok FROM (
            SELECT entity_id, country, unnest(string_split(name_core, ' ')) AS tok FROM ent0
        ) WHERE length(tok) >= 2
        """
    )
    con.execute("CREATE OR REPLACE TABLE tokdf AS SELECT country, tok, count(*) AS df FROM tok GROUP BY country, tok")
    con.execute("CREATE OR REPLACE TABLE ncountry AS SELECT country, count(*) AS n FROM ent0 GROUP BY country")
    # Same IDF definition as features.prepare_entities: ln((N+1)/(df+1)) / ln(N+1), N = entities in the country.
    con.execute(
        """
        CREATE OR REPLACE TABLE rare AS
        SELECT entity_id, tok AS rare_tok, idf AS max_idf FROM (
            SELECT t.entity_id, t.tok, ln((n.n + 1.0) / (d.df + 1)) / ln(n.n + 1.0) AS idf,
                   ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY d.df ASC, t.tok) AS rn
            FROM tok t JOIN tokdf d USING (country, tok) JOIN ncountry n USING (country)
        ) WHERE rn = 1
        """
    )
    con.execute("DROP TABLE tok")
    con.execute("DROP TABLE tokdf")
    con.execute(
        """
        CREATE OR REPLACE TABLE ent AS
        SELECT e.*, r.rare_tok, r.max_idf FROM ent0 e LEFT JOIN rare r USING (entity_id)
        """
    )
    con.execute("DROP TABLE ent0")
    con.execute("DROP TABLE rare")

    def block(name: str, key_cols: list, extra_where: str) -> None:
        on = " AND ".join(f"a.{c} = c.{c}" for c in key_cols)
        keys = ", ".join(key_cols)
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT s1_id, cand_id FROM (
                SELECT a.entity_id AS s1_id, c.entity_id AS cand_id,
                       ROW_NUMBER() OVER (PARTITION BY a.entity_id ORDER BY c.entity_id) AS rn
                FROM ent a
                JOIN (SELECT {keys} FROM ent WHERE src <> '1' AND {extra_where.replace('a.', '')}
                      GROUP BY {keys} HAVING count(*) <= {KEY_CAP}) k
                  ON {" AND ".join(f"k.{c} = a.{c}" for c in key_cols)}
                JOIN ent c ON {on}
                WHERE a.src = '1' AND c.src <> '1' AND {extra_where} AND {extra_where.replace('a.', 'c.')}
            ) WHERE rn <= {BLOCK_CAP}
            """
        )

    block("k1", ["country", "name_core"], "a.name_core <> ''")
    block("k2", ["country", "postal", "rare_tok"], "a.postal IS NOT NULL AND a.rare_tok IS NOT NULL")
    block("k3", ["country", "street", "first_tok"], f"length(a.street) >= {MIN_STREET_LEN} AND a.first_tok <> ''")
    # Extra exact keys (same spirit: one equality join each, within a country). K1-K3 alone reach only ~53% pair recall.
    block("k4", ["country", "street", "hn1"], f"length(a.street) >= {MIN_STREET_LEN} AND a.hn1 <> ''")          # same street + same house number
    block("k5", ["country", "street", "rare_tok"], f"length(a.street) >= {MIN_STREET_LEN} AND a.rare_tok IS NOT NULL")  # same street + same rarest name token
    block("k6", ["country", "name_sorted"], "a.name_sorted <> ''")                                              # same name tokens, any order
    con.execute("ALTER TABLE ent ADD COLUMN nospace4 VARCHAR")
    con.execute("UPDATE ent SET nospace4 = substr(nospace, 1, 4)")
    block("k7", ["country", "nospace"], "length(a.nospace) >= 5")                                                # same name, spacing ignored ("prasana associates" = "prasanaassociates")
    block("k8", ["country", "street", "nospace4"], f"length(a.street) >= {MIN_STREET_LEN} AND length(a.nospace) >= 4")  # same street + same first 4 letters (typo in the tail)
    block("k9", ["country", "rare_tok"], "a.rare_tok IS NOT NULL")                                               # same rarest name token (rare enough to pass KEY_CAP)
    con.execute(
        f"""
        COPY (
            SELECT s1_id, cand_id, max(k1) AS k1, max(k2) AS k2, max(k3) AS k3, max(k4) AS k4, max(k5) AS k5, max(k6) AS k6, max(k7) AS k7, max(k8) AS k8, max(k9) AS k9 FROM (
                SELECT s1_id, cand_id, 1 AS k1, 0 AS k2, 0 AS k3, 0 AS k4, 0 AS k5, 0 AS k6, 0 AS k7, 0 AS k8, 0 AS k9 FROM k1
                UNION ALL SELECT s1_id, cand_id, 0, 1, 0, 0, 0, 0, 0, 0, 0 FROM k2
                UNION ALL SELECT s1_id, cand_id, 0, 0, 1, 0, 0, 0, 0, 0, 0 FROM k3
                UNION ALL SELECT s1_id, cand_id, 0, 0, 0, 1, 0, 0, 0, 0, 0 FROM k4
                UNION ALL SELECT s1_id, cand_id, 0, 0, 0, 0, 1, 0, 0, 0, 0 FROM k5
                UNION ALL SELECT s1_id, cand_id, 0, 0, 0, 0, 0, 1, 0, 0, 0 FROM k6
                UNION ALL SELECT s1_id, cand_id, 0, 0, 0, 0, 0, 0, 1, 0, 0 FROM k7
                UNION ALL SELECT s1_id, cand_id, 0, 0, 0, 0, 0, 0, 0, 1, 0 FROM k8
                UNION ALL SELECT s1_id, cand_id, 0, 0, 0, 0, 0, 0, 0, 0, 1 FROM k9
            ) GROUP BY s1_id, cand_id
        ) TO '{cand_path.as_posix()}' (FORMAT PARQUET)
        """
    )
    counts = {k: con.execute(f"SELECT count(*) FROM {k}").fetchone()[0] for k in ("k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9")}
    counts["union"] = con.execute(f"SELECT count(*) FROM read_parquet('{cand_path.as_posix()}')").fetchone()[0]
    return counts


PAIR_SQL = """
SELECT p.s1_id, p.cand_id,
       (a.name_core = c.name_core AND a.name_core <> '') AS core_eq,
       (a.postal IS NOT NULL AND c.postal IS NOT NULL AND a.postal = c.postal) AS postal_eq,
       (a.postal IS NOT NULL AND c.postal IS NOT NULL AND a.postal <> c.postal) AS postal_conflict,
       (length(a.street) >= {min_street} AND a.street = c.street) AS street_eq,
       a.name_full AS a_full, c.name_full AS b_full, a.hn AS a_hn, c.hn AS b_hn, a.street AS a_street, c.street AS b_street,
       coalesce(a.max_idf, 0.0) AS idf
FROM read_parquet('{cand}') p JOIN ent a ON a.entity_id = p.s1_id JOIN ent c ON c.entity_id = p.cand_id
"""


@lru_cache(maxsize=500_000)
def _digits(s: str) -> tuple:
    return tuple(s.split(",")) if s else ()


def pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add name_sim and the house-number relation code to one batch of enriched pairs."""
    a, b = df["a_full"].tolist(), df["b_full"].tolist()
    df = df.copy()
    df["name_sim"] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, dtype=np.float32, workers=1) if a else np.empty(0, np.float32)
    df["name_sort"] = process.cpdist(a, b, scorer=fuzz.token_sort_ratio, dtype=np.float32, workers=1) if a else np.empty(0, np.float32)
    sa, sb = df["a_street"].tolist(), df["b_street"].tolist()
    sim = process.cpdist(sa, sb, scorer=fuzz.token_set_ratio, dtype=np.float32, workers=1) if sa else np.empty(0, np.float32)
    sim[[(len(x) < MIN_STREET_LEN or len(y) < MIN_STREET_LEN) for x, y in zip(sa, sb)]] = 0.0
    df["street_sim"] = sim
    df["hn_rel"] = np.fromiter(
        (features.house_number_relation(_digits(x), _digits(y)) for x, y in zip(df["a_hn"], df["b_hn"])),
        dtype=np.int8, count=len(df),
    )
    return df.drop(columns=["a_full", "b_full", "a_hn", "b_hn", "a_street", "b_street"])


def iter_pair_features(con, cand_path: Path):
    reader = con.execute(PAIR_SQL.format(cand=cand_path.as_posix(), min_street=MIN_STREET_LEN)).fetch_record_batch(PAIR_BATCH_ROWS)
    for batch in reader:
        yield pair_features(batch.to_pandas())


# ----------------------------------------------------------------------------- rules
def apply_rules(F: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rows of F that satisfy at least one enabled rule, with a score for one-to-one."""
    hn_ok = F["hn_rel"].isin(list(HN_OK_MODES[cfg["hn_mode"]]))
    street_ok = (F["street_sim"] >= cfg["street_min"]) & hn_ok      # 100 = identical street text; lower = near-exact
    r1 = F["core_eq"] & (F["postal_eq"] | street_ok)
    r2 = (F[cfg["sim"]] >= cfg["x"]) & street_ok           # "name_sim" = token_set_ratio (extra words ignored), "name_sort" = token_sort_ratio (extra words penalised)
    r3 = F["core_eq"] & ~F["postal_conflict"] & (F["idf"] >= cfg["idf_min"]) & (F["street_sim"] >= cfg["r3_street"])
    enabled = {"R1": r1, "R2": r2, "R3": r3}
    hit = np.zeros(len(F), dtype=np.int8)
    for name in cfg["rules"]:
        hit += enabled[name].to_numpy(dtype=np.int8)
    out = F.loc[hit > 0, ["s1_id", "cand_id", "name_sim", "postal_eq", "street_sim"]].copy()
    out["score"] = hit[hit > 0] * 100.0 + out["name_sim"].astype(float) + out["postal_eq"] * 10.0 + out["street_sim"].astype(float) / 20.0
    return out


def one_to_one(matched: pd.DataFrame) -> pd.DataFrame:
    """Each candidate id keeps only its best-scoring S1 (ties: smaller S1 id)."""
    if matched.empty:
        return matched
    return (matched.sort_values(["score", "s1_id"], ascending=[False, True])
            .drop_duplicates("cand_id", keep="first"))


def to_preds(matched: pd.DataFrame, ids) -> dict:
    preds = {i: set() for i in ids}
    for s1, cand in zip(matched["s1_id"], matched["cand_id"]):
        if s1 in preds:
            preds[s1].add(cand)
    return preds


# ----------------------------------------------------------------------------- validation slice
def _val_batches(val_dir: Path):
    for i in (1, 2, 3):
        cols = ["entity_id", "business_name", "business_address", "country"]
        df = pd.read_parquet(val_dir / f"source{i}.parquet", columns=cols)
        for a in range(0, len(df), CHUNK_ROWS):
            yield df.iloc[a:a + CHUNK_ROWS]


def load_truth(val_dir: Path) -> tuple:
    s1 = pd.read_parquet(val_dir / "source1.parquet", columns=["entity_id", "is_eval", "country"])
    eval_ids = set(s1.loc[s1["is_eval"], "entity_id"])
    country = dict(zip(s1["entity_id"], s1["country"]))
    gt = pd.read_parquet(val_dir / "ground_truth_in_slice.parquet")
    truths = {a: (set(b.split(",")) if isinstance(b, str) and b.strip() else set())
              for a, b in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if a in eval_ids}
    return eval_ids, truths, country


def score(matched: pd.DataFrame, eval_ids: set, truths: dict, country: dict) -> dict:
    preds = to_preds(one_to_one(matched), eval_ids)
    rep = evaluate.score_report(preds, truths, group_of=country)
    rep["singleton_fp"] = sum(1 for a, t in truths.items() if not t and preds.get(a))
    rep["n_singletons"] = sum(1 for t in truths.values() if not t)
    return rep


def run_val(args, suffix_sets, translit_map) -> dict:
    work = config.WORK_DIR / "rules"
    work.mkdir(parents=True, exist_ok=True)
    val_dir = config.PARQUET_DIR / "val_slice"
    ent_path, cand_path = work / "val_entities.parquet", work / "val_candidates.parquet"
    t0 = time.time()
    n = write_entities(_val_batches(val_dir), ent_path, suffix_sets, translit_map)
    con = _new_connection(args.duckdb_memory, args.duckdb_threads)
    counts = build_candidates(con, ent_path, cand_path)
    F = pd.concat(list(iter_pair_features(con, cand_path)), ignore_index=True)
    con.close()
    eval_ids, truths, country = load_truth(val_dir)
    print(f"[val] {n} records, candidate pairs {counts}, prep {time.time() - t0:.0f}s", flush=True)

    # candidate recall (per pair) on the eval S1s
    cand_by = F.groupby("s1_id")["cand_id"].apply(set).to_dict()
    tp = tot = 0
    for s, t in truths.items():
        tot += len(t)
        tp += len(t & cand_by.get(s, set()))
    print(f"[val] candidate recall (pairs): {tp / max(tot, 1):.4f}   avg candidates/S1: "
          f"{len(F) / max(F['s1_id'].nunique(), 1):.1f}", flush=True)

    # grid search on macro F0.5
    results = []
    for rules, x, hn_mode, idf_min, street_min, r3_street, sim in product(RULE_SETS, (80, 85, 90, 95), HN_OK_MODES, (0.3, 0.5, 0.6), (100, 90, 80), (0, 50, 70), ("name_sim", "name_sort")):
        if "R2" not in rules and x != 90:
            continue                     # x only matters for R2
        if "R3" not in rules and idf_min != 0.5:
            continue                     # idf_min only matters for R3
        if not ({"R1", "R2"} & set(rules)) and hn_mode != "near":
            continue                     # hn_mode only matters for R1/R2
        if "R2" not in rules and sim != "name_sim":
            continue                     # sim only matters for R2
        if "R3" not in rules and r3_street != 0:
            continue                     # r3_street only matters for R3
        if not ({"R1", "R2"} & set(rules)) and street_min != 100:
            continue                     # street_min only matters for R1/R2
        cfg = {"rules": list(rules), "x": x, "hn_mode": hn_mode, "idf_min": idf_min, "street_min": street_min, "r3_street": r3_street, "sim": sim}
        results.append((evaluate.macro_f_beta(to_preds(one_to_one(apply_rules(F, cfg)), eval_ids), truths), cfg))
    results.sort(key=lambda r: -r[0])
    print("[val] top configs (macro F0.5):", flush=True)
    for f, cfg in results[:8]:
        print(f"   {f:.4f}  {cfg}", flush=True)
    best_f, best_cfg = results[0]

    # honest check: tune on one half of the S1s, score the other half (and vice versa)
    ids = sorted(eval_ids)
    halves = [set(ids[0::2]), set(ids[1::2])]
    cross = []
    for tune_ids, test_ids in (halves, halves[::-1]):
        Ft = F[F["s1_id"].isin(tune_ids | (set(F["s1_id"]) - eval_ids))]
        best = max(
            (evaluate.macro_f_beta(to_preds(one_to_one(apply_rules(Ft, c)), tune_ids), {k: truths[k] for k in tune_ids}), i)
            for i, (_, c) in enumerate(results[:60])
        )
        cfg_i = results[best[1]][1]
        cross.append(evaluate.macro_f_beta(to_preds(one_to_one(apply_rules(F, cfg_i)), test_ids), {k: truths[k] for k in test_ids}))
    print(f"[val] two-fold check (tune on one half of S1, score the other): {np.mean(cross):.4f}  (halves: {cross[0]:.4f}, {cross[1]:.4f})", flush=True)

    rep = score(apply_rules(F, best_cfg), eval_ids, truths, country)
    (work / "config.json").write_text(json.dumps(best_cfg), encoding="utf-8")
    print("\n" + "=" * 78 + "\nRULES BASELINE -- validation slice (seed-42 slice, eval S1 only)\n" + "=" * 78)
    print(f"  config: {best_cfg}")
    print(f"  macro F0.5: {rep['macro_f_beta']:.4f}   micro precision: {rep['micro_precision']:.4f}   micro recall: {rep['micro_recall']:.4f}")
    print(f"  singleton false positives: {rep['singleton_fp']} / {rep['n_singletons']}")
    for c, g in sorted(rep["by_group"].items()):
        print(f"    {c:8s} n={g['n']:>5}  F0.5={g['f_beta']:.4f}  P={g['precision']:.4f}  R={g['recall']:.4f}")
    ml = args.ml_score
    print(f"  ML result on this slice: {ml:.4f}   ->   rules baseline is {rep['macro_f_beta'] - ml:+.4f} vs ML")
    print("=" * 78, flush=True)
    rep["config"] = best_cfg
    rep["two_fold_check"] = cross
    return rep


# ----------------------------------------------------------------------------- test set
def _test_batches(con):
    for i in (1, 2, 3):
        io_utils.register_source_view(con, f"test_source{i}", getattr(config, f"TEST_SOURCE{i}"))
        reader = con.execute(f"SELECT * FROM test_source{i}").fetch_record_batch(CHUNK_ROWS)
        for batch in reader:
            yield batch.to_pandas()


def run_test(args, suffix_sets, translit_map) -> dict:
    work = config.WORK_DIR / "rules"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else DEFAULT_CFG
    print(f"[test] rule config: {cfg}", flush=True)
    ent_path, cand_path = work / "test_entities.parquet", work / "test_candidates.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else config.OUTPUT_DIR / "rules"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rcon = duckdb.connect()
    rcon.execute("SET memory_limit = '1GB'")
    n = write_entities(_test_batches(rcon), ent_path, suffix_sets, translit_map)
    print(f"[test] entity keys for {n:,} records ({time.time() - t0:.0f}s)", flush=True)

    con = _new_connection(args.duckdb_memory, args.duckdb_threads)
    counts = build_candidates(con, ent_path, cand_path)
    print(f"[test] candidate pairs {counts} ({time.time() - t0:.0f}s)", flush=True)

    kept = []
    n_pairs = 0
    for F in iter_pair_features(con, cand_path):
        n_pairs += len(F)
        kept.append(apply_rules(F, cfg))
    matched = one_to_one(pd.concat(kept, ignore_index=True)) if kept else pd.DataFrame(columns=["s1_id", "cand_id", "score"])
    print(f"[test] scored {n_pairs:,} pairs -> {len(matched):,} matched pairs after one-to-one ({time.time() - t0:.0f}s)", flush=True)

    s1_country = dict(rcon.execute("SELECT entity_id, country FROM test_source1").fetchall())
    rcon.close()
    con.close()
    required = set(s1_country)
    preds = to_preds(matched, required)
    matching_path, candidate_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    write_outputs.write_matching_results(preds, required, matching_path)
    # candidate_pairs.tsv wants (s1_id, cand_id) only: re-export those two columns
    ccon = duckdb.connect()
    ccon.execute(f"COPY (SELECT s1_id, cand_id FROM read_parquet('{cand_path.as_posix()}')) TO '{(work / 'test_cand_ids.parquet').as_posix()}' (FORMAT PARQUET)")
    ccon.close()
    n_with = write_outputs.write_candidate_pairs_from_parquet((work / "test_cand_ids.parquet").as_posix(), required, candidate_path)
    code, output = write_outputs.run_validator(matching_path, candidate_path, config.TEST_DIR)
    print(output)
    print("VALIDATOR:", "PASS" if code == 0 else "FAIL", flush=True)

    per_country = {}
    for c in sorted(set(s1_country.values())):
        ids = [i for i, cc in s1_country.items() if cc == c]
        per_country[c] = (len(ids), sum(1 for i in ids if preds[i]))
    print("\n% of S1 with at least one match, by country:")
    for c, (tot, m) in per_country.items():
        print(f"   {c:8s} {m:>9,} / {tot:>9,}  = {m / max(tot, 1):6.1%}")
    print(f"   S1 with at least one candidate: {n_with:,} / {len(required):,}")
    return {"validator_exit": code, "per_country": per_country, "n_pairs": n_pairs, "n_matched_pairs": len(matched)}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test", "both"], default="val")
    ap.add_argument("--duckdb-memory", default="2GB", help="DuckDB memory_limit (spills to disk past it). Default suits a ~6GB laptop.")
    ap.add_argument("--duckdb-threads", type=int, default=2)
    ap.add_argument("--out-dir", default=None, help="Where the test TSVs go (default: <output dir>/rules).")
    ap.add_argument("--ml-score", type=float, default=0.9653, help="ML validation macro F0.5, for the comparison line.")
    ap.add_argument("--min-val-score", type=float, default=0.85, help="With --split both: only run on test if the validation score is at least this.")
    args = ap.parse_args()

    peak = {"mb": 0.0}
    stop = threading.Event()
    if psutil is not None:
        def poll():
            proc = psutil.Process()
            while not stop.wait(1):
                peak["mb"] = max(peak["mb"], proc.memory_info().rss / 1e6 + sum(c.memory_info().rss / 1e6 for c in proc.children(recursive=True)))
        threading.Thread(target=poll, daemon=True).start()

    t0 = time.time()
    suffix_sets, translit_map = load_resources()
    rep = None
    if args.split in ("val", "both"):
        rep = run_val(args, suffix_sets, translit_map)
    if args.split == "test" or (args.split == "both" and rep["macro_f_beta"] >= args.min_val_score):
        run_test(args, suffix_sets, translit_map)
    elif args.split == "both":
        print(f"[both] validation score {rep['macro_f_beta']:.4f} < {args.min_val_score}: not running on test.")
    stop.set()
    print(f"\nTOTAL runtime {(time.time() - t0) / 60:.1f} min, peak RAM (this process + children) {peak['mb']:.0f} MB")


if __name__ == "__main__":
    main()
