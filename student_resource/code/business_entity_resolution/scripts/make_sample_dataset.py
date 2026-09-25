"""Build a tiny, self-contained copy of the challenge dataset for LOCAL crash
testing of the full pipeline (used by `scripts/run_pipeline.py --sample N`;
never used for a real run).

Layout written under <out_dir>/ (same schema as the real dataset):
    train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
    test/test_source1.tsv,  test_source2.tsv,  test_source3.tsv

Train sample: N Source-1 rows (spread evenly over the countries present,
deterministic hash order), every Source-2/3 record that is a true match of
one of them, plus DECOYS_PER_S1 random other records per source per S1 so
blocking has something to reject. Ground truth is restricted to the sampled S1.
Test sample: N Source-1 rows spread over ALL test countries (so France, which
has no training rows, is exercised), plus a random pool of Source-2/3 records
from the same countries. The test pool is random, so few true matches exist
in it -- this is a crash test, not a quality test.

Standalone on purpose (no src.config import): it runs BEFORE run_pipeline has
decided which data directory to use. Reads the real TSVs through DuckDB, so
nothing bigger than the sample is ever held in memory.
"""

import csv
from pathlib import Path

import duckdb
import pandas as pd

DECOYS_PER_S1 = 3
COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _view(con, name: str, path: Path, columns: list) -> None:
    types = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv('{path.as_posix()}', delim='\\t', "
        f"header=true, quote='', escape='', columns={{{types}}}, strict_mode=false)"
    )


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")


def _per_country_s1(con, view: str, n: int) -> pd.DataFrame:
    """About n rows of `view`, split evenly over its countries, in a fixed
    (hash) order so the same n always gives the same sample."""
    countries = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM {view} ORDER BY country").fetchall()]
    per = max(1, n // len(countries))
    return con.execute(
        f"""
        SELECT entity_id, business_name, business_address, country FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY hash(entity_id)) AS rn FROM {view}
        ) WHERE rn <= {per} ORDER BY entity_id
        """
    ).fetchdf()


def build_sample(data_dir: Path, out_dir: Path, n_s1: int) -> Path:
    """Write the sample dataset under out_dir and return out_dir. Skips the
    work (returns immediately) if a sample of the same size already exists."""
    marker = out_dir / f"_sample_{n_s1}.ok"
    if marker.exists():
        return out_dir

    con = duckdb.connect()
    con.execute("SET threads = 2")
    for split in ("train", "test"):
        for i in (1, 2, 3):
            _view(con, f"{split}_source{i}", data_dir / split / f"{split}_source{i}.tsv", COLUMNS)
    _view(con, "train_ground_truth", data_dir / "train" / "train_ground_truth.tsv", GT_COLUMNS)

    # ---- train
    s1 = _per_country_s1(con, "train_source1", n_s1)
    con.register("sample_s1_ids", s1[["entity_id"]])
    gt = con.execute(
        "SELECT g.* FROM train_ground_truth g JOIN sample_s1_ids s ON g.source1_entity_id = s.entity_id"
    ).fetchdf()
    con.register("sample_gt", gt)
    con.execute(
        "CREATE OR REPLACE VIEW sample_matched AS SELECT DISTINCT unnest(string_split(matched_entity_ids, ',')) AS entity_id "
        "FROM sample_gt WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''"
    )
    n_decoys = DECOYS_PER_S1 * n_s1
    _write(s1, out_dir / "train" / "train_source1.tsv")
    _write(gt, out_dir / "train" / "train_ground_truth.tsv")
    for i in (2, 3):
        pool = con.execute(
            f"""
            SELECT * FROM train_source{i}
            WHERE entity_id IN (SELECT entity_id FROM sample_matched)
               OR entity_id IN (SELECT entity_id FROM train_source{i} ORDER BY hash(entity_id) LIMIT {n_decoys})
            ORDER BY entity_id
            """
        ).fetchdf()
        _write(pool, out_dir / "train" / f"train_source{i}.tsv")

    # ---- test (every country, France included)
    t1 = _per_country_s1(con, "test_source1", n_s1)
    _write(t1, out_dir / "test" / "test_source1.tsv")
    per_country_pool = max(1, (DECOYS_PER_S1 * n_s1) // t1["country"].nunique())
    for i in (2, 3):
        pool = con.execute(
            f"""
            SELECT entity_id, business_name, business_address, country FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY hash(entity_id)) AS rn FROM test_source{i}
            ) WHERE rn <= {per_country_pool} ORDER BY entity_id
            """
        ).fetchdf()
        _write(pool, out_dir / "test" / f"test_source{i}.tsv")

    con.close()
    marker.write_text("ok", encoding="utf-8")
    return out_dir
