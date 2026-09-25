"""DuckDB-backed I/O helpers.

Every stage that touches a full source file (2.2M-5.3M rows each) goes through
DuckDB rather than pandas: DuckDB streams off disk and spills past its memory
cap instead of materializing everything in RAM, which matters on an 8GB box
with ~1.6GB typically free. Pandas is still fine downstream, once a query has
already cut a file down to a small, already-filtered slice (e.g. one
candidate-pair feature table for a fold).

All source columns are read as VARCHAR on purpose: entity_id / business_name /
business_address / country are free text, and letting DuckDB's type sniffer
guess a numeric or date type for a stray-looking value (e.g. an address that's
just digits) would silently corrupt it.
"""

from pathlib import Path
from typing import Iterable, Optional

import duckdb

from . import config

SOURCE_COLUMNS = [
    config.COL_ENTITY_ID,
    config.COL_BUSINESS_NAME,
    config.COL_BUSINESS_ADDRESS,
    config.COL_COUNTRY,
]

GROUND_TRUTH_COLUMNS = [
    config.COL_SOURCE1_ENTITY_ID,
    config.COL_MATCHED_ENTITY_IDS,
]


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the shared on-disk DuckDB database with memory-safe settings.

    Inputs: read_only (True to avoid taking the write lock when only querying).
    Output: a configured duckdb connection; caller is responsible for closing it
    (or using it as a context manager).
    """
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=read_only)
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {config.DUCKDB_THREADS}")
    con.execute(f"SET temp_directory = '{config.DUCKDB_TMP_DIR.as_posix()}'")
    con.execute(f"SET max_temp_directory_size = '{config.DUCKDB_MAX_TEMP_DIRECTORY_SIZE}'")
    # Spill to disk aggressively rather than risk an OOM on an 8GB machine.
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET enable_progress_bar = false")
    return con


def _read_csv_expr(path: Path, columns: Iterable[str]) -> str:
    """Build a read_csv_auto(...) SQL expression that forces VARCHAR columns.

    Inputs: path to a .tsv file, the expected column names in file order.
    Output: a SQL fragment usable inside FROM/CREATE VIEW.
    """
    types = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    posix_path = path.as_posix()
    return (
        f"read_csv('{posix_path}', delim='\\t', header=true, quote='', "
        f"escape='', columns={{{types}}}, strict_mode=false)"
    )


def register_source_view(
    con: duckdb.DuckDBPyConnection, view_name: str, path: Path
) -> None:
    """Create/replace a DuckDB view over one source TSV (S1/S2/S3 schema).

    Inputs: open connection, name for the view, path to the .tsv file.
    Output: none; the view is registered on the connection.
    """
    expr = _read_csv_expr(path, SOURCE_COLUMNS)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {expr}")


def register_ground_truth_view(
    con: duckdb.DuckDBPyConnection, view_name: str = "train_ground_truth"
) -> None:
    """Create/replace a DuckDB view over train_ground_truth.tsv.

    Inputs: open connection, optional view name override.
    Output: none; the view is registered on the connection.
    """
    expr = _read_csv_expr(config.TRAIN_GROUND_TRUTH, GROUND_TRUTH_COLUMNS)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {expr}")


def register_all_standard_views(con: duckdb.DuckDBPyConnection) -> None:
    """Register the six standard source views plus ground truth, by convention.

    Inputs: open connection.
    Output: none. Registers: train_source1, train_source2, train_source3,
    test_source1, test_source2, test_source3, train_ground_truth.
    """
    register_source_view(con, "train_source1", config.TRAIN_SOURCE1)
    register_source_view(con, "train_source2", config.TRAIN_SOURCE2)
    register_source_view(con, "train_source3", config.TRAIN_SOURCE3)
    register_source_view(con, "test_source1", config.TEST_SOURCE1)
    register_source_view(con, "test_source2", config.TEST_SOURCE2)
    register_source_view(con, "test_source3", config.TEST_SOURCE3)
    register_ground_truth_view(con)


def export_parquet(
    con: duckdb.DuckDBPyConnection, query: str, out_path: Path
) -> Path:
    """Run a SQL query and stream its result to a parquet file on WORK_DIR (D:).

    Inputs: open connection, a full SELECT query, destination path.
    Output: the same out_path, for chaining. Uses COPY so DuckDB streams the
    result to disk rather than materializing it fully in Python first.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT PARQUET)")
    return out_path


def row_count(con: duckdb.DuckDBPyConnection, relation: str) -> int:
    """Cheap row count for a view/table/parquet-glob.

    Inputs: open connection, a relation name or a SQL FROM-expression.
    Output: row count as int.
    """
    return con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
