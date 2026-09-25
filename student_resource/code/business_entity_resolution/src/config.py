"""Central paths, constants, and knobs shared by every pipeline stage.

Kept dependency-free (stdlib only) so every other module can import it cheaply.
All large/generated artifacts are pinned to WORK_DIR on the D: drive: the C:
drive on the dev machine has only ~5GB free, nowhere near enough for the raw
data plus intermediate parquet/DuckDB files, while D: has ~490GB free.
"""

import os
from pathlib import Path

# student_resource/code/business_entity_resolution/src/config.py -> student_resource/
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = REPO_ROOT / "output"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Large-artifact working directory. Overridable via AML_WORK_DIR so teammates
# on a different machine/drive layout are not hardcoded to D:.
WORK_DIR = Path(os.environ.get("AML_WORK_DIR", "D:/amazon_ml_work"))
DUCKDB_DIR = WORK_DIR / "duckdb"
DUCKDB_TMP_DIR = WORK_DIR / "tmp"
PARQUET_DIR = WORK_DIR / "parquet"
SUBMISSIONS_DIR = WORK_DIR / "submissions"
DUCKDB_PATH = DUCKDB_DIR / "aml.duckdb"

# DuckDB memory ceiling. Machine has 8GB total / ~1.6GB free at times, so keep
# this conservative and let DuckDB spill to DUCKDB_TMP_DIR (on D:) past this.
# threads=2 (not more) and max_temp_directory_size are both explicit after two
# blocking-stage OOM crashes (Phase 3, PROJECT_LOG.md) -- fewer threads means
# fewer large intermediates alive at once, and a hard temp-dir cap turns "fill
# the disk" into a normal query error instead of a machine-wide problem.
DUCKDB_MEMORY_LIMIT = os.environ.get("AML_DUCKDB_MEMORY_LIMIT", "3GB")
DUCKDB_THREADS = int(os.environ.get("AML_DUCKDB_THREADS", "2"))
DUCKDB_MAX_TEMP_DIRECTORY_SIZE = os.environ.get("AML_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "30GB")

# Entity id prefixes / column names, used instead of literals across modules.
SOURCE1_PREFIX = "S1-"
SOURCE2_PREFIX = "S2-"
SOURCE3_PREFIX = "S3-"

COL_ENTITY_ID = "entity_id"
COL_BUSINESS_NAME = "business_name"
COL_BUSINESS_ADDRESS = "business_address"
COL_COUNTRY = "country"

COL_SOURCE1_ENTITY_ID = "source1_entity_id"
COL_MATCHED_ENTITY_IDS = "matched_entity_ids"
COL_CANDIDATE_ENTITY_IDS = "candidate_entity_ids"

# Scoring.
F_BETA = 0.5

RANDOM_SEED = 42

for _dir in (WORK_DIR, DUCKDB_DIR, DUCKDB_TMP_DIR, PARQUET_DIR, SUBMISSIONS_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
