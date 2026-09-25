"""Central paths, constants, and knobs shared by every pipeline stage.

Kept dependency-free (stdlib only) so every other module can import it cheaply.
Every path below is overridable via an AML_* environment variable (which
scripts/run_pipeline.py's CLI flags set) so the exact same code runs
unmodified on the local dev machine (Windows, D: drive) and on Kaggle
(Windows or Linux, /kaggle/input + /kaggle/working) -- nothing here assumes a
specific OS or drive letter; the only OS-specific default (D:) is used only
when that drive actually exists, and falls back to a repo-relative path
otherwise.
"""

import os
from pathlib import Path

# student_resource/code/business_entity_resolution/src/config.py -> student_resource/
REPO_ROOT = Path(__file__).resolve().parents[3]

# Dataset root (contains train/ and test/). On Kaggle this is normally
# /kaggle/input/<dataset-name>/dataset -- set AML_DATA_DIR to that path.
DATA_DIR = Path(os.environ.get("AML_DATA_DIR", str(REPO_ROOT / "dataset")))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Final submission files. On Kaggle set AML_OUTPUT_DIR to somewhere under
# /kaggle/working (the only persisted-output location there).
OUTPUT_DIR = Path(os.environ.get("AML_OUTPUT_DIR", str(REPO_ROOT / "output")))

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"


def _default_work_dir() -> str:
    """D:/amazon_ml_work on this dev machine (D: has ~490GB free vs C:'s
    ~5GB); a repo-relative .work/ directory anywhere else (Kaggle, another
    teammate's machine, CI) where D: doesn't exist. AML_WORK_DIR always wins
    over both.
    """
    d_drive = Path("D:/")
    if d_drive.exists():
        return "D:/amazon_ml_work"
    return str(REPO_ROOT / ".work")


# Large-artifact working directory (parquet/duckdb/tmp intermediates -- never
# committed, see .gitignore). On Kaggle set AML_WORK_DIR to /kaggle/temp/... :
# temp is fast local scratch and doesn't count against output size limits.
WORK_DIR = Path(os.environ.get("AML_WORK_DIR", _default_work_dir()))
DUCKDB_DIR = WORK_DIR / "duckdb"
DUCKDB_TMP_DIR = WORK_DIR / "tmp"
PARQUET_DIR = WORK_DIR / "parquet"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
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
