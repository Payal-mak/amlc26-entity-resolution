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

# DuckDB memory ceiling. Defaults now assume Kaggle (31GB RAM, 4 CPUs) -- the
# real/full-size runs all happen there as of 2026-09-25 (see PROJECT_LOG.md:
# repeated local OOMs even after several rounds of memory-safety patches).
# Local runs are dev-only smoke tests on tiny samples, where the actual usage
# stays far under whatever the ceiling is set to regardless -- override with
# AML_DUCKDB_MEMORY_LIMIT/AML_DUCKDB_THREADS if that ever stops being true.
DUCKDB_MEMORY_LIMIT = os.environ.get("AML_DUCKDB_MEMORY_LIMIT", "20GB")
DUCKDB_THREADS = int(os.environ.get("AML_DUCKDB_THREADS", "4"))
# recall-v2 (2026-09-25): raised 30GB -> 200GB after a real Kaggle crash hit
# this exact soft cap (not real disk exhaustion -- see the longer note next
# to GEO_MAX_CAND_TOKEN_DF below, right before it's actually used).
DUCKDB_MAX_TEMP_DIRECTORY_SIZE = os.environ.get("AML_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "200GB")

# LightGBM knobs (src/model.py). Same reasoning as the DuckDB settings above:
# Kaggle-appropriate defaults (real thread count, the library's own default
# max_bin, two_round off), overridable via env var for the rare case a local
# run needs the laptop-safe compromises back (max_bin=63/two_round=True/
# n_jobs=1 -- see PROJECT_LOG.md for why those existed briefly and were
# reverted once Kaggle was confirmed working).
LGBM_NUM_THREADS = int(os.environ.get("AML_LGBM_THREADS", "4"))
LGBM_MAX_BIN = int(os.environ.get("AML_LGBM_MAX_BIN", "255"))
LGBM_TWO_ROUND = os.environ.get("AML_LGBM_TWO_ROUND", "false").strip().lower() in ("1", "true", "yes")

# recall-v2 (2026-09-25): src/blocking.py's char-3-gram TF-IDF block
# (block_b_tfidf_char_ngram) measured very strong standalone recall on a
# local SUBSET test (77% India / 94% US -- see PROJECT_LOG.md), but its real
# full-country-scale cost (a sparse matmul, not an indexed SQL join like
# every other block) was deliberately never run locally on this 8GB machine
# per explicit instruction -- the first real Kaggle run is the first time it
# runs at full scale. This flag exists so it can be turned off with no code
# change (AML_ENABLE_TFIDF_BLOCK=false) if it turns out too slow/memory-heavy
# there, without losing the rest of the recall-v2 blocking changes.
ENABLE_TFIDF_BLOCK = os.environ.get("AML_ENABLE_TFIDF_BLOCK", "true").strip().lower() in ("1", "true", "yes")

# Kaggle full-train-scale crash (2026-09-25, first real Kaggle run):
# block_b_geo OOM'd on India (883k S1 / 4.1M candidates -- ~5x the
# validation-slice scale every recall-v2 measurement above was done at).
# Unlike B2, block_b_geo had no candidate-side document-frequency cap on its
# join at all -- register_geo_tokens picks each record's OWN rarest
# available token, but that token can still be extremely common across the
# CANDIDATE pool as a whole (e.g. a common locality name that's merely the
# least-bad option for a poorly-detailed address), and the join fans out on
# EVERY such token before the per-S1 top-K cut (which happens after, too
# late to bound the join itself -- same class of bug that shaped this whole
# module, see its docstring). Never swept/tuned locally (this scale doesn't
# fit on the 8GB dev machine at all); chosen as a safety margin, not a
# fitted optimum -- override via this env var if it still needs adjusting.
GEO_MAX_CAND_TOKEN_DF = int(os.environ.get("AML_GEO_MAX_CAND_TOKEN_DF", "3000"))

# Same crash: block_and_cap_country was being called with an entire
# country's S1 set at once (883k for India) against the full candidate pool
# (4.1M) -- every block's join input was that big, all at once. Batching S1
# within each country bounds every block's join to (this many S1) x (the
# full candidate pool) instead, regardless of how big the country is overall.
S1_BATCH_SIZE = int(os.environ.get("AML_S1_BATCH_SIZE", "100000"))

# Persistent checkpoint dir (survives a Kaggle SESSION restart, unlike
# /tmp/WORK_DIR) -- e.g. /kaggle/working/ckpt. None (the default) disables
# checkpointing entirely; scripts/run_pipeline.py is the only reader.
CKPT_DIR = os.environ.get("AML_CKPT_DIR")

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

# Two-stage model (src/model.py train_two_stage_oof): stage 2 re-scores every pair
# with context features rebuilt from stage-1 out-of-fold probabilities. Off by
# default; switch on with --two-stage on scripts/run_pipeline.py or AML_TWO_STAGE=true.
TWO_STAGE = os.environ.get("AML_TWO_STAGE", "false").strip().lower() in ("1", "true", "yes")

# Scoring.
F_BETA = 0.5

RANDOM_SEED = 42

for _dir in (WORK_DIR, DUCKDB_DIR, DUCKDB_TMP_DIR, PARQUET_DIR, SUBMISSIONS_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
