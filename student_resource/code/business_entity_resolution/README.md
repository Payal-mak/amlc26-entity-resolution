# Business Entity Resolution -- Team 8bit

Reproduction instructions for the pipeline that produces `output/matching_results.tsv`
and `output/candidate_pairs.tsv`. See `../../PROJECT_LOG.md` (repo root, one level above
`student_resource/`) for the full dated history of decisions, measurements, and
experiments behind this code -- this README only covers how to run it.

## Setup

```
pip install -r requirements.txt
```

Python 3.11 used in development. No GPU required for the pipeline as built (Phase 1-5);
an optional embedding/cross-encoder add-on, if used, runs separately on a cloud notebook
(Kaggle/Colab) -- see PROJECT_LOG.md.

### Paths: local vs. Kaggle

Every path (dataset location, large-artifact scratch dir, final output dir, DuckDB
memory/thread settings) is read from an `AML_*` environment variable at import time
(`src/config.py`) -- nothing in the code hardcodes `D:/` or any other Windows-specific
path. `scripts/run_pipeline.py`'s CLI flags are a thin wrapper that just set these env
vars before anything else imports `src.config`, so the identical code runs locally and
on Kaggle:

| Variable | Flag | Local default | Kaggle value |
|---|---|---|---|
| `AML_DATA_DIR` | `--data-dir` | `dataset/` (this repo) | `/kaggle/input/<dataset-name>/dataset` |
| `AML_WORK_DIR` | `--work-dir` | `D:/amazon_ml_work` if `D:` exists, else `.work/` | `/kaggle/temp/amazon_ml_work` |
| `AML_OUTPUT_DIR` | `--output-dir` | `output/` (this repo) | `/kaggle/working/output` |
| `AML_DUCKDB_MEMORY_LIMIT` | `--duckdb-memory` | `3GB` (was lowered to `1GB` locally during a real-memory crisis -- see PROJECT_LOG.md) | `20GB`+ (Kaggle gives ~30GB RAM) |
| `AML_DUCKDB_THREADS` | `--duckdb-threads` | `2` | `4` |

`AML_WORK_DIR` holds intermediate artifacts (DuckDB database, parquet files) that are
never committed (see `.gitignore`) and never need to survive between runs; on Kaggle,
point it at `/kaggle/temp` (fast local scratch, doesn't count against output size
limits), not `/kaggle/working`.

Setting any of these directly (running an individual `scripts/phaseN_*.py` rather than
`run_pipeline.py`) still works the same way:

```
set AML_WORK_DIR=E:\some\other\path      # Windows cmd
$env:AML_WORK_DIR = "E:\some\other\path" # PowerShell
export AML_WORK_DIR=/some/other/path     # bash
```

## Pipeline stages

### Validation harness (local, on a held-out slice of train)

Run as modules from this directory (`code/business_entity_resolution/`), one process
per stage -- each stage loads only what it needs and exits, which keeps peak memory
predictable on an 8GB dev machine (DuckDB, used for anything touching a full source
file, streams off disk rather than materializing everything in RAM). These stages
build and score a held-out slice of `dataset/train/`, never touch `dataset/test/`, and
exist purely to measure/tune before spending a real submission.

| # | Command | Status |
|---|---|---|
| 1 | `python -m scripts.build_validation_split --target-total 45000` | done |
| 2 | `python -m scripts.phase2_normalization_report` then `python -m scripts.build_translit_token_map` | done |
| 3 | `python -m scripts.phase3_blocking_report` | done (recall ceiling 79.98% uncapped / 79.07% capped -- see PROJECT_LOG.md) |
| 4 | `python -m scripts.phase4_build_features` then `python -m scripts.phase4_train_and_decide` | in progress |

Run unit tests:

```
python -m unittest discover -s tests -v
```

### Real end-to-end run (`scripts/run_pipeline.py`) -> submission files

This is the single entry point that actually produces `output/matching_results.tsv` +
`output/candidate_pairs.tsv`: normalize -> block -> build features -> train (GroupKFold
LightGBM on the FULL train set) -> predict (score `dataset/test/`, no ground truth,
decision layer only) -> write + validate + archive. Every stage writes its output to
disk before the next starts, so `--stage <name>` resumes after a crash without redoing
earlier stages.

Local:

```
python -m scripts.run_pipeline
```

Kaggle (Windows or Linux notebook; adjust `<dataset-name>` to whatever the competition
dataset is mounted as under `/kaggle/input/`):

```
python -m scripts.run_pipeline \
    --data-dir /kaggle/input/<dataset-name>/dataset \
    --work-dir /kaggle/temp/amazon_ml_work \
    --output-dir /kaggle/working/output \
    --duckdb-memory 20GB --duckdb-threads 4
```

Resume just one stage (e.g. after fixing a bug found in `predict`):

```
python -m scripts.run_pipeline --stage predict
python -m scripts.run_pipeline --stage write
```

After `write`, check the printed `utils/validate_submission.py` result -- `PASS` means
safe to submit; a versioned copy of the two output files (+ the train report) is also
archived under `submissions/<timestamp>/` (gitignored, kept for local history only).

## Module map (`src/`)

- `config.py` -- paths, constants; every path is `AML_*`-env-var-overridable (see
  "Paths: local vs. Kaggle" above), with the D:-drive default only used on the local
  dev machine.
- `io_utils.py` -- DuckDB connection + source-view helpers (memory-safe I/O).
- `evaluate.py` -- exact reimplementation of the official macro F0.5 metric. This is
  the only source of truth for "is change X an improvement" -- see its docstring and
  `tests/test_evaluate.py` for the worked example / edge cases from the problem
  statement.
- `normalize.py` -- text normalization: `basic_clean`, `expand_abbreviations`,
  `normalize_full`, `name_core` (data-driven suffix stripping, not hardcoded --
  see PROJECT_LOG.md for the France-with-zero-training-rows result), `has_devanagari`,
  `transliterate_devanagari`. Unit tested (`tests/test_normalize.py`).
- `blocking.py` -- candidate generation (5 blocks, each self-capped before union --
  see module docstring for the OOM history that shaped this design).
  `block_and_cap_country` / `run_all_blocks` are the reusable per-country entry points
  used by both the validation report and `scripts/run_pipeline.py`.
- `features.py` -- pairwise + context features (rapidfuzz string similarity, address
  digit-overlap, per-block flags, rank/reverse-rank). `build_country_features` is the
  memory-safe entry point real callers should use (streams in chunks, then a DuckDB
  window-function pass for the context features -- see PROJECT_LOG.md for the OOM this
  replaced).
- `model.py` -- LightGBM `LGBMClassifier` under GroupKFold (leak-free OOF probabilities).
  **Import order matters on the dev machine**: `lightgbm` must be imported before
  `pandas` anywhere in the process or its native Dataset construction segfaults (see
  the module docstring and PROJECT_LOG.md) -- every entry point that trains/predicts
  imports `lightgbm` as its own first line for this reason.
- `decide.py` -- one-to-one assignment, then whichever per-entity policy (global
  threshold vs. expected-F0.5 subset selection) wins on labeled data; the
  no-ground-truth policy is exactly what real test-time inference calls too.
- `write_outputs.py` -- writes `matching_results.tsv` / `candidate_pairs.tsv`, runs
  `utils/validate_submission.py`, archives a versioned copy under `submissions/`.

## `scripts/run_pipeline.py`

The real end-to-end entry point -- see "Real end-to-end run" above for usage. Stages:
`normalize`, `train_features`, `train`, `test_features`, `predict`, `write`.

## `scripts/build_validation_split.py`

Builds a held-out validation slice from `dataset/train/` with realistic decoy density.
Went through three iterations -- postal-code bucketing (rejected, ~0.2%/7.9% coverage),
flat rarest-address-token bucketing (rejected, inflated decoy rate because most
"decoys" were really records whose true owner just wasn't in the sample), and the kept
EVAL/CONTEXT design: `is_eval=True` rows are the ~45k scored entities, `is_eval=False`
("context") rows are every S1 whose true match landed in the pool anyway, included
(with real name/address/country + ground truth) so one-to-one assignment and
reverse-rank features are trained/tested against real competition instead of an
absent owner. See the module docstring and PROJECT_LOG.md (2026-09-25, "Phase 1
addendum") for the full history, including a first version that pulled context
entities' own address-neighborhoods into the pool and had to be killed after pushing
system RAM down to ~300MB -- fixed by holding the pool fixed once context entities are
found rather than re-expanding around them.

Output goes to `$AML_WORK_DIR/parquet/val_slice/`: `source1.parquet` (includes
`is_eval`), `source2.parquet`, `source3.parquet`, `ground_truth_in_slice.parquet`
(score EVAL rows from this one), `ground_truth_full_for_slice.parquet`, `report.json`.

Always filter to `is_eval == True` before scoring with `src/evaluate.py` -- context
rows exist to be candidates' correct owners, not to be scored themselves.

## `scripts/phase2_normalization_report.py`

Measures the data-driven suffix/stop-token list per country, the Devanagari
script-mismatch rate among India true pairs, and transliteration's effect on token
overlap; also produces 15 before/after normalization examples (US/India/France).
Output: `$AML_WORK_DIR/parquet/phase2/report.json` and `suffix_token_candidates.parquet`.
See PROJECT_LOG.md (2026-09-25, "Phase 2") for the full measured results.
