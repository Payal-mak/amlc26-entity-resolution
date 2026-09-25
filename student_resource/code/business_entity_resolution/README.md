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

### Working directory for large files

All intermediate artifacts (DuckDB database, parquet files, staged submissions) are
written under `AML_WORK_DIR` (default `D:/amazon_ml_work`), **not** under this repo.
This is a deliberate choice for the dev machine: the raw dataset plus intermediates
exceed what's comfortably available on the repo's own drive. Override with an
environment variable if reproducing elsewhere:

```
set AML_WORK_DIR=E:\some\other\path      # Windows cmd
$env:AML_WORK_DIR = "E:\some\other\path" # PowerShell
export AML_WORK_DIR=/some/other/path     # bash
```

## Pipeline stages

Run as modules from this directory (`code/business_entity_resolution/`), one process
per stage -- each stage loads only what it needs and exits, which keeps peak memory
predictable on an 8GB dev machine (DuckDB, used for anything touching a full source
file, streams off disk rather than materializing everything in RAM).

| # | Command | Status |
|---|---|---|
| 1 | `python -m scripts.build_validation_split --target-total 45000` | done |
| 2 | `python -m scripts.phase2_normalization_report` then `python -m scripts.build_translit_token_map` | done |
| 3 | `python -m scripts.phase3_blocking_report` | done (recall ceiling 79.98% uncapped / 79.07% capped -- see PROJECT_LOG.md) |
| 4 | features + LightGBM train (OOF) + decision layer | TODO |
| 5 | full test inference (Kaggle, not local -- see PROJECT_LOG.md) -> `output/matching_results.tsv` / `candidate_pairs.tsv` | TODO |

Run unit tests:

```
python -m unittest tests.test_evaluate tests.test_normalize -v
```

## Module map (`src/`)

- `config.py` -- paths, constants, the D:-drive working-directory convention.
- `io_utils.py` -- DuckDB connection + source-view helpers (memory-safe I/O).
- `evaluate.py` -- exact reimplementation of the official macro F0.5 metric. This is
  the only source of truth for "is change X an improvement" -- see its docstring and
  `tests/test_evaluate.py` for the worked example / edge cases from the problem
  statement.
- `normalize.py` -- text normalization: `basic_clean`, `expand_abbreviations`,
  `normalize_full`, `name_core` (data-driven suffix stripping, not hardcoded --
  see PROJECT_LOG.md for the France-with-zero-training-rows result), `has_devanagari`,
  `transliterate_devanagari`. Unit tested (`tests/test_normalize.py`).
- `blocking.py`, `features.py`, `model.py`, `decide.py`, `write_outputs.py` -- Phase
  3-5 stubs; each file's docstring is that stage's plan.

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
