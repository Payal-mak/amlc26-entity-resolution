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
| `AML_DATA_DIR` | `--data-dir` | `dataset/` (this repo) | `/kaggle/input/datasets/<user>/<dataset-name>/dataset` |
| `AML_WORK_DIR` | `--work-dir` | `D:/amazon_ml_work` if `D:` exists, else `.work/` | `/tmp/work` |
| `AML_OUTPUT_DIR` | `--output-dir` | `output/` (this repo) | `/kaggle/working/output` |
| `AML_DUCKDB_MEMORY_LIMIT` | `--duckdb-memory` | `20GB` (Kaggle-sized default; local runs are tiny smoke tests, so the actual usage stays far under this regardless) | `20GB` (Kaggle gives ~31GB RAM) |
| `AML_DUCKDB_THREADS` | `--duckdb-threads` | `4` | `4` |
| `AML_LGBM_THREADS` | `--lgbm-threads` | `4` | `4` |
| `AML_LGBM_MAX_BIN` | `--lgbm-max-bin` | `255` (LightGBM's own default) | `255` |
| `AML_LGBM_TWO_ROUND` | `--lgbm-two-round` | `false` | `false` |
| `AML_ENABLE_TFIDF_BLOCK` | `--disable-tfidf-block` | `true` | `true` (flip off if it's too slow/memory-heavy on the first real run -- never tested at full scale locally, see PROJECT_LOG.md) |
| `AML_GEO_MAX_CAND_TOKEN_DF` | `--geo-max-cand-token-df` | `3000` | `3000` (untuned safety margin after a real full-train-scale OOM in `block_b_geo`, see PROJECT_LOG.md) |
| `AML_S1_BATCH_SIZE` | `--s1-batch-size` | `100000` | `100000` (bounds every block's join input regardless of country size) |
| `AML_DUCKDB_MAX_TEMP_DIRECTORY_SIZE` | `--duckdb-max-temp-dir-size` | `200GB` | `200GB` (a soft cap independent of real free disk space -- raise if Kaggle's `/tmp` genuinely needs more) |
| `AML_CKPT_DIR` | `--ckpt-dir` | unset (disabled) | `/kaggle/working/ckpt` (survives a Kaggle SESSION restart, unlike `--work-dir`; completed train/test batches + the trained model are mirrored here and restored from here if missing) |

`AML_WORK_DIR` holds intermediate artifacts (DuckDB database, parquet files) that are
never committed (see `.gitignore`) and never need to survive between runs; on Kaggle,
point it at `/tmp` or `/kaggle/temp` (fast local scratch, doesn't count against output
size limits), not `/kaggle/working`.

Setting any of these directly (running an individual `scripts/phaseN_*.py` rather than
`run_pipeline.py`) still works the same way:

```
set AML_WORK_DIR=E:\some\other\path      # Windows cmd
$env:AML_WORK_DIR = "E:\some\other\path" # PowerShell
export AML_WORK_DIR=/some/other/path     # bash
```

## Pipeline stages

**As of 2026-09-25: local is dev-only.** Repeated real OOMs on this 8GB machine, even
after several rounds of memory-safety fixes (see PROJECT_LOG.md), led to a hard split:

- **Local** -- `scripts/run_pipeline.py`'s individual stages, run with a small
  `--val-target-total` and/or against a tiny hand-built sample dataset, purely to check
  the code runs end to end after a change. No full-size train/test-scale run locally.
- **Kaggle** (31GB RAM, 4 CPUs) -- every real run: the full validation-slice OOF
  report, full-train model fit, full-test blocking + inference, writing both output
  TSVs. See "Kaggle: the real run" below.

### `scripts/run_pipeline.py` -- the one command, all stages

Single entry point, driven entirely by CLI flags / `AML_*` env vars (see "Paths: local
vs. Kaggle" above) -- the same code runs locally and on Kaggle. Every stage writes its
output to disk before the next starts, so `--stage <name>` resumes after a crash
without redoing earlier stages, and logs a timestamped start/end line with peak RSS
(via `psutil`) for each stage.

| Stage | What it does |
|---|---|
| `normalize` | learn suffix tokens + transliteration token map from the FULL train+test data |
| `val_slice` | build the held-out validation slice from `dataset/train/` (`--val-target-total`, default 45000) |
| `val_blocking` | block + cap candidates on that slice, report recall |
| `val_features` | build labeled features for the slice |
| `val_train` | GroupKFold LightGBM OOF training + decision layer on the slice -- **prints macro F0.5/precision/recall/singleton accuracy, overall and per country: the headline validation number** |
| `train_features` | block + build labeled features for the FULL `train_source1` vs `train_source2/3` |
| `train` | fit the model actually used for test inference (GroupKFold LightGBM on the full train set); also prints its own OOF numbers as a sanity check against `val_train` |
| `test_features` | block + build UNlabeled features for the FULL `test_source1` vs `test_source2/3` |
| `predict` | score test features with the saved models; decision layer (no ground truth) |
| `write` | write `output/matching_results.tsv` + `candidate_pairs.tsv`, run `utils/validate_submission.py`, archive a versioned copy |

`--stage all` (the default) runs every stage above in order, UNLESS `--skip-full-train`
is also set, in which case `train_features`/`train` are skipped and `predict` scores
test with `val_train`'s model instead -- a fast first submission that doesn't wait on a
full-train model fit.

### Local smoke test (dev only)

```
python -m scripts.run_pipeline --val-target-total 2000 --stage normalize
python -m scripts.run_pipeline --val-target-total 2000 --stage val_slice
python -m scripts.run_pipeline --val-target-total 2000 --stage val_blocking
python -m scripts.run_pipeline --val-target-total 2000 --stage val_features
python -m scripts.run_pipeline --val-target-total 2000 --stage val_train
```

`val_slice`/`val_blocking`/`val_features`/`val_train` need real data with some
geographic diversity to build buckets (`build_validation_split.py`'s bucket selection),
so use the real `dataset/train/` with a small `--val-target-total` rather than a fully
synthetic sample for those four. `normalize`/`train_features`/`train`/`test_features`/
`predict`/`write` have no such requirement and can run against a tiny hand-built
sample dataset (`--data-dir` pointed at a folder with the same train/test TSV schema).

### Kaggle: the real run

```
python -m scripts.run_pipeline \
    --data-dir /kaggle/input/datasets/<username>/<dataset-name>/dataset \
    --work-dir /tmp/work \
    --output-dir /kaggle/working/output \
    --ckpt-dir /kaggle/working/ckpt \
    --duckdb-memory 20GB --duckdb-threads 4
```

(`--lgbm-threads`/`--lgbm-max-bin`/`--lgbm-two-round` also exist if LightGBM ever needs
tuning on Kaggle, but the defaults are already the real values -- `n_jobs=4`,
`max_bin=255`, `two_round=False` -- not the laptop-memory compromises from earlier
today. `--geo-max-cand-token-df`/`--s1-batch-size`/`--duckdb-max-temp-dir-size` exist to
retune the block_b_geo/batching/temp-dir fixes below without a code change, if the
defaults still aren't right for a given run.)

For a fast first submission, skip the full-train stages and predict with the
validation-slice model instead (see `--skip-full-train` under "the one command" above):

```
python -m scripts.run_pipeline \
    --data-dir /kaggle/input/datasets/<username>/<dataset-name>/dataset \
    --work-dir /tmp/work \
    --output-dir /kaggle/working/output \
    --ckpt-dir /kaggle/working/ckpt \
    --duckdb-memory 20GB --duckdb-threads 4 \
    --skip-full-train --disable-tfidf-block
```

**Rough time estimate** (extrapolated from local measurements at smaller scale, not
measured at full size -- treat as an estimate, not a promise): `normalize` ~3min,
`val_slice` ~4min, `val_blocking` ~9min, `val_features` ~20min, `val_train` ~10min are
all measured-ish (scaled from the ~45k-entity validation slice, pre-recall-v2 sizing).
`train_features` and `test_features` block+featurize the FULL train/test sets (test
alone is 1.7M+ S1 rows, bigger than even India's full train set), so they dominate the
runtime and are the least certain estimate -- a first real attempt OOM'd in
`block_b_geo` after ~3500s on India train alone (see PROJECT_LOG.md); after the
DF-cap/batching/temp-dir fixes below, plausibly **1-3 hours** for `test_features` alone.
Total for `--stage all` without `--skip-full-train`: **rough order of a few hours**,
likely fitting a single Kaggle session but not by a wide margin. If it doesn't fit,
`--stage <name>` resumes from wherever it stopped, and `--ckpt-dir` means even a full
session restart doesn't lose already-completed batches (see "Memory safety at full
scale" below).

### Memory safety at full scale (recall-v2, after a real Kaggle OOM)

A first real run OOM'd in `block_b_geo` on India's full train scale (883k S1 / 4.1M
candidates -- see PROJECT_LOG.md). Three independent fixes, all in `src/config.py`/
`src/blocking.py`/`scripts/run_pipeline.py`:
- `block_b_geo` now excludes a candidate from its JOIN (not just the post-join rank
  cut) if its geo token's candidate-side document frequency exceeds
  `GEO_MAX_CAND_TOKEN_DF` -- same principle as B2's own cap, which B_geo never had.
- `build_country_pairs_and_features` batches S1 within each country
  (`S1_BATCH_SIZE`, default 100k) before calling any block, so every block's join input
  is bounded regardless of country size. Each batch writes its own
  `features_{country}_{batch}.parquet`.
- `DUCKDB_MAX_TEMP_DIRECTORY_SIZE` raised 30GB -> 200GB (the actual thing that hit
  first that night -- a soft cap independent of real free disk space).

`--ckpt-dir` (e.g. `/kaggle/working/ckpt`) mirrors each completed batch's output (and
the trained model) to a location that survives a Kaggle SESSION restart, unlike
`--work-dir` (typically `/tmp/...`, wiped between sessions). A batch already present
there is restored and skipped rather than recomputed -- to resume after a crash or
restart, just rerun the same `--stage train_features`/`test_features`/`all` command;
already-checkpointed batches log `already done (checkpoint), skipping` instead of
redoing the work.

Resume just one stage (e.g. after fixing a bug found in `predict`):

```
python -m scripts.run_pipeline --stage predict
python -m scripts.run_pipeline --stage write
```

After `write`, check the printed `utils/validate_submission.py` result -- `PASS` means
safe to submit; a versioned copy of the two output files (+ the train report) is also
archived under `submissions/<timestamp>/` (gitignored, kept for local history only).

Run unit tests:

```
python -m unittest discover -s tests -v
```

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
  see PROJECT_LOG.md for the France-with-zero-training-rows result), `has_devanagari`/
  `transliterate_devanagari` (Devanagari only) and the generalized
  `has_indic_script`/`transliterate_indic` (every Indic script the data contains --
  recall-v2, see PROJECT_LOG.md). Unit tested (`tests/test_normalize.py`).
- `blocking.py` -- candidate generation, 7 blocks (recall-v2 added B_address and
  B_tfidf on top of the original 5), each self-capped before union -- see module
  docstring for the OOM history that shaped this design. `BLOCK_NAMES` lists the
  active set (`B_tfidf` is gated behind `AML_ENABLE_TFIDF_BLOCK`, default on).
  `block_and_cap_country` / `run_all_blocks` are the reusable per-country entry points
  used by both the validation report and `scripts/run_pipeline.py`. Unit tested
  (`tests/test_blocking_recall_v2.py` for the two new blocks).
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
