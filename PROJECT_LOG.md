# Project Log — Amazon ML Challenge 2026: Business Entity Resolution

**Team:** Team 8bit (Payal Makwana, Ansh Raythatha)
**Purpose of this file:** a running, dated log of every analysis step, verified fact, decision, and change of plan. Append new entries at the bottom of the relevant section (or a new dated section) — do not delete history, strike it through or mark it superseded so the reasoning trail stays intact. This is the file to check before re-deriving something we already figured out.

---

## 2026-09-25 — Environment audit

Ran locally on Payal's machine before writing any pipeline code, to size the problem correctly before committing to an approach.

| Resource | Finding | Implication |
|---|---|---|
| RAM | 8 GB total, **only ~1.6 GB free** at check time | Very tight. Full-file pandas loads of the big S2/S3 files are risky. Close background apps before big runs. |
| GPU | NVIDIA MX450, **2 GB VRAM**; installed `torch` is the **CPU-only** build | Effectively no local GPU. Batch-embedding or fine-tuning transformers over millions of rows locally is not viable. |
| Disk | C: has only **~5 GB free**; D: has **~490 GB free** | Raw dataset alone is ~2.3 GB; intermediate parquet (candidates, features) will exceed 5 GB easily. **All working data/outputs must live on D:, not C:.** |
| CPU | 8 logical cores | Fine for LightGBM, rapidfuzz, chunked/vectorized blocking. |
| Packages present | `pandas 2.3`, `pyarrow 17`, `scikit-learn 1.5.1`, `torch 2.7.1+cpu` | TF-IDF/sparse blocking (sklearn) works out of the box. |
| Packages missing | `duckdb`, `polars`, `lightgbm`, `rapidfuzz`, `sentence-transformers` | All pip-installable, all MIT/BSD-licensed tools (not "external data"), none violate fair-play rules. |

**Decision:** given the RAM/GPU/disk ceilings, the local box is the default venue for blocking + LightGBM (both are memory-light: DuckDB streams off disk, LightGBM histogram-bins its inputs). Anything embedding- or transformer-shaped (multilingual sentence embeddings, cross-encoder fine-tuning) is **secondary/optional** and should run on a free-tier cloud notebook (Kaggle preferred: ~30 GB RAM + GPU quota; Colab as backup), not on this laptop, given the corpus sizes below. Compute infra choice is not a fair-play issue — only external *data* lookup is prohibited.

---

## 2026-09-25 — Dataset facts (verified, not assumed)

The provided strategy doc made several assumptions (30%+ singleton rate, embeddings-for-everything, etc.). Ran quick, memory-safe chunked scripts against the real `train/` files to check which assumptions actually hold before designing around them. All numbers below are measured, not guessed.

### Scale
| File | Rows |
|---|---|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

This is a **large-scale** ER problem, not a toy dataset — total raw text is ~2.3 GB across 6 files, ~30M records combined. This single fact drives most of the tooling decisions below (no naive pandas full-file joins, no full-corpus transformer embedding, prototype on a sample before scaling to full data).

### Singleton rate
**5.58%** of S1 training entities have zero matches (123,247 / 2,206,821) — much lower than the 30%+ the strategy doc speculated. Match-count distribution is roughly bell-shaped, mode at 3 matches, range 0–11:
`{0: 123247, 1: 119157, 2: 375212, 3: 530841, 4: 484115, 5: 321957, 6: 164868, 7: 63968, 8: 18680, 9: 4205, 10: 534, 11: 37}`

**Implication:** correctly predicting singletons is still worth a full 1.0 each and false merges on them still hurt, but the "cheap 30% win" the original strategy assumed isn't there — most of the score comes from getting the non-singleton match sets precisely right, not from singleton detection alone. Don't over-invest in singleton-only tuning at the expense of subset-selection quality on the 94% that do have matches.

### S1-dedup / one-to-one property — **confirmed, 0 violations**
Checked whether any S2/S3 id appears under more than one S1 in ground truth: **0 violations out of 3,693,619 S2 references and 3,944,746 S3 references.** Every matched S2/S3 record belongs to exactly one S1 entity in the training data.

**Implication:** this is a real, exploitable constraint, not just a "likely" heuristic — the decision layer should enforce one-to-one assignment (each S2/S3 candidate id goes to at most one S1, whichever it scores highest with) as a hard post-processing step. At this scale, exact optimal assignment (Hungarian algorithm) is not feasible over millions of nodes — use greedy conflict resolution sorted by descending model probability instead.

### Country-consistency of matches — **confirmed, 0 mismatches**
Checked 182,932 matched (S1, S2/S3) pairs sampled from ground truth (50,000 S1 rows worth of matches): **0 country mismatches.** Every true match pairs records with identical `country` string.

**Implication:** blocking by exact country-string equality is safe and should be the *first* partition applied, before any similarity computation — it alone cuts the candidate space roughly in half-to-a-third depending on bucket (see distribution below) at zero recall cost observed so far. Must be implemented as generic string equality (`groupby("country")`), never a hardcoded `{US, India}` set, since France must work the same way with zero train examples to tune on.

### Country distribution — France is ~15% of test, and has zero rows in train
| File | US | India | France |
|---|---|---|---|
| train_source1 | 1,323,633 (60%) | 883,188 (40%) | 0 |
| train_source2 | 3,016,817 | 2,017,799 | 0 |
| train_source3 | 3,170,056 | 2,115,547 | 0 |
| test_source1 | 663,106 (38%) | 809,986 (47%) | **259,452 (15%)** |
| test_source2 | 1,871,330 | 2,312,565 | 703,378 |
| test_source3 | 1,945,701 | 2,405,000 | 731,615 |

**Implication:** France is not an edge case to patch in later — it's ~15% of the leaderboard score, present on **both** public and private leaderboards, and there is **no labeled France data anywhere** to validate against. The only way to estimate France performance locally is the robustness proxy the strategy doc suggested: fit on one country (e.g. US), validate on the other (India), as a stand-in for the train→unseen-country gap. This must be a standing item in the validation plan, not a nice-to-have.

### Coverage / decoy rate in S2 and S3
Of 5,034,616 train_source2 rows, 3,693,619 (**73.4%**) are referenced by some ground-truth match; the remaining **26.6%** are true non-matches (decoys) with no S1 counterpart at all. Source3 is similar: 3,944,746 / 5,285,603 matched (**74.6%**), **25.4%** decoys.

**Implication:** roughly a quarter of every S2/S3 pool is genuine noise with no correct answer. Negative-pair training data is abundant and doesn't need to be synthesized — it's naturally present. Also a sanity check for blocking: a healthy blocking stage should be finding candidates for the ~74% that do match while not being fooled into false positives by the ~26% that structurally don't.

### Missing fields
`business_name` is **never blank** in any of the three sources (0.00%). `business_address` is blank in **3.36%** of source2 and **3.33%** of source3 rows (source1 has 0% blank address in the sample checked). So address-based features need a "missing address" fallback path (fall back to name+country only), but this is a minority case, not a dominant one.

### Cross-script transliteration — important, concrete finding
Found a confirmed ground-truth match where the **S1 record is in Latin script and its true S2 match is in Devanagari script** (India). Example: S1 `"Laxmi Developers Private Limited"` (Latin) truly matches an S2 record whose `business_name` is written in Devanagari. Other India records (in source3, e.g. `"International South Consultants Private Ltd"`) are already transliterated to Latin. So **India name script is mixed and inconsistent per source** — the same real business can appear in Latin in one source and native Devanagari in another.

**Implication — this is the single biggest threat to the blocking stage as originally scoped.** Character n-gram TF-IDF has **zero shared substrings** between a Latin string and a Devanagari string, so pure char-ngram/token blocking will systematically miss these pairs with no partial-credit signal to lean on. Three mitigations, not mutually exclusive:
1. Lean harder on the **address** field (digits, postal-code-like tokens, city tokens) as a script-invariant blocking signal when the name scripts disagree.
2. A lightweight **rule-based transliteration/romanization** pass for Devanagari → Latin before n-gramming (this is text normalization, not external data lookup — same category as the abbreviation expansion already planned, should be fine under fair-play rules, but flag it on the Google Form if genuinely unsure).
3. A **multilingual embedding model** (e.g. multilingual-e5-small, MIT) as a blocking signal specifically for the India (and France) buckets — these models are trained to place transliterated equivalents close in vector space even with zero shared characters. This elevates embeddings from "nice-to-have for France" (original strategy) to "load-bearing for a meaningful slice of India too."

---

## Revised strategic plan (supersedes the raw compute assumptions in the original strategy doc; problem framing and pipeline shape are unchanged and still good)

### Tooling decisions
- **Data engine: DuckDB**, not pandas, for anything touching a full S2/S3 file (joins, blocking-candidate generation, groupby aggregation). It streams off disk instead of materializing everything in RAM, which matters directly given the 1.6 GB free / 5 GB disk ceiling. Pandas is still fine for small, already-filtered slices (candidate-pair feature tables, error-analysis samples).
- **Working directory for all generated data/artifacts: D: drive.** C: cannot hold the dataset plus intermediates.
- **Blocking:** country-string equality first (verified safe, ~free recall), then inverted-index token/rare-token/postal-digit blocking, then batched (chunked, not one giant dense matmul) sparse TF-IDF char n-gram cosine top-K within each country bucket. Add a multilingual-embedding-based candidate channel specifically to cover the cross-script India cases and France, run on Kaggle/Colab GPU over the (much smaller) post-country-block set, not the full raw corpus.
- **Pairwise features:** rapidfuzz (fast, C++-backed, fine at tens of millions of pairs on 8 CPU cores) for name/address string similarity; digit/postal-token Jaccard; rank/gap context features per the original strategy (these are cheap and were correctly flagged as high-value).
- **Matching model: LightGBM** (MIT, ≤8B-param rule is moot — it's not a parameter-counted model at all, and it's the right tool: histogram-binned training keeps memory low even at tens of millions of rows, which suits this box far better than a neural classifier would). GroupKFold by S1, as planned.
- **Decision layer:** greedy one-to-one assignment (sorted by descending probability, confirmed necessary and sufficient by the 0-violation check above) + per-entity expected-F0.5 subset selection, computed on out-of-fold predictions with an exact local reimplementation of the macro F0.5 formula.
- **Optional Stage 4 (cross-encoder / LLM-assisted scoring on borderline pairs):** deprioritized relative to the original plan given local hardware. Only pursue after the Day-1 baseline is solid, on Kaggle/Colab, applied narrowly to a borderline probability band, not the full candidate set.

### Repo/notebook structure
Given the RAM ceiling, prefer **CLI-run Python scripts, one per pipeline stage**, over one long-lived Jupyter kernel — each script loads what it needs, writes parquet to D:, and exits, so memory is actually released between stages (a single notebook session on an 8 GB box accumulates dataframe references across cells and will eventually thrash). Planned stages: normalize → block → recall-ceiling report → features → train (LightGBM, OOF) → decision layer → write `matching_results.tsv` / `candidate_pairs.tsv` → validate.

Reserve a lightweight notebook (Jupyter or VS Code interactive) **only** for: initial EDA on samples, reading back small parquet slices for error analysis (false positives/negatives), and plotting recall-ceiling-vs-candidate-count / feature-importance / threshold curves. Not for running full-corpus operations.

Always **prototype the full pipeline end-to-end on a small stratified sample first** (e.g. ~20–50k S1 entities spanning US/India, with a held-out US-only/India-only split to proxy the France generalization gap) before scaling any stage to the full 2.2M/5M/5M data — at this scale, debugging a broken full run costs hours, not seconds.

### Open questions / risks to revisit
- Confirm (via the Google Form, per guidelines) whether a rule-based Devanagari→Latin transliteration step is acceptable — almost certainly yes (it's normalization of provided data, not external lookup), but worth a paper trail given how central it is to the India blocking recall.
- Country-consistency was verified on US/India only (no France ground truth exists to check it against) — treat it as a strong prior, not a certainty, for France.
- Actual local wall-clock time for country-bucketed sparse blocking at full scale (2.2M × ~5M×2) is still unmeasured — needs a timed run on the small sample first, then extrapolated, before committing to it for the full test set on submission day.

---

*(Add new dated entries below as work progresses — EDA follow-ups, blocking recall-ceiling numbers, OOF F0.5 scores, submission log, etc.)*

---

## 2026-09-25 — Phase 1: repo skeleton + validation harness

Context: Day 1, 12:30 PM IST. Public leaderboard top 3 sitting at 0.9805/0.9803/0.9789
macro F0.5 — the data is highly matchable and the fight is in the third decimal, so a
trustworthy local validation harness (this phase) matters more than usual: without it
we can't tell a real improvement from noise. 0/5 submissions used today; priority is a
valid scored submission tonight, which Phase 1 unblocks but does not itself produce.

### Environment setup
Installed `duckdb==1.5.5`, `lightgbm==4.7.0`, `rapidfuzz==3.14.6` into the existing
(shared, not project-scoped) Python 3.11.5 environment — all MIT-licensed. pip's first
attempt failed on a temp-dir error, worked after pointing `TMPDIR`/`TEMP`/`TMP` at `D:`
(consistent with the C: 5GB-free constraint — even pip's build scratch space needed
redirecting). The install upgraded `numpy` to 2.2.6, which pip flagged as incompatible
with other, unrelated packages already in this shared env (`label-studio`, `tensorflow-intel`,
`streamlit`'s protobuf pin) — irrelevant to this project but noting it in case those
tools misbehave later for an unrelated reason. Not worth a dedicated venv given the C:
disk pressure; flag if it becomes a problem.

Confirmed already-present: `pandas==2.3.0`, `pyarrow==17.0.0` (both used; both permissively
licensed — BSD-3 / Apache-2.0). `scikit-learn` and `torch` are present in the shared env
but unused by any code so far (sklearn is slated for Phase 3 TF-IDF blocking; torch isn't
planned locally at all — see the hardware-audit entry above on why embeddings run on
Kaggle/Colab instead of this machine).

### Repo skeleton
Created under `student_resource/` (this is the actual submission-zip root — `output/`
and `code/` need to be direct children of the same directory as `Documentation_template.md`,
which already lives in `student_resource/`, so building here means zipping the final
package is just `zip -r submission.zip output code Documentation_template.md` from
inside `student_resource/`, no restructuring needed later):

```
student_resource/
├── output/                                  # empty until Phase 5
└── code/business_entity_resolution/
    ├── src/
    │   ├── config.py        # paths/constants; WORK_DIR defaults to D:/amazon_ml_work
    │   ├── io_utils.py       # DuckDB connection + view helpers, memory-safe settings
    │   ├── evaluate.py       # DONE — exact official metric, see below
    │   ├── normalize.py      # Phase 2 stub (docstring = plan only)
    │   ├── blocking.py       # Phase 3 stub
    │   ├── features.py       # Phase 4 stub
    │   ├── model.py          # Phase 4 stub
    │   ├── decide.py         # Phase 4 stub
    │   └── write_outputs.py  # Phase 5 stub
    ├── scripts/
    │   └── build_validation_split.py   # DONE, see below
    ├── tests/
    │   └── test_evaluate.py  # 14 tests, all passing
    ├── README.md
    └── requirements.txt
```

`WORK_DIR` (DuckDB db file, parquet, staged submissions) defaults to `D:/amazon_ml_work`
per the hardware audit — C: has ~5GB free, nowhere near enough for intermediates.

### `evaluate.py` — exact metric, verified
Implements `f_beta_score` (per-entity, with the three official edge cases: empty/empty→1.0,
non-empty pred/empty truth→0.0, empty pred/non-empty truth→0.0), `macro_f_beta` (the
actual leaderboard metric), plus diagnostics not part of the official score: micro
precision/recall, singleton accuracy, and a per-group (e.g. per-country) breakdown.
14 unit tests pass, including the problem statement's worked example
(pred=[S2-00047,S2-00193,S3-00812], truth=[S2-00047,S3-00812]) reproducing **0.714**
exactly, and one that specifically checks the precision-weighting direction (a false
positive hurts more than a false negative, as F0.5 requires).

### `scripts/build_validation_split.py` — two iterations, second one kept

**v1 (postal code, rejected):** used a regex-extracted postal code (5-7 digit run near
the end of `business_address`) as the neighborhood key. Measured coverage before
trusting it: **only 7.9% of US and 0.2% of India** `train_source1` rows have an
extractable postal code at all — most addresses in this dataset simply don't include
one (confirmed by eye: none of the first 15 India rows sampled had a PIN code anywhere).
Running it anyway produced a badly distorted slice: singleton rate 13.1% vs the true
5.58% baseline, and 30% of ground-truth matches silently dropped because a match's
partner record lacked a postal code and so never entered any bucket. Rejected before
it could quietly poison every later measurement.

**v2 (rarest address token, kept):** for each `train_source1` row, tokenize
`business_address` (lowercase, split on non-alphanumerics, drop tokens <3 chars or
purely numeric) and take the token with the lowest document frequency in its country
(floor of 3 occurrences, to avoid keying off typos). This needs no hardcoded place-name
list — it's pure data-driven IDF, so it degrades gracefully for a country with zero
training rows (France). Measured coverage: **100.0% of India, 99.995% of US** get a geo
token this way — night and day versus postal code. Spot-checked the resulting top
tokens by bucket size: all genuine place names (Brooklyn, Houston, Washington;
Kailash/Vasant/Okhla/Rohini/Lajpat — all real Delhi-area localities), confirming the
signal is doing what it's supposed to, not picking up junk.

Picking buckets largest-first (obvious first attempt) hit the S1 target with only 51
buckets, all mega-city tokens — realistic in the sense that they're genuine
neighborhoods, but their S2/S3 candidate pools were enormous (n_s2=385k, n_s3=712k for
a 45k-entity slice) because a handful of huge cities dominated. Switched to capping any
single bucket at 2,000 S1 rows and shuffling (fixed seed) before greedy accumulation, to
get many smaller, more diverse neighborhoods instead. Final run (`--target-total 45000`,
~46s wall clock, DuckDB memory-limited to 3GB, no swapping observed):

| metric | slice | full-train baseline |
|---|---|---|
| n_s1 | 45,036 | 2,206,821 |
| country split (US/India) | 27,338 / 18,038 (~60/40) | ~60/40 (measured) |
| n_buckets | 2,484 | — |
| n_s2 / n_s3 | 470,752 / 420,860 | 5,034,616 / 5,285,603 |
| singleton rate | **0.05664** | **0.0558** |
| matches dropped at slice boundary | **0** (by construction) | — |
| S2 / S3 decoy rate | 0.841 / 0.810 | 0.266 / 0.254 |

Singleton rate and country proportions match the full-train baseline closely — those
are the two properties that stay comparable regardless of how many S1 entities you
sample. Matches dropped is exactly 0 because every true-match partner of a sliced S1 is
explicitly unioned into the S2/S3 slice regardless of whether it shares a geo token
(peeking at train ground truth to build our *own* offline dev benchmark is fine — it's
provided data, and this isn't the submission pipeline).

**The decoy-rate row does not match the baseline, and I don't think it should — flagging
this rather than quietly forcing a match.** The 26.6%/25.4% baseline is a *global*
figure: a S2/S3 record counts as "matched" there if it matches *any* of the full 2.2M
S1 entities. Our slice only contains ~45k S1 entities (a ~2% sample), so a nearby S2/S3
record whose true partner is one of the other ~98% of S1 entities — the ones not in our
sample — reads as a local decoy even though it isn't a decoy globally. Those two
percentages are answering different questions and I don't think there's a way to make a
small S1 subsample's local decoy rate match the global one without also restricting the
S2/S3 pool to only records whose true partner is in-sample — which would silently put us
back to the "only true pairs, nothing else" problem this whole exercise exists to avoid.
Net effect: this validation slice is a **harder** precision test than the global average
would suggest (more same-neighborhood non-matches per S1 than the full dataset's average
density), which errs in the safe direction for a validation harness — if anything it
should make local F0.5 a slight *underestimate* of true test performance, not an
overestimate. Worth re-checking once Phase 3 blocking exists and we can compare this
slice's candidate density to blocking's actual output density on the same S1s.

**Open item for the team, not resolved unilaterally:** accept this slice as-is and move
into Phase 2, or add a further cap on total decoys per S1 (e.g. subsample the geo-derived
S2/S3 pool down to some multiple of n_s1) to bring the pool size down further? Current
pool (~890k S2+S3 rows) is not a performance problem — the whole build takes well under
a minute — so the only reason to shrink it further would be decoy-rate cosmetics, which
the reasoning above suggests isn't the right target to chase. Proceeding on the
assumption that this is fine; revisit if OOF and LB scores diverge later (per the
standing instruction to treat that as a signal the validation slice is wrong).

**France proxy (train-on-US, validate-on-India):** needs no separate artifact — it's
this same slice's ground truth, filtered by the S1 record's own `country` column, at
model-evaluation time. Will run this for robustness once there's a model to check (Phase
4), not every iteration, per the original guidance.

### Experiments log
Empty until Phase 4 produces the first OOF predictions. Table convention going forward:

| id | date/time | change | val F0.5 | val precision | val recall | LB score |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

### Next
Stopping here per instructions, to show these numbers before continuing into Phase 2
(normalization) and Phase 3 (blocking).

---

## 2026-09-25 — Phase 1 addendum: EVAL/CONTEXT closure (rejecting the decoy-rate gap)

Pushback on the Phase 1 checkpoint above, and it was correct: the ~81-84% decoy rate
wasn't "conservative," it was a distortion. Most of those extra pool "decoys" were S2/S3
records whose true S1 owner exists in the full 2.2M-row train_source1 but simply wasn't
one of the ~45k sampled into the slice. On the real test set that owner *is* present, so
(1) reverse-rank/context features would be trained on a distribution that won't exist at
test time, (2) one-to-one assignment would go undertested, and (3) the decision
threshold would get tuned against ~4x the real decoy density, costing recall on test.
None of that shows up by just comparing aggregate rates — it took the actual mechanism
argument to see it.

### The fix, and a crash along the way
Redesigned the split as EVAL (the ~45k scored entities, unchanged construction) plus
CONTEXT: every S1 (searched across the *full* 2.2M-row train_source1, not just the
slice) whose true match already landed in the EVAL pool. Context entities are added
with their real name/address/country and ground truth so the pipeline can route
contested candidates to their rightful owner, but are excluded from scoring
(`is_eval=False`). `evaluate.py` callers must filter to `is_eval == True`.

First implementation also pulled each new context entity's own geo-token neighborhood
into the pool (reasoning: they should "participate in blocking" like a real query
entity). **This crashed the dev machine.** Round 1 alone found 252,639 context entities
against the 45k EVAL sample — nearly 6x the EVAL set size — and pulling all of their
address neighborhoods, several thousand of which have a rarest-token document frequency
in the tens or hundreds of thousands (max observed: 273,616, e.g. common city names),
pushed available system RAM from ~3GB down to **~300MB** before it was manually killed
(`Stop-Process` on the runaway `python` process; the harness's background-task timeout
had already kicked in at 300s with the job still running). This is exactly the kind of
thing the hard rule about estimating peak RAM before a full-data step exists to catch —
in this case the estimate wasn't done up front because the blowup mechanism (closure
over the *entire* 2.2M-row dataset, not just the slice) wasn't obvious until measured.
Logging it plainly rather than glossing over it.

**Root cause, confirmed by direct measurement before trusting the diagnosis:** of the
219,416 round-1 context entities found in an earlier probe, 12,561 had a rarest-token
document frequency over 2,000, summing to ~53M raw token-frequency across just those —
i.e. re-pulling their neighborhoods approached full-dataset scale.

**Fix:** don't re-expand the pool around context entities at all. A context entity's
true-match record is, by definition, already in the pool (that's the only reason it was
found) — for Phase 3 blocking to later route that record to its rightful context owner
instead of a competing EVAL entity, blocking just needs the context entity's real
address text available (it is, via `source1.parquet`) so a real blocking pass can
independently rediscover the link; no pre-computed decoy neighborhood is needed for
that. With the pool held fixed, closure completes in exactly one round by construction —
a second `owners_of()` pass over the same pool cannot find any owner the first pass
didn't already find. That second pass still runs, as a cheap, honest confirmation of
closure rather than an assumed one (see `closure_log` below: round 2 found exactly 0).

### Final measured numbers (target_total=45000, full rebuild, 3m50s wall clock)

| metric | value |
|---|---|
| n_eval (scored) | 45,052 |
| n_context (unscored, routing only) | 252,639 |
| n_s1_total | 297,691 |
| eval country split | US 27,009 / India 18,043 (~60/40, matches baseline) |
| closure | round 1: +252,639 context; round 2: +0 (confirmed closed) |
| residual orphans after closure | 0 |
| pool size (S2+S3) | 1,065,780 (n_s2=534,557, n_s3=531,223) |
| singleton rate (EVAL only) | 0.05565 vs 0.0558 baseline |
| EVAL matches dropped at slice boundary | **0** (verified directly from output parquet) |
| context matches dropped at slice boundary | 323,453 — expected and harmless: context entities are pulled in for the ONE match that landed in the pool; their other, unrelated matches elsewhere in the full dataset were never going to be recoverable here and don't need to be, since context entities aren't scored |
| **naive decoy rate** (old, EVAL-only-ownership definition) | 0.854 |
| **effective decoy rate** (pool records whose owner is outside EVAL+CONTEXT) | **0.260** |
| genuine full-train decoy rate baseline | 0.266 (S2) / 0.254 (S3) |

The effective decoy rate landing at 0.260 against a genuine baseline of ~0.26 is the
result that matters: it confirms the mechanism argument was right, not just plausible —
once context entities absorb the records that legitimately belong to them, what's left
really is close to the true global decoy rate, not an artifact of undersampling S1.

Verified the EVAL-matches-dropped=0 claim directly against the written parquet files
(not just trusted the script's own accounting) — 155,791 EVAL match-references, 0
dropped; all 323,453 dropped references belong to the 955,824 context match-references.

### Phase 3 note (blocking.py updated accordingly)
Given postal codes are present in only ~7.9% of US / ~0.2% of India addresses, demoted
the postal-code block (B3) to a minor/supplementary block in `blocking.py`'s plan and
promoted rare-address-token matching (the ~100%-coverage signal built for this script)
to the primary geographic block. Still to be measured like every other block once
Phase 3 actually runs (marginal recall contribution, not just coverage).

### Next
Continuing into Phase 2 (normalization) now, per instructions.

---

## 2026-09-25 — Phase 2: normalization

Installed `indic-transliteration==2.3.82` (MIT license, confirmed via package metadata
before use). Built `src/normalize.py` (basic_clean, expand_abbreviations, normalize_full,
name_core, has_devanagari, transliterate_devanagari — all unit tested, 13/13 passing,
`tests/test_normalize.py`) and `scripts/phase2_normalization_report.py` (the
measurement/report script; DuckDB for anything full-corpus, per the standing memory
rule). Full report: `$AML_WORK_DIR/parquet/phase2/report.json`. Total measurement run:
204s, no memory pressure observed (unlike the Phase 1 addendum incident).

### Data-driven suffix/stop tokens — the France story worked
Computed per-country token document frequency over ALL of train+test business_name
(unsupervised, provided data — allowed) and kept anything at ≥2% frequency as a
suffix/stop token for `name_core`. No hardcoded list anywhere. Result, top tokens by
country (fraction of that country's names containing the token):

- **India:** limited (50.4%), private (42.3%), india (5.9%), llp-expanded (3.8%),
  services (3.7%), company (3.1%) — exactly the Pvt/Ltd pattern the problem statement
  described.
- **US:** llc-expanded (19.2%), incorporated (14.8%), corporation (5.1%), plus some
  single-letter noise tokens (l/c/s/d — residue of initials/punctuation splitting,
  harmless to strip).
- **France (zero training rows):** sarl (21.2%), sas (14.9%), eurl (5.9%), sasu (4.3%),
  sci (3.8%) — these are real French legal-entity types (SARL/SAS/EURL/SASU/SCI) —
  **discovered automatically with no training examples at all**, exactly the
  generalization this approach exists to provide. Also picked up generic French words
  (de, du, amicale, club, ecole, comite) the same way it picks up "and"/"of" for
  English. This is the clearest validation yet that the "learn it from data" constraint
  (rather than a hardcoded suffix list) was the right call, not just a compliance
  checkbox.
- Known minor imperfection: expanding "llc" to "limited liability company" leaves
  "liability" stuck in `name_core` when only "limited" and "company" individually clear
  the frequency threshold (e.g. "Tippit Resources LLC" -> core "tippit resources
  liability"). Not fixed — Phase 4's fuzzy similarity features (rapidfuzz) should absorb
  this kind of residue; flagging rather than over-engineering suffix detection further
  before there's a model to check it against.

### Devanagari script mismatch — bigger than a spot check suggested
Measured across the full 3,059,843 India true pairs in train ground truth (not a
sample): **312,725 (10.22%)** have one side in Devanagari and the other in Latin script.
Over 1 in 10 India matches has this problem — a real, load-bearing chunk of India
matching difficulty, not an edge case.

### A bug this measurement caught in our own code
First pass of `normalize_full` produced an **empty string** for Devanagari input:
`basic_clean`'s punctuation filter only keeps `[a-z0-9\s]`, so a pure-Devanagari name
normalizes to `""` unless transliterated first. Caught by literally reading the Phase 2
example output (`राम मार्केटिंग प्राइवेट लिमिटेड` → `normalized_full: ""`), not by
inspection beforehand. This would have been actively harmful downstream: every such
record would have an identical empty `name_core`, making them all spuriously
"match" each other in B1 blocking. Fixed by having `normalize_full` transliterate
Devanagari to ASCII first, then clean/expand as normal; added a regression test
(`test_normalize_full_never_empty_for_devanagari`) so this can't silently reappear.
Lesson: running the actual function on real edge-case data, not just reasoning about
the regex, is what surfaced this — worth doing that check on every normalization
function before Phase 3 builds on top of it.

### Transliteration's actual effect on token overlap (measured, not assumed)
On the 312,725 mismatched pairs (50,000 randomly sampled for this per-string check,
since `indic_transliteration` has no SQL equivalent and has to run in Python):

| | before transliteration | after transliteration |
|---|---|---|
| pairs with ≥1 shared normalized token | 3,664 (7.3%) | 10,119 (20.2%) |

Transliteration **nearly triples** the token-overlap rate (+6,455 pairs, +12.9pp) — a
real, worthwhile gain for blocking recall. Being honest about the limit, though: even
after transliteration, **~80% of these pairs still share zero exact tokens**
(e.g. "Private Limited" transliterates phonetically to "praiveta limiteda" — close but
not string-identical to the English spelling). So transliteration alone will not close
the India cross-script gap; it needs to be paired with fuzzy string similarity
(rapidfuzz, already planned for Phase 3/4) rather than relied on as a complete fix.
Sample of newly-bridged pairs is in the full report if needed for spot-checking.

### 15 before/after examples
Saved in `report.json["examples"]` (US x5, India x4 regular + 1 Devanagari, France x5).
Confirms the France suffix-stripping works end to end with zero training data, e.g.
"Forge Groupe SARL" → core "forge"; "Maison Refuge SAS" → core "refuge". The Devanagari
example ("गुजरात प्रोड्यूसर प्रा. लि." → transliterated "gujarAta proDyUsara prA. li."
→ normalized "gujarata prodyusara pra li") shows the fixed pipeline no longer produces
an empty string, and also shows the known limit directly: "pra"/"li" (transliterated
Pvt/Ltd) don't match the India suffix set learned from the mostly-English corpus, so
`name_core` doesn't shorten further here — consistent with the 80%-still-no-overlap
finding above.

### Next
Continuing into Phase 3 (blocking) now, per instructions — will stop after it to show
recall-ceiling numbers before Phase 4.

---

## 2026-09-25 — Phase 3: blocking (three OOM crashes, a redesign, and a transliteration fix)

Context: Day 1 afternoon, 0/5 submissions used, target a validated submission by ~9 PM.
This phase took far longer than planned because the first two implementations crashed
the machine; logging the failures in full because the failure mode (unbounded joins) is
the actual lesson, not just the final numbers.

### Attempt 1: whole-slice blocking, lazy views — OOM, spilled to 43.7GB
Built 5 blocks (B1 exact-core, B2 rare-token, B3 postal-minor, B4 name-prefix, B_geo
address-token) as DuckDB views over the full validation slice (297,691 S1, 1,065,780
pool) at once. Two compounding problems, both real, both fixed before retrying blindly
a third time:
1. **B2 had no upper bound on token document frequency.** The suffix table only excludes
   tokens at ≥2% country frequency (Phase 2's name_core threshold); a token just under
   that — still shared by tens of thousands of records — has unbounded join fan-out.
   First run OOM'd mid-B2 (`43.7 GiB/43.7 GiB used`, DuckDB's temp-directory spill cap,
   which had defaulted to ~available disk space).
2. **Every block was a lazy VIEW, not a materialized TABLE**, and got referenced
   multiple times downstream (the union, each marginal-recall "without block X" query,
   the miss-analysis lookup) — each reference re-executed the full expensive join from
   scratch. Fixed by converting every block's output to `CREATE OR REPLACE TABLE`.
3. With (1) fixed at cap=500 and (2) fixed, the run got further but then B4
   (prefix-of-name_core, 4 chars) produced **251,434,018 raw pairs** — a 4-char prefix
   is far too coarse a key against a 1M+ candidate pool. The subsequent union-and-score
   step (aggregating ~301M raw rows) then OOM'd at `2.7 GiB/2.7 GiB used` (the
   memory_limit, not disk this time).

### Attempt 2: same design, B4 patched narrowly — still crashed
Widened B4's prefix to 6 chars with its own DF cap. This masked B4 specifically but the
underlying design flaw (cap the union afterward instead of each block) remained; the run
pushed system available RAM down to ~252MB before being manually killed
(`Stop-Process`). At that point the user (redirecting from patch-by-block) called for a
structural fix instead of continuing to chase individual blocks — correctly:
patching one block at a time was treating a symptom.

### Redesign (attempt 3, kept): every block caps itself, before the union
Per the user's design:
- Every block computes a score and self-caps via `QUALIFY ROW_NUMBER() OVER
  (PARTITION BY s1_id ORDER BY score DESC) <= 20` **inside its own query**, before its
  output ever reaches a union. A block can now never contribute more than 20 candidates
  per S1 no matter how common its matching key is.
- B4 (prefix) replaced by a proper **sorted-neighborhood block**: sort S1+candidates
  together by name_core within country, and pair each S1 with candidates within a
  window of 10 positions either side. Implemented via 20 `LAG`/`LEAD` window-function
  passes (each O(n)) rather than a range self-join (`ABS(rn_a - rn_b) <= 10`), which
  risks the query planner falling back to a nested-loop join at ~1.4M rows. Output is
  bounded by construction (≤20 neighbors/S1), no separate cap needed.
- B2's candidate-side token DF cap lowered to 150 (the measured p99 of the token-DF
  distribution).
- **Processed per country** (2 loops: India, US), writing each country's tagged/capped
  pairs to parquet on D: and dropping all working DuckDB tables before the next country
  — peak working set bounded to one country's data (~657k candidate rows for India, the
  larger side) rather than the full 1.07M-row pool at once.
- DuckDB connection settings made explicit: `memory_limit='3GB'`, `threads=2` (down from
  4), `preserve_insertion_order=false`, `temp_directory` on D:,
  `max_temp_directory_size='30GB'` (was previously left at DuckDB's own
  disk-space-based default).

This ran clean: **532s (8m52s), no memory crisis, no crash.**

### Transliteration bug fix (found by inspecting the actual token list, not guessed)
The Phase 2 suffix-frequency table itself contained the clue: `limiteda` (42,729
occurrences) and `praiveta` (35,525) sitting right next to `limited`/`private` as
separate tokens. Diagnosis: raw ITRANS transliteration keeps Devanagari's written
inherent vowel ("schwa") on a final consonant that spoken/loanword Hindi doesn't
pronounce there — प्राइवेट (Private) transliterates letter-for-letter to `prAiveTa`,
not `praivet`.

Two-part fix, both measured on the full India cross-script true-pair population
(not assumed):
1. **Rule-based schwa deletion** (`normalize._apply_schwa_deletion`): drop a bare
   trailing lowercase "a" after a consonant, applied on the case-preserved ITRANS output
   *before* lowercasing (case is the only signal distinguishing a droppable short schwa
   from a real long vowel). `limiTeDa` → `limiTeD` → `limited`, an exact match.
2. **Data-driven token alignment map** (`scripts/build_translit_token_map.py`): for
   confirmed India true pairs with a script mismatch, fuzzy-align (rapidfuzz, MIT
   license) each schwa-deleted transliterated token against the Latin side's tokens from
   the *same confirmed pair*; keep a mapping only when one Latin token dominates (≥5
   observations, ≥50% share). Caught what schwa deletion can't — letter-order/
   substitution drift, not just a vowel (`iMDasTrIja` → `industries`, `phuds` →
   `foods`). Learned **83 map entries** from 50,000 sampled mismatched pairs (72,967 raw
   alignment observations collapsed to 107 distinct token pairs before the dominance
   filter).

Measured token-overlap rate on the same 50,000 pairs, all four stages:

| stage | overlap rate |
|---|---|
| no transliteration at all (Phase 2 baseline) | 7.3% |
| transliterated, no schwa deletion | 19.6% |
| + schwa deletion | **83.2%** |
| + learned token map | **95.8%** |

Schwa deletion did almost all of the work; the learned map closed most of the remaining
gap. 13 new unit tests added (`tests/test_normalize.py`), all passing (31/31 total
across both test files).

**Devanagari prevalence** (asked for directly, not estimated): 0.00% of `train_source1`
India rows, **13.35%** of `train_source2` India rows (269,424 / 2,017,799), **7.47%** of
`train_source3` India rows (158,003 / 2,115,547) contain Devanagari. Confirms S1 is
Latin-only by construction (consistent with "deduplicated reference source") and the
mismatch only ever runs S1-Latin vs S2/3-Devanagari, never the reverse.

### Phase 3 final numbers (per-block-capped, per-country, cap=150)

| metric | value |
|---|---|
| recall, uncapped union | **79.98%** |
| recall, after final 50-cap | **79.07%** |
| target | 98.5% — **not met** |
| avg / p99 / max candidates per S1 (capped) | 37.5 / 50 / 50 |
| S1 with zero candidates | 0 |
| recall by block alone | sorted-neighborhood 63.6%, B1 exact-core 43.9%, B2 rare-token 37.9%, B_geo 26.5%, B3 postal 2.8% |
| marginal loss from removing a block | sorted-neighborhood −8.4pp (most load-bearing), B_geo −7.1pp, B2 −2.5pp, B1 −1.9pp, B3 postal −0.5pp (confirms the "demote to minor" call) |

**Tried raising B2's token-DF cap 150→400** (hypothesis: excluding common-but-useful
shared words like "vijay"+"ventures"/"bright"/"golden" was costing real recall).
Measured effect was negligible: uncapped recall +0.38pp (79.98%→80.36%), capped recall
essentially flat-to-worse (79.07%→79.05%). Reverted to 150 (cheaper, marginally better
on the number that actually matters). This wasn't the dominant lever after all — see
miss categorization below for what actually is. (This rerun also exposed a second,
unrelated bug: the *reporting* code held too many large DataFrames in Python memory at
once and crashed with a plain numpy allocation failure after the DuckDB blocking stage
had already completed successfully — recomputed recall directly from the saved
per-country parquet files with a streaming/per-country accumulator instead of
re-running the expensive blocking stage.)

### 30-miss read-through (categorized by actually reading the text, not counted mechanically)

| category | count seen (of ~27) | example |
|---|---|---|
| Domain-glued name (no spaces, ".com"-style) | 5 | "Prasana & Associates" vs "prasanaassociates.com" |
| Common-word DF-cap exclusion | ≥5 | "Vijay Ventures Pvt Ltd" vs "Sri Vijay Ventures Pvt [Ltd]" — shares "vijay"/"ventures", likely both too common in-country to survive the cap |
| DBA / trade-name pattern (explicitly named in the problem statement) | 2 | "Grand Automation Concepts Company" vs "Belolyra doing business as Grand Automation Concepts Company" |
| **Non-Devanagari Indic script** — Telugu, Kannada seen; our transliteration fix only covers Devanagari | 2 | "Sree Energy Private Limited" vs Telugu-script candidate; "Sky Technologies..." vs Kannada-script candidate |
| Devanagari still imperfect post-fix (95.8% overlap, not 100%) | 4 (one entity, 4 candidates) | same Devanagari string "स्मार्ट इंजीनियरिंग..." missed for 4 different candidate records |
| Heavy typo/letter-scramble | ~4 | "Biomedical" vs "Beiomtedbincal" |
| Same address, unrelated name (no address-exact block exists) | 2 | "Ginni Businesses Private Limited" vs "Pyraonyx Labs", identical address |

Two concrete, not-yet-built improvements this surfaced: (a) an exact-address block would
catch the same-address/different-name cases no name-based block ever can; (b) the
"other Indic scripts" gap (Telugu/Kannada, likely more) is the same empty-string bug
class Phase 2 fixed for Devanagari, just unaddressed for other scripts — worth
generalizing `has_devanagari`/`transliterate_devanagari` to a broader Indic-script check
if there's time later.

### Decision: move to Phase 4 now, accept 80% recall ceiling as today's baseline
Per the user's explicit time-box instruction and confirmed via a direct question rather
than assumed: **do not keep tuning blocking further today.** Reasoning executed on: the
DF-cap experiment above showed the cheapest remaining lever is exhausted; the other miss
categories (domain-glued names, DBA patterns, other scripts, heavy typos) each need a
new capability (fuzzy/substring matching, broader script detection, an address-exact
block), not a parameter tweak, so the next unit of engineering time has a much flatter
payoff curve than the sorted-neighborhood/transliteration work already banked. Revisit
blocking after a first submission exists, if time permits.

### Phase 5 infra decision (test set sizes measured, not estimated)
test_source1 = 1,732,544 (US 663,106 / India 809,986 / **France 259,452**);
test_source2 = 4,887,273; test_source3 = 5,082,316. Combined test S2+S3 = 9,969,589.
That's **5.8x this slice's S1 count and 9.4x its pool size**. Extrapolating today's 532s
per-slice blocking run roughly linearly puts full-test local blocking alone at ~1-1.5
hours, before Phase 4 feature/model work on top — on a machine that has now crashed
three times today.

**Decision (confirmed with the user, not unilateral): develop and validate the full
pipeline locally on this slice (fast, already working) through Phase 4; run the final
full-test blocking + inference (Phase 5) on Kaggle** (30GB RAM) once the pipeline logic
is proven, rather than risk a last-minute local crash near the 9 PM deadline. Revisit if
Kaggle setup itself becomes a bigger time sink than expected.

### Next
Moving into Phase 4 (features + LightGBM + decision layer) now, per the time-box
decision above.

## 2026-09-25 — Git/GitHub setup, a real Phase 4 OOM crash, and Kaggle portability

### Repo pushed to GitHub
Added `.gitignore` (excludes `dataset/`, `*.tsv`/`*.parquet`/`*.duckdb`, `output/`,
`submissions/` contents, model artifacts, `__pycache__/`, `.venv/`, OS junk) — verified
the staged diff had zero data files before the first commit. Remote pointed at a
different repo than intended; corrected to `github.com/Payal-mak/amlc26-entity-resolution`
and pushed to `main`. Going forward: commit after each completed phase.

### Phase 4 feature-build actually crashed (not just slow) — real OOM, root-caused and fixed
The backgrounded `phase4_build_features.py` run from the previous session died silently:
the harness lost track of it across a session boundary, and the log showed a plain
`numpy._core._exceptions._ArrayMemoryError` trying to allocate **51 MiB** — trivially
small — while doing `pairs["blocks"].fillna("")` over India's 6.69M-row capped-pairs
table. Measured cause: system-available RAM had dropped to ~250-500MB from Chrome/VS
Code memory pressure (unrelated to this job's own growth, confirmed by checking with
nothing of mine running) — at that level, even a modest allocation fails.

**Fix (structural, not a bigger machine):** rewrote the country-scale feature build to
never hold more than one 100k-row batch of pairs in pandas at a time:
1. `features.build_pair_features_base` split out from the old `build_pair_features` —
   computes only pairwise (non-context) features, safe to call chunk-by-chunk.
2. `features.build_base_features_streaming` streams a capped-pairs parquet via
   `pyarrow.parquet.ParquetFile.iter_batches`, writing each batch straight to a temp
   "base features" parquet (never materializes the whole country's pairs table).
3. Context features (`rank_by_s1`, `gap_to_best_by_s1`, `reverse_rank`) **cannot** be
   computed per-chunk — they need every row for a given `s1_id`/`cand_id` at once, so
   chunking them would silently corrupt them (the exact bug class already fixed once
   for the val-slice EVAL/CONTEXT design). Moved to a second pass,
   `features.add_context_features_via_duckdb`, as DuckDB window functions
   (`ROW_NUMBER() OVER (PARTITION BY ...)`) over the on-disk base file — DuckDB spills
   past its memory cap instead of holding it all in RAM, same pattern as Phase 3's
   blocking.
4. `features.build_country_features` wraps both steps as the one real entry point.

Smoke-tested the whole path (streaming write + DuckDB context pass) on synthetic data
before relaunching for real, with `AML_DUCKDB_MEMORY_LIMIT=1GB` given the memory crisis.

**Result — Phase 4 feature build completed cleanly:**

| country | pairs | positives | time |
|---|---|---|---|
| India | 6,694,557 | 378,841 | 1413s base + 44s context |
| US | 4,868,930 | 248,901 | 471s + 20s |

Total 33m13s. Both `features_India.parquet` / `features_US.parquet` now exist; training
(`phase4_train_and_decide.py`) is next.

### Kaggle portability + `scripts/run_pipeline.py` (requested mid-session, ahead of the actual Phase 5 run)
User's framing: "next runs may happen on Kaggle" — so before running the real
train/test pipeline anywhere, made every path configurable and built the single
end-to-end entry point Phase 5 was always going to need, generalized rather than
val-slice-specific.

- `src/config.py`: `DATA_DIR`/`OUTPUT_DIR` now read `AML_DATA_DIR`/`AML_OUTPUT_DIR` (previously
  hardcoded to repo-relative paths); `WORK_DIR`'s `D:/amazon_ml_work` default is now
  guarded by `Path("D:/").exists()`, falling back to a repo-relative `.work/` dir
  everywhere else (Kaggle is normally Linux — no `D:` drive at all). `AML_WORK_DIR`/
  `AML_DUCKDB_MEMORY_LIMIT`/`AML_DUCKDB_THREADS` already existed from Phase 3's crash
  fixes and needed no change. `SUBMISSIONS_DIR` moved from under `WORK_DIR` to
  `REPO_ROOT/submissions` (repo-relative, gitignored) so versioned-copy history isn't
  lost if `WORK_DIR` gets wiped between runs.
- `src/blocking.py`: promoted `register_suffix_table`/`run_all_blocks` (formerly
  duplicated inside `scripts/phase3_blocking_report.py`) to first-class module
  functions, plus a new `block_and_cap_country` one-call wrapper — both the validation
  report and the new real pipeline now call the same code.
- `src/write_outputs.py`: implemented for real (was a TODO stub) — TSV writer for both
  output files (one row per **required** id, passed explicitly by the caller so a
  pipeline bug that drops an S1 entity becomes a validator error, not a silently short
  file), a `run_validator` wrapper around `utils/validate_submission.py`, and
  `save_versioned_copy` into `submissions/<timestamp>/`.
- `scripts/run_pipeline.py` (new): `normalize -> train_features -> train ->
  test_features -> predict -> write`, driven by `--data-dir`/`--work-dir`/
  `--output-dir`/`--duckdb-memory`/`--duckdb-threads`/`--stage` flags that just set
  `AML_*` env vars before `src.config` is ever imported. `--stage <name>` re-runs one
  stage only, for resuming after a crash. Per-country blocking+feature loop is shared
  with the Phase 4 validation build via `features.build_country_features`.
- End-to-end smoke-tested against hand-built synthetic train/test TSVs (6 train S1s
  incl. one true singleton, 4 test S1s incl. a France-only row with zero training
  presence) through every stage, `--stage all`, including the validator (`PASS`) and the
  versioned-copy archive.

### A second real bug the smoke test caught: LightGBM segfaults if `pandas` is imported first
Training crashed even on the tiny synthetic data with `OSError: exception: access
violation reading 0x0000000000000000` inside LightGBM's native `set_label` call — not a
data problem (confirmed by bisection: contiguous/correctly-typed arrays, no NaN/Inf,
crashes even on pure-random X with our real y and vice versa). Root cause, isolated by
changing only import order in an otherwise identical script: **on this dev machine,
`import lightgbm` after `import pandas` has already loaded its native extensions
reliably segfaults on any data at all; importing `lightgbm` first fixes it completely**
— a DLL/native-runtime conflict specific to this Windows environment, not a code bug.
Fixed by adding `import lightgbm` as the literal first import (before `pandas`/`sys.path`
setup/anything from `src`) in every entry point that trains or predicts:
`scripts/phase4_train_and_decide.py`, `scripts/run_pipeline.py`, and `src/model.py`
itself (defensive, in case it's ever imported first by something else). Also hardened
`src/model.py`'s `train_oof`/`predict_with_fold_models` to force
`np.ascontiguousarray(..., dtype=np.float64)` on `X`/`y` regardless — didn't fix this
particular crash on its own, but is cheap insurance against a real, separate class of
bug (`.values` on a multi-dtype-block DataFrame can hand back a non-contiguous array).
**This would have hit the real Phase 4 training run too** — caught before spending the
real 33-minute feature build's output on a crash.

Added `lightgbm==4.7.0` (MIT) and `scikit-learn==1.5.1` (BSD-3-Clause) to
`requirements.txt` with their licenses noted.

### Next
Run `scripts.phase4_train_and_decide` on the real completed features (India + US) and
report val F0.5/precision/recall/singleton accuracy, overall and per country.

## 2026-09-25 — Two more real local OOMs, then the decision: Kaggle for everything real

### Two more genuine crashes training on the real India+US features (11.56M rows)
1. `df[feature_cols].values` promoted to **float64** before an `np.ascontiguousarray(...,
   dtype=np.float32)` cast could ever apply -- pandas' block-manager interleave picks
   the array's dtype from the mixed-dtype columns (float32 features + int64 rank
   columns) at `.values` time, ignoring what you cast it to afterward. Needed 2.15GiB
   and failed. Fixed the *pattern* (not just the one call site) by switching to
   `df[cols].to_numpy(dtype=np.float32)`, which threads the target dtype into the
   initial allocation (~1.08GiB instead).
2. Past that, LightGBM's own native Dataset construction (`__init_from_np2d` /
   `construct`) raised `bad allocation` -- a crash inside the library's internal
   histogram-binning memory, unrelated to X/y's own size. Traced to genuinely holding
   too much alive at once: the full mixed-dtype features dataframe (with string id/
   country columns) was staying alive in memory for the entire GroupKFold loop, on top
   of X/y and each fold's training slice. Split `model.train_oof` into
   `build_xy(df, ...)` (extract contiguous float32 X/y/groups) + `train_oof(X, y,
   groups, ...)` (no longer takes a dataframe at all) specifically so callers can `del`
   the original dataframe before training starts. Also added `max_bin=63` +
   `two_round=True` to LightGBM's params as an additional laptop-memory compromise.

### User call: stop patching local memory, move all real runs to Kaggle
After the second crash mid-training, explicit direction: Kaggle is set up and verified
(31GB RAM, 4 CPUs, repo cloned, dependencies installed). New split, going forward:
- **Local** = dev-only. Small smoke tests (e.g. ~2-20k pairs) to check code runs end to
  end after a change. No full-size training or inference locally anymore.
- **Kaggle** = every real run: feature build, training, test blocking + inference,
  writing both output TSVs.

Killed the running local training process (`Stop-Process`) rather than let it keep
fighting for memory. Reverted the laptop-only compromises now that Kaggle is the real
target: LightGBM `max_bin` back to the library default (255, was 63), `n_jobs` back to
a real thread count (4, was 1), `two_round` off (was `True`). All three (plus DuckDB's
memory/thread settings) are now `config.py` values read from `AML_LGBM_*`/
`AML_DUCKDB_*` env vars -- configurable, with Kaggle-appropriate values as the defaults
everywhere, not just on Kaggle.

### `scripts/run_pipeline.py` finished as the one real end-to-end command
Extended from 6 stages to 10, adding the validation-slice OOF report as an explicit
stage sequence rather than a separate workflow: `normalize -> val_slice -> val_blocking
-> val_features -> val_train -> train_features -> train -> test_features -> predict ->
write`. `val_train` (thin wrapper around the already-tested
`phase4_train_and_decide.run()`) is the headline number -- macro F0.5/precision/recall/
singleton accuracy, overall and per country, printed in a clearly-delimited block
(`_print_metrics_report`) so it's easy to spot in a long Kaggle log; `train` (fits the
model actually used for test inference, on the FULL train set) prints the same
breakdown as a sanity check against `val_train`. Extended
`evaluate.per_group_macro_f_beta` to also return per-group precision/recall (was
f_beta-only) so both stages get the full per-country breakdown from one already-
tested, single-source-of-truth function. Every stage now logs a timestamped start/end
line with peak RSS (polled via `psutil` on a background thread every 2s -- there's no
one cross-platform "peak memory" syscall that works identically on Windows and
Kaggle's Linux).

### Smoke-tested every stage (per the new local=dev-only rule)
- `normalize`/`train_features`/`train`/`test_features`/`predict`/`write`: against a
  tiny hand-built synthetic train/test dataset (6 train S1s incl. one true singleton,
  4 test S1s incl. a France-only row with zero training presence) -- all six passed,
  validator `PASS`.
- `val_slice`/`val_blocking`/`val_features`/`val_train`: these need real geographic
  diversity for `build_validation_split.py`'s bucket selection to find anything (it
  errored with an empty-bucket-list exception on the synthetic data), so smoke-tested
  against the REAL local `dataset/train/` instead, with `--val-target-total 2000` (not
  the real 45000) and a throwaway `--work-dir` so the real 45,036-entity slice built
  earlier today wasn't overwritten. **First real numeric Phase 4 result** (small
  sample, 2,032 eval entities, so a noisier read than the full run will give, but a
  genuine end-to-end number): macro F0.5 **0.9295**, micro precision 0.9830 / recall
  0.8516, singleton accuracy 0.9649; by country -- India F0.5 0.9077 (precision 0.9833,
  recall 0.8079), US F0.5 0.9440 (precision 0.9827, recall 0.8804). Policy chosen:
  `global_threshold(t=0.75)`.

### Also: removed challenge PDFs/image from git tracking
`guidelines.pdf`, the problem-statement PDF, and `image.png` were committed earlier
today (before the "never push challenge data" rule was interpreted narrowly as just
`dataset/`/`*.tsv`). Ran `git rm --cached` on all three and added `*.pdf`/`*.png` to
`.gitignore` -- they still exist locally, just untracked now.

### Kaggle time estimate (extrapolated, not measured at full scale -- flagged as such to the user)
`normalize`/`val_slice`/`val_blocking`/`val_features`/`val_train` are all measured-ish
(scaled from today's real ~45k-entity slice runs): rough order of 3+4+9+20+10 ≈ 45
minutes combined. `train_features` and `test_features` block+featurize the FULL train/
test sets -- test alone measured at 5.8x the validation slice's S1 count -- so they
dominate and are the least certain part of the estimate: plausibly 1-3 hours *each*.
Total for `--stage all`: rough order of a few hours, likely fitting one Kaggle session
but not by a wide margin. Told the user this explicitly rather than a false-precise
number, and that `--stage <name>` resumes if a session times out mid-run.

### Next
Hand off to Kaggle for the real run. Once it completes: read back the val_train OOF
report (the real one, at 45k scale) and the final matching_results.tsv/
candidate_pairs.tsv, run the validator, and get the first real submission out.

## 2026-09-25 — recall-v2 branch: closing the miss_analysis.py gaps

Kaggle run started (from the README command). While it runs, on a separate `recall-v2`
branch (no changes to anything the running job's files depend on): built the four fixes
`miss_analysis.py`'s 100-pair stratified miss sample pointed at, measured each on the
real validation slice before moving to the next, per explicit instruction.

### miss_analysis.py: committed, with its counting bug fixed
`compute_final_cap_miss_count` was counting every pair lost to the final cross-block cap,
not just TRUE-match pairs among them (should mirror `compute_miss_universe`'s anti-join
restricted to `eval_truth`). Fixed; recomputed value against the on-disk tagged/capped
parquet is 2,039 (351 India + 1,688 US) -- not the ~1,419 the earlier session's
percentage-based estimate implied, which turned out to be arithmetic on a stale
`report.json` vs. the current on-disk tagged/capped parquet (a rerun since then shifted
both by the ~0.003% cross-run tie-breaking noise already flagged, at these totals enough
to move the count meaningfully). Not chased further; the corrected query itself is right.

### Fix #1: B2's document-frequency cap (`B2_MAX_CAND_TOKEN_DF`)
Phase 3's own docstring already recorded that raising this 150->400 in isolation was a
wash (`recall_capped` flat-to-worse). Reproduced that exactly on this branch
(`scripts/recall_v2_b2_df_sweep.py`: 150->0.79069, 400->0.79045) -- confirms the earlier
finding, but it turned out not to generalize to bigger relaxation. At 2000 (still with
`CANDIDATES_PER_S1_CAP` unchanged at 50): `recall_capped` 0.8068, a real +1.6pp the 400
test alone couldn't see, because the newly-found B2 pairs were mostly getting crowded
out at the final cross-block cut, not failing to be found at all -- confirmed directly
with `scripts/recall_v2_cap_interaction.py`, sweeping `CANDIDATES_PER_S1_CAP` at
B2_MAX_CAND_TOKEN_DF=2000: cap=50 -> 0.8068, cap=75 -> 0.8226, cap=100/150/250 -> 0.8226
(no further gain -- 75 already covers essentially every S1 that has that many real
candidates). Adopted: `B2_MAX_CAND_TOKEN_DF=2000`, `CANDIDATES_PER_S1_CAP=75`. Cost:
avg candidates/S1 37.5 -> 43.1 (+15%), i.e. ~15% more feature-build/training compute on
Kaggle for +3.19pp capped recall. Didn't pin down the true optimum between 400 and 2000,
or above 2000 -- this 8GB machine hit real join-fanout OOMs sweeping that range (first
attempt: `AML_DUCKDB_MEMORY_LIMIT` defaulted to config.py's Kaggle-sized 20GB even
though this is the 8GB laptop, since the sweep script never set the local override
before importing `config` -- fixed by setting `AML_DUCKDB_MEMORY_LIMIT=3GB` first, same
env-var-before-import rule as everywhere else; second attempt still OOM'd past
df=400 -- genuine local memory ceiling, not a bug). 2000 is `miss_analysis.py`'s own
tested relaxed value, not a fitted optimum; worth a tighter Kaggle-side sweep later.

### Fix #2: char-3-gram TF-IDF block (`block_b_tfidf_char_ngram`, new)
Cosine similarity over character 3-grams, within country, top-K (15) per S1, `max_df`
(fraction) capping vocabulary the same way `B2_MAX_CAND_TOKEN_DF` caps B2's fan-out.
Targets exactly what token-equality blocks structurally cannot reach: domain-glued names
("butlerhall.com" as one token), leading-#/hashtag-style names, word-order scrambles,
heavy typos. Chose a batched scipy sparse matmul (fit one `TfidfVectorizer` per country,
transform candidates once, transform+multiply S1 in fixed-size batches) over the
`sparse_dot_topn` package, to avoid a new dependency of uncertain Kaggle availability.
Unit tested (`tests/test_blocking_recall_v2.py`) for correctness on tiny synthetic data.
**Per explicit instruction, never run at full country scale locally** (a sparse matmul
over char-trigrams is a fundamentally different memory/CPU profile than the indexed SQL
joins every other block uses -- at full scale, 185k x 658k for India, this is untested
outside Kaggle). Small-subset test only (1,500 sampled S1 + a 15-20k candidate
background per country, `scripts/recall_v2_tfidf_subset_test.py`): standalone recall on
the subset **77% India / 94% US** at ~14 candidates/S1 -- stronger alone than any
original block's standalone recall in the Phase 3 report (best was
`b_sorted_neighborhood` at 63.6%). Wired into `src/blocking.py`'s `BLOCK_NAMES`/
`run_all_blocks`, gated behind `config.ENABLE_TFIDF_BLOCK` (env `AML_ENABLE_TFIDF_BLOCK`,
CLI `--disable-tfidf-block` on `run_pipeline.py`, default ON) specifically so it can be
turned off with no code change if the real Kaggle run finds it too slow/memory-heavy.

### Fix #3: transliteration generalized to every Indic script, not just Devanagari
`miss_analysis.py`'s 100-pair sample found 10/49 structural misses were non-Devanagari
Indic scripts -- same failure mode Devanagari had before transliteration existed
(`basic_clean`'s a-z0-9 filter silently produces an EMPTY name_core). Generalized
`has_devanagari`/`transliterate_devanagari` to `has_indic_script`/`transliterate_indic`
(covers Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Oriya, Tamil, Telugu, in
addition to Devanagari, via `indic_transliteration`'s other schemes, same schwa-deletion
pass); `normalize_full` now dispatches through the generic version. Devanagari-specific
functions and the learned `TRANSLIT_TOKEN_MAP` (Phase 2) are unchanged -- still
Devanagari-only, since that map was learned specifically from Devanagari<->Latin
confirmed pairs. Measured how big this actually is (not just the 10/49 miss-sample
count, which is a small tail): counted script per record across the full real train
data -- **source2 alone has ~195k non-Devanagari-Indic-script records** (Telugu 39,323;
Kannada 37,211; Tamil 33,781; Gujarati 30,929; Bengali 30,723; Malayalam 18,773; Oriya
7,493; Gurmukhi 6,688), comparable in scale to Devanagari's 269,424 -- these were
previously ALWAYS block-invisible on the name side, in every country/block, not a rare
edge case. One coverage gap found while unit-testing: `indic_transliteration`'s
Tamil->ITRANS scheme doesn't map the alveolar 'ன' character -- harmless in practice,
since `basic_clean`'s a-z0-9 filter strips whatever's left, same as any other stray
character; `normalize_full` still never returns empty (test coverage:
`tests/test_normalize.py::TestOtherIndicScripts`, 8 new tests).

### Fix #4: address-only block (`block_b_address_rare_tokens`, new)
Shared rare address tokens (score = sum of 1/df, DF-capped both ends like B2) plus a
house-number-match bonus (first digit run in the address, `HOUSE_NUMBER_EXPR`), within
country. Always on (SQL/indexed-join based, same cost profile as B1/B2/B3/B_geo, so
measured at the real full validation-slice scale locally, no subset caveat needed).
Turned out much bigger than "a niche DBA-only fix": for TRUE match pairs in general,
addresses tend to agree even when normalized names don't (same business scraped from
multiple sources), so this block is really a stronger generalization of B_geo's
single-rarest-token signal (which alone already gave 26.5% standalone recall in the
Phase 3 report) rather than a rare-case patch. Measured
(`scripts/recall_v2_address_block_eval.py`, held the other 5 blocks fixed from the
existing tagged parquet, added B_address's own pairs on top, cap=75 per fix #1):
baseline recall_capped 0.8036 -> 0.9028 with B_address added, +15,476 net-new true pairs
recovered that no other block found at all. Cost: avg candidates/S1 39.6 -> 48.1.

### Final combined local measurement (5 original blocks + B_address, B2/cap at the new
defaults, transliteration fix live in every normalize_full call -- **B_tfidf excluded**,
per fix #2's explicit "never full-scale locally" instruction)
Reran the real `scripts/phase3_blocking_report.py` unmodified, with `src/blocking.py`'s
updated defaults and `AML_ENABLE_TFIDF_BLOCK=false`, against the same real validation
slice the original 0.7907 baseline came from (original `tagged_*.parquet`/`report.json`
backed up to `parquet/phase3/recall_v2_baseline_backup/` first).

First attempt crashed (`pyarrow.lib.ArrowMemoryError` writing `tagged_India.parquet` --
the bigger candidate counts at the new cap finally overflowed this 8GB machine's
remaining ~2.6GB free at the pandas->arrow conversion step, unrelated to DuckDB's own
`memory_limit`). Rewrote as `scripts/recall_v2_final_check.py`: same real block calls,
but computes recall directly from each country's in-memory result (no parquet write, no
cross-country accumulation of raw pairs -- just running counters), one country fully
processed and freed before the next. That version completed:

| | recall_uncapped | recall_capped | avg candidates/S1 | n_capped_pairs |
|---|---|---|---|---|
| **Original baseline** (5 blocks, DF=150, cap=50) | 0.7998 | **0.7907** | 37.5 | 11,153,420 |
| **recall-v2, minus B_tfidf** (6 blocks, DF=2000, cap=75) | 0.9121 | **0.9117** | 51.6 (p99=73) | 15,372,379 |

By country (recall_capped): India 0.7477 -> **0.8752**, US 0.8197 -> **0.9363**. A real
**+12.1pp** overall, from B2's relaxed DF cap + the raised final cap (fix #1) and
B_address (fix #4) together -- bigger than either measured in isolation (0.8226 and
0.9028 respectively against a cap=75-only baseline), since the two fixes' newly-found
pairs aren't fully overlapping. Cost: n_capped_pairs +38% (11.15M -> 15.37M) -- real
extra feature-build/training compute on the next Kaggle run, not free.

### What's left / next Kaggle run
- 0.9117 is a FLOOR, not the ceiling: B_tfidf (fix #2) is not included in it at all (never
  run at full scale outside Kaggle), and its subset-alone recall (77% India / 94% US) was
  higher than any other single block's standalone number and targets failure modes (
  domain-glued names, hashtag-style, word-order scrambles, typos) none of the other 6
  blocks reach at all -- the real combined number on Kaggle should be meaningfully higher
  still, plausibly within reach of the 97% target rather than clearly short of it (the
  read going into this branch, before B_address's real size was known).
- `--disable-tfidf-block` exists as an immediate fallback if B_tfidf's full-scale cost is
  too high on the first real run; nothing else in this branch depends on it being on, and
  the 0.9117 floor holds either way.
- Recommended next Kaggle command: the same `scripts/run_pipeline.py --stage all` command
  as before (README), now picking up recall-v2's `src/blocking.py`/`src/normalize.py`
  changes automatically (no pipeline-script changes were needed for the blocking side) --
  just merge/rebase `recall-v2` onto whatever branch Kaggle pulls from first. Worth
  watching the `val_blocking` stage's timing/memory specifically on the first run, since
  B_tfidf's full-country-scale cost is genuinely untested before that point; fall back to
  `--disable-tfidf-block` if it stalls rather than losing the whole run.
- The +38% candidate-count growth (11.15M -> 15.37M pairs, before B_tfidf adds more on
  top) is real extra `train_features`/`test_features`/`train` compute -- the README's
  existing "a few hours, likely fitting one Kaggle session but not by a wide margin"
  estimate should now be read as tighter, not looser; `--stage <name>` resume is the
  safety net if a session times out mid-run.
