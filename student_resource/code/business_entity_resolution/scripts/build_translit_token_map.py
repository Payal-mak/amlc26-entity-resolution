"""Learn a transliterated-token -> Latin-token map from confirmed train true
pairs, to fix transliteration gaps that rule-based schwa deletion can't
(letter-order/substitution quirks like "iMDasTrIja" (industries) or "phuds"
(foods), not just a trailing inherent vowel). See src/normalize.py's module
docstring and PROJECT_LOG.md for why this exists.

Method: for each India true pair with a confirmed Latin<->Devanagari name
mismatch, transliterate + schwa-delete the Devanagari side into tokens, fuzzy-
align each one (rapidfuzz.ratio, MIT license) against the Latin side's tokens
from the SAME confirmed pair, and keep the best match above a threshold as one
observation. Aggregate observations across all pairs; keep a mapping only when
one Latin token clearly dominates a transliterated token's observations (this
is what makes it "learned from data" rather than a coincidence in one pair).

Usage (from code/business_entity_resolution/):
    python -m scripts.build_translit_token_map
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, io_utils, normalize  # noqa: E402
from scripts.phase2_normalization_report import measure_devanagari_mismatch  # noqa: E402

MATCH_RATIO_THRESHOLD = 65
MIN_OBSERVATIONS = 5
MIN_DOMINANT_FRACTION = 0.5


def _pre_translit_tokens(text: str) -> list:
    return [t for t in normalize.expand_abbreviations(normalize.basic_clean(text)).split(" ") if t]


def _post_fix_tokens(devanagari_text: str) -> list:
    """Transliterate + schwa-delete + clean, WITHOUT the (not-yet-built) map."""
    return [t for t in normalize.normalize_full(devanagari_text).split(" ") if t]


def align_pairs(mismatched: pd.DataFrame) -> Counter:
    """For every mismatched pair, align transliterated tokens to Latin tokens.

    Inputs: DataFrame with s1_name (Latin), matched_name (Devanagari).
    Output: Counter of (translit_token, latin_token) -> observation count.
    """
    votes = Counter()
    for _, row in mismatched.iterrows():
        latin_tokens = _pre_translit_tokens(row["s1_name"])
        translit_tokens = _post_fix_tokens(row["matched_name"])
        if not latin_tokens or not translit_tokens:
            continue
        for tt in translit_tokens:
            best_l, best_score = None, 0
            for lt in latin_tokens:
                score = fuzz.ratio(tt, lt)
                if score > best_score:
                    best_l, best_score = lt, score
            if best_l is not None and best_score >= MATCH_RATIO_THRESHOLD and tt != best_l:
                votes[(tt, best_l)] += 1
    return votes


def build_map(votes: Counter) -> dict:
    """Keep a (translit_token -> latin_token) mapping only when one target
    token dominates that transliterated token's observations.

    Inputs: Counter from align_pairs.
    Output: {translit_token: latin_token}.
    """
    by_source = defaultdict(Counter)
    for (tt, lt), n in votes.items():
        by_source[tt][lt] += n

    mapping = {}
    for tt, counter in by_source.items():
        total = sum(counter.values())
        if total < MIN_OBSERVATIONS:
            continue
        best_l, best_n = counter.most_common(1)[0]
        if best_n / total >= MIN_DOMINANT_FRACTION:
            mapping[tt] = best_l
    return mapping


def evaluate_overlap(mismatched: pd.DataFrame, translit_map: dict) -> dict:
    """Token-overlap rate at three stages: no fix, schwa-only, schwa+map.

    Inputs: mismatched pairs, the learned map.
    Output: dict of overlap counts/rates for each stage, over ALL mismatched
    pairs (not a sample -- rapidfuzz-based alignment already ran on the full
    set, so this reuses the same data for a consistent before/after story).
    """
    n = len(mismatched)
    n_before = n_schwa = n_map = 0
    for _, row in mismatched.iterrows():
        latin = set(_pre_translit_tokens(row["s1_name"]))
        if not normalize.has_devanagari(row["matched_name"]):
            continue
        raw_translit = normalize.transliterate_devanagari(row["matched_name"])
        no_fix = set(t for t in normalize.expand_abbreviations(normalize.basic_clean(raw_translit)).split(" ") if t)
        schwa_only = set(_post_fix_tokens(row["matched_name"]))
        with_map = set(normalize.normalize_full(row["matched_name"], translit_map=translit_map).split(" "))

        n_before += bool(latin & no_fix)
        n_schwa += bool(latin & schwa_only)
        n_map += bool(latin & with_map)

    return {
        "n_pairs": n,
        "overlap_no_fix": n_before, "overlap_no_fix_pct": n_before / n if n else None,
        "overlap_schwa_only": n_schwa, "overlap_schwa_only_pct": n_schwa / n if n else None,
        "overlap_schwa_plus_map": n_map, "overlap_schwa_plus_map_pct": n_map / n if n else None,
    }


def run() -> dict:
    t0 = time.time()
    con = io_utils.get_connection()
    io_utils.register_all_standard_views(con)
    mismatch_summary, mismatched = measure_devanagari_mismatch(con)
    con.close()
    print(f"loaded {len(mismatched)} mismatched pairs ({time.time()-t0:.1f}s)")

    votes = align_pairs(mismatched)
    print(f"aligned tokens: {len(votes)} (token,token) observations ({time.time()-t0:.1f}s)")

    translit_map = build_map(votes)
    print(f"learned map: {len(translit_map)} entries ({time.time()-t0:.1f}s)")

    overlap = evaluate_overlap(mismatched, translit_map)
    print(f"evaluated overlap ({time.time()-t0:.1f}s)")

    out_dir = config.PARQUET_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_df = pd.DataFrame(list(translit_map.items()), columns=["translit_token", "latin_token"])
    map_df.to_parquet(out_dir / "translit_token_map.parquet", index=False)

    report = {
        "build_time_seconds": round(time.time() - t0, 1),
        "n_mismatched_pairs": len(mismatched),
        "n_vote_observations": sum(votes.values()),
        "n_map_entries": len(translit_map),
        "overlap_before_after": overlap,
        "sample_map_entries": map_df.sort_values("translit_token").head(40).values.tolist(),
    }
    with open(out_dir / "translit_map_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
