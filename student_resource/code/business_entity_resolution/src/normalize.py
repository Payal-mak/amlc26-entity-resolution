"""Text normalization for business_name / business_address.

Two layers, deliberately kept separate:
  - `basic_clean` / `normalize_full` / `expand_abbreviations`: pure string
    transforms, safe to hardcode (lowercasing, accent stripping, generic
    English abbreviation expansion like "rd"->"road"). This is language
    normalization, not business-identity data, so a small fixed map is fine
    here -- unlike the suffix/stop-token list below.
  - The suffix/stop-token list used to derive `name_core` from `name_full`
    MUST be learned from data (document frequency over train+test names,
    per country), not hardcoded -- France has zero training rows, so any
    fixed list of legal suffixes would silently do nothing for it. See
    `build_name_token_df` / `suffix_tokens` below, and PROJECT_LOG.md
    (2026-09-25, Phase 2) for the measured suffix tokens and thresholds.

Devanagari handling: PROJECT_LOG.md's Phase 1 EDA found a confirmed
Latin<->Devanagari true match in India data. `has_devanagari` /
`transliterate_devanagari` (via `indic_transliteration`, MIT-licensed) exist
to bridge that -- see scripts/phase2_normalization_report.py for the measured
mismatch rate and the before/after token-overlap improvement.

Other Indic scripts (recall-v2, 2026-09-25): miss_analysis.py's 100-pair
stratified miss sample found 10 of 49 structural (fully-relaxed-blocking)
misses were non-Devanagari Indic scripts -- Gujarati/Tamil/Bengali/etc.
records that `has_devanagari`'s Devanagari-only regex let straight through
`basic_clean`'s a-z0-9 filter as an EMPTY name_core, same failure mode
Devanagari itself had before transliteration was added. `has_indic_script` /
`transliterate_indic` generalize the same bridge to every script
`indic_transliteration` supports (Bengali, Gujarati, Gurmukhi, Oriya, Tamil,
Telugu, Kannada, Malayalam, in addition to Devanagari), reusing the same
schwa-deletion pass. `has_devanagari`/`transliterate_devanagari` are kept
as-is (Devanagari-only) since scripts/phase2_normalization_report.py and
scripts/build_translit_token_map.py's learned TRANSLIT_TOKEN_MAP are
specifically about the Devanagari<->Latin mismatch measurement; only
`normalize_full`'s dispatch was switched to the generic version.

Schwa deletion + learned token map (added after the first transliteration
measurement): raw ITRANS output keeps the Devanagari script's inherent final
vowel that spoken/loanword Hindi drops -- "प्राइवेट" (Private) transliterates
letter-for-letter to "prAiveTa", not "praivet", because the script itself
carries a trailing "a" that Hindi speakers don't pronounce there. This showed
up directly in the Phase 2 suffix-frequency measurement: "limiteda" (42,729
occurrences) and "praiveta" (35,525) sitting right next to "limited"/"private"
as separate, non-matching tokens. `_apply_schwa_deletion` is a cheap rule-based
first pass (drop a bare trailing "a"); `TRANSLIT_TOKEN_MAP` (built by
scripts/build_translit_token_map.py from confirmed train true-pairs, not
hand-written) catches the harder cases schwa deletion alone can't, like
"iMDasTrIja" -> "industries" (a written-j-for-z substitution, not a vowel
issue). See PROJECT_LOG.md for the measured before/after token-overlap rates.
"""

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_VOWELS = set("aeiouAEIOU")

# Unicode block -> indic_transliteration sanscript scheme name, checked in
# this order (each block is disjoint, so order doesn't affect detection --
# kept alphabetical by scheme name for readability). Covers every major
# Indic script that appears in the challenge data (see PROJECT_LOG.md,
# recall-v2: counts per script measured over the real train data).
_INDIC_SCRIPT_RANGES = [
    ("BENGALI", re.compile(r"[ঀ-৿]")),
    ("DEVANAGARI", _DEVANAGARI_RE),
    ("GUJARATI", re.compile(r"[઀-૿]")),
    ("GURMUKHI", re.compile(r"[਀-੿]")),
    ("KANNADA", re.compile(r"[ಀ-೿]")),
    ("MALAYALAM", re.compile(r"[ഀ-ൿ]")),
    ("ORIYA", re.compile(r"[଀-୿]")),
    ("TAMIL", re.compile(r"[஀-௿]")),
    ("TELUGU", re.compile(r"[ఀ-౿]")),
]

# Generic, language-level abbreviation expansion (legal-entity and street
# words). Whole-token replacement only (never substring), so this can't
# mangle a token that merely contains one of these as a substring.
ABBREVIATIONS = {
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
    "ltd": "limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "pvt": "private",
    "assn": "association",
    "bros": "brothers",
    "mfg": "manufacturing",
    "intl": "international",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "apt": "apartment",
    "ste": "suite",
    "hwy": "highway",
    "pkwy": "parkway",
}


def strip_accents(text: str) -> str:
    """Unicode NFKD-decompose `text` and drop combining marks (accents).

    Input: any string. Output: the same string with accents removed
    (e.g. "café" -> "cafe"); non-Latin scripts (Devanagari, etc.) pass
    through unchanged since they have no Latin combining-mark decomposition.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def basic_clean(text: str) -> str:
    """Lowercase, strip accents, '&'->'and', drop punctuation, collapse whitespace.

    Input: raw business_name or business_address (or None). Output: a
    lowercase string of space-separated alphanumeric tokens (script-preserving
    -- Devanagari characters are untouched, only Latin accents are stripped).
    """
    if not text:
        return ""
    text = strip_accents(text.lower())
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def expand_abbreviations(text: str) -> str:
    """Expand common name/address abbreviations, whole-token only.

    Input: an already-basic_clean'd string. Output: the same string with any
    token found in ABBREVIATIONS replaced by its expansion.
    """
    return " ".join(ABBREVIATIONS.get(tok, tok) for tok in text.split(" ") if tok)


def normalize_full(text: str, translit_map: dict = None) -> str:
    """Transliterate (if needed) + basic_clean + abbreviation expansion.

    Input: raw business_name or business_address; optional translit_map
    ({transliterated_token: latin_token}, from
    scripts/build_translit_token_map.py) applied as a final per-token
    substitution pass, after schwa deletion, lowercasing and abbreviation
    expansion. Output: normalized string, used as-is for full-text similarity
    features and as the input to `name_core`.

    Devanagari text is transliterated to ASCII first, then schwa-deleted (see
    module docstring). This matters more than it looks: `basic_clean`'s
    punctuation regex only keeps a-z0-9, so without transliteration a
    Devanagari name normalizes to an EMPTY string -- found by running this
    exact function on a real Devanagari record during Phase 2 measurement
    (see PROJECT_LOG.md), not caught by inspection beforehand. An empty
    name_full/name_core would be actively harmful for blocking (every such
    record would spuriously token-match every other one).
    """
    if has_indic_script(text or ""):
        text = _apply_schwa_deletion(transliterate_indic(text))
    cleaned = expand_abbreviations(basic_clean(text))
    if translit_map:
        cleaned = " ".join(translit_map.get(tok, tok) for tok in cleaned.split(" ") if tok)
    return cleaned


def _apply_schwa_deletion(itrans_text: str) -> str:
    """Drop a bare trailing inherent-vowel 'a' from each ITRANS token.

    Input: raw (case-preserved) ITRANS output. Output: same, with a single
    trailing lowercase 'a' dropped per token where it looks like an unwritten
    Hindi schwa rather than a real long vowel -- must run before lowercasing,
    since ITRANS distinguishes short 'a' (droppable) from long 'A'/'aa'
    (a real vowel, must be kept) only by case. Never drops down to nothing.
    """
    out = []
    for tok in itrans_text.split(" "):
        if len(tok) > 2 and tok.endswith("a") and not tok.endswith("aa") and tok[-2] not in _VOWELS:
            tok = tok[:-1]
        out.append(tok)
    return " ".join(out)


def name_core(normalized_full_name: str, suffix_set: set) -> str:
    """Drop data-driven suffix/stop tokens from an already-normalized name.

    Input: output of `normalize_full` on a business_name, and a per-record
    suffix_set (typically `suffix_tokens_for_country(country)` -- see
    scripts/phase2_normalization_report.py for how that set is learned).
    Output: the 'core' name with high-frequency tokens removed, falling back
    to the full name if that would strip every token (an all-suffix name,
    e.g. a shell company named just "Private Limited", still needs SOME
    signal for blocking).
    """
    tokens = [t for t in normalized_full_name.split(" ") if t and t not in suffix_set]
    return " ".join(tokens) if tokens else normalized_full_name


def has_devanagari(text: str) -> bool:
    """Whether `text` contains any Devanagari-script character.

    Input: raw or normalized string. Output: bool.
    """
    return bool(_DEVANAGARI_RE.search(text or ""))


def transliterate_devanagari(text: str) -> str:
    """Romanize Devanagari text to ASCII (ITRANS scheme) for token bridging.

    Input: a string that may contain Devanagari characters (non-Devanagari
    characters pass through unchanged). Output: ASCII romanization, e.g.
    "राम मार्केटिंग" -> "rAma mArkeTiMga". Uses `indic_transliteration`
    (MIT-licensed) -- imported lazily so modules that don't need
    transliteration don't pay its import cost.
    """
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit

    return _translit(text, sanscript.DEVANAGARI, sanscript.ITRANS)


def detect_indic_script(text: str) -> str:
    """Which Indic script (if any) `text` contains, as a sanscript scheme
    name ("DEVANAGARI", "TAMIL", ...), or None if it's plain Latin/other.

    Input: raw or normalized string. Output: scheme name string or None.
    Checks every script in _INDIC_SCRIPT_RANGES; a mixed-script string
    (rare -- not observed in this data) returns whichever scheme's block
    appears first in that list, which is fine since normalize_full only
    needs "transliterate this" to fire, not a precise script census.
    """
    for scheme, pattern in _INDIC_SCRIPT_RANGES:
        if pattern.search(text or ""):
            return scheme
    return None


def has_indic_script(text: str) -> bool:
    """Whether `text` contains any supported Indic-script character (not
    just Devanagari -- see module docstring). Input: raw or normalized
    string. Output: bool.
    """
    return detect_indic_script(text) is not None


def transliterate_indic(text: str) -> str:
    """Romanize `text` to ASCII (ITRANS scheme) from whichever Indic script
    it's actually written in, generalizing `transliterate_devanagari` to
    every script `indic_transliteration` supports.

    Input: a string that may contain any supported Indic script (non-Indic
    characters pass through unchanged). Output: ASCII romanization. Falls
    back to returning `text` unchanged if no supported script is detected
    (mirrors `transliterate_devanagari`'s contract of "safe to call on
    already-Latin text", used by scripts/build_translit_token_map.py-style
    code that may not have checked first).
    """
    scheme_name = detect_indic_script(text)
    if scheme_name is None:
        return text

    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit

    return _translit(text, getattr(sanscript, scheme_name), sanscript.ITRANS)
