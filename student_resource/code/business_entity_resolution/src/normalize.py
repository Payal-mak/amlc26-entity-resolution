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


# Address-side abbreviations: the generic street words from ABBREVIATIONS
# (never the legal-entity ones like "inc"/"co", which mean something else in an
# address) plus a few more street/place words. Language-level, not
# country-specific -- both sides of every comparison go through the same map,
# so an ambiguous token ("st", "ct") only has to be consistent, not correct.
ADDRESS_ABBREVIATIONS = {
    **{k: v for k, v in ABBREVIATIONS.items() if k in {"rd", "st", "ave", "blvd", "dr", "ln", "ct", "pl", "apt", "ste", "hwy", "pkwy"}},
    "cir": "circle", "sq": "square", "ter": "terrace", "trl": "trail", "expy": "expressway",
    "rte": "route", "bldg": "building", "nr": "near", "opp": "opposite",
}


def expand_address_abbreviations(text: str) -> str:
    """Whole-token expansion of ADDRESS_ABBREVIATIONS on a basic_clean'd address."""
    return " ".join(ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in text.split(" ") if tok)


def street_key(address: str) -> str:
    """Address with every number removed and abbreviations expanded.

    Input: raw business_address (or None). Output: lowercase text made of the
    address's non-numeric tokens only (any token containing a digit is dropped:
    house/unit/plot numbers, PIN/ZIP codes, "c-440" style codes), e.g.
    "7800 Valburn Dr, Austin, TX" -> "valburn drive austin tx". Used to compare
    the street/locality part of two addresses when their house numbers are
    truncated, mistyped or missing.
    """
    tokens = [t for t in basic_clean(address).split(" ") if t and not any(c.isdigit() for c in t)]
    return expand_address_abbreviations(" ".join(tokens))


def house_numbers(address: str, postal_code: str = None) -> tuple:
    """All digit runs in an address (leading zeros stripped), postal code excluded.

    Input: raw address, and optionally its already-extracted postal code (one
    run equal to it is dropped, so two different houses in the same ZIP/PIN
    don't look like an "equal number"). Output: tuple of digit strings, in
    order of appearance ("03153 Twelve Oaks" -> ("3153",)).
    """
    runs = re.findall(r"\d+", address or "")
    if postal_code and postal_code in runs:
        runs.remove(postal_code)
    return tuple(r.lstrip("0") or "0" for r in runs)


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
    if has_devanagari(text or ""):
        text = _apply_schwa_deletion(transliterate_devanagari(text))
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
