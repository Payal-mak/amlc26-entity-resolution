"""Unit tests for src.normalize.

Run with: python -m unittest tests.test_normalize -v
(from code/business_entity_resolution/)
"""

import unittest

from src import normalize


class TestBasicClean(unittest.TestCase):
    def test_lowercase_and_whitespace_collapse(self):
        self.assertEqual(normalize.basic_clean("  Smart   FINANCIAL Networks "), "smart financial networks")

    def test_ampersand_becomes_and(self):
        self.assertEqual(normalize.basic_clean("Smith & Sons"), "smith and sons")

    def test_accents_stripped(self):
        self.assertEqual(normalize.basic_clean("Café Francais"), "cafe francais")

    def test_punctuation_removed(self):
        self.assertEqual(normalize.basic_clean("A.B.C., Inc."), "a b c inc")

    def test_none_and_empty(self):
        self.assertEqual(normalize.basic_clean(None), "")
        self.assertEqual(normalize.basic_clean(""), "")


class TestExpandAbbreviations(unittest.TestCase):
    def test_whole_token_only(self):
        # "st" expands, but "street" (which merely contains no "st" token) must not change
        self.assertEqual(normalize.expand_abbreviations("st street"), "street street")

    def test_does_not_touch_substrings(self):
        # "coast" contains "st" as a substring but must not be affected
        self.assertEqual(normalize.expand_abbreviations("coast rd"), "coast road")


class TestNameCore(unittest.TestCase):
    def test_strips_suffix_tokens(self):
        full = normalize.normalize_full("Anand Builders Private Limited")
        core = normalize.name_core(full, {"private", "limited"})
        self.assertEqual(core, "anand builders")

    def test_falls_back_to_full_if_everything_is_suffix(self):
        full = "private limited"
        core = normalize.name_core(full, {"private", "limited"})
        self.assertEqual(core, "private limited")  # fallback, never empty


class TestDevanagari(unittest.TestCase):
    def test_has_devanagari_true(self):
        self.assertTrue(normalize.has_devanagari("राम मार्केटिंग प्राइवेट लिमिटेड"))

    def test_has_devanagari_false(self):
        self.assertFalse(normalize.has_devanagari("Ram Marketing Private Limited"))

    def test_transliterate_is_ascii(self):
        out = normalize.transliterate_devanagari("राम मार्केटिंग")
        self.assertTrue(all(ord(c) < 128 for c in out))
        self.assertTrue(len(out) > 0)

    def test_normalize_full_never_empty_for_devanagari(self):
        # Regression test: normalize_full used to silently produce "" for
        # Devanagari text because basic_clean's a-z0-9 filter strips non-Latin
        # scripts entirely. Caught during Phase 2 measurement (PROJECT_LOG.md).
        full = normalize.normalize_full("राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertNotEqual(full, "")
        self.assertTrue(all(ord(c) < 128 for c in full))

    def test_schwa_deletion_exact_match_on_limited(self):
        # "लिमिटेड" (Limited) letter-for-letter transliterates to "limiTeDa"
        # (trailing inherent vowel Hindi doesn't pronounce there); schwa
        # deletion should recover the exact English token.
        full = normalize.normalize_full("लिमिटेड")
        self.assertEqual(full, "limited")

    def test_schwa_deletion_does_not_touch_long_vowel_ending(self):
        # A token ending in 'aa' (long vowel) or a vowel immediately before
        # the final 'a' must be left alone -- only a bare schwa after a
        # consonant gets dropped.
        self.assertEqual(normalize._apply_schwa_deletion("maa"), "maa")
        self.assertEqual(normalize._apply_schwa_deletion("kaa"), "kaa")

    def test_schwa_deletion_drops_trailing_vowel_after_consonant(self):
        self.assertEqual(normalize._apply_schwa_deletion("limiTeDa"), "limiTeD")

    def test_translit_map_applied_after_schwa_deletion(self):
        full = normalize.normalize_full("राम मार्केटिंग", translit_map={"marketimg": "marketing"})
        self.assertIn("marketing", full.split())


class TestOtherIndicScripts(unittest.TestCase):
    """recall-v2: has_indic_script/transliterate_indic generalize the
    Devanagari-only bridge to every script indic_transliteration supports.
    """

    def test_detect_tamil(self):
        self.assertEqual(normalize.detect_indic_script("தமிழ்"), "TAMIL")

    def test_detect_gujarati(self):
        self.assertEqual(normalize.detect_indic_script("ગુજરાતી"), "GUJARATI")

    def test_detect_devanagari_still_works(self):
        self.assertEqual(normalize.detect_indic_script("राम मार्केटिंग"), "DEVANAGARI")

    def test_detect_latin_is_none(self):
        self.assertIsNone(normalize.detect_indic_script("Ram Marketing Private Limited"))

    def test_has_indic_script_true_for_non_devanagari(self):
        self.assertTrue(normalize.has_indic_script("தமிழ் நிறுவனம்"))

    def test_transliterate_indic_tamil_nonempty(self):
        # Not asserting pure-ASCII here: indic_transliteration's Tamil->ITRANS
        # coverage has a gap (observed: alveolar 'ன' passes through
        # untransliterated) -- harmless for the real pipeline since
        # normalize_full's basic_clean step strips whatever's left (see
        # test_normalize_full_never_empty_for_tamil below), same as any other
        # non a-z0-9 character.
        out = normalize.transliterate_indic("தமிழ் நிறுவனம்")
        self.assertTrue(len(out) > 0)

    def test_transliterate_indic_passthrough_for_latin(self):
        self.assertEqual(normalize.transliterate_indic("Ram Marketing"), "Ram Marketing")

    def test_normalize_full_never_empty_for_tamil(self):
        # Same regression class as the Devanagari test above, generalized:
        # basic_clean's a-z0-9 filter would silently produce "" for any
        # untransliterated Indic script.
        full = normalize.normalize_full("தமிழ் நிறுவனம்")
        self.assertNotEqual(full, "")
        self.assertTrue(all(ord(c) < 128 for c in full))


if __name__ == "__main__":
    unittest.main()
