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


class TestAddressHelpers(unittest.TestCase):
    def test_street_key_drops_numbers_and_expands_abbreviations(self):
        self.assertEqual(normalize.street_key("7800 Valburn Dr, Austin, TX"), "valburn drive austin tx")
        self.assertEqual(normalize.street_key("Jefferson City, 1205 Satinwood Drive, MO"), "jefferson city satinwood drive mo")

    def test_street_key_drops_mixed_digit_tokens(self):
        self.assertEqual(normalize.street_key("C-440 Near Block C, Sushant Lok-I"), "c near block c sushant lok i")

    def test_street_key_expands_new_abbreviations_and_handles_none(self):
        self.assertEqual(normalize.street_key("12 Las Palmas Cir"), "las palmas circle")
        self.assertEqual(normalize.street_key(None), "")

    def test_street_key_leaves_legal_abbreviations_alone(self):
        # "co"/"inc" are business words, not address words
        self.assertEqual(normalize.street_key("Main St Co"), "main street co")

    def test_house_numbers_strip_leading_zeros(self):
        self.assertEqual(normalize.house_numbers("03153 Twelve Oaks Boulevard"), ("3153",))

    def test_house_numbers_drop_the_postal_code_once(self):
        self.assertEqual(normalize.house_numbers("12 Main St 12345", "12345"), ("12",))
        self.assertEqual(normalize.house_numbers("12345 Main St 12345", "12345"), ("12345",))

    def test_house_numbers_none(self):
        self.assertEqual(normalize.house_numbers(None), ())


class TestNameDecorations(unittest.TestCase):
    strip = staticmethod(normalize.strip_name_decorations)

    def test_id_in_parentheses(self):
        self.assertEqual(self.strip("Global Harbor Rocket Llc (ID: 28974)"), "Global Harbor Rocket Llc")
        self.assertEqual(self.strip("Cafe Du Monde (No. 12)"), "Cafe Du Monde")

    def test_long_digit_run_after_dash(self):
        self.assertEqual(self.strip("Gemous Worldwide Company - 1521888650"), "Gemous Worldwide Company")
        self.assertEqual(self.strip("Peak Energy  Corporation - 4847674375"), "Peak Energy Corporation")

    def test_short_digit_runs_are_kept(self):
        self.assertEqual(self.strip("Studio 54"), "Studio 54")
        self.assertEqual(self.strip("Route 66 Diner 12345"), "Route 66 Diner 12345")  # 5 digits: not a phone/account number

    def test_domain_only_name_loses_its_ending(self):
        self.assertEqual(self.strip("jarluspmv.com"), "jarluspmv")
        self.assertEqual(self.strip("FOOTANKLEPARTNERS.COM"), "FOOTANKLEPARTNERS")
        self.assertEqual(self.strip("-- maanursing.com"), "maanursing")
        self.assertEqual(self.strip("bestservices.in"), "bestservices")  # no TLD list: keyed on the label.tld shape

    def test_www_and_generic_tld_inside_a_longer_name(self):
        self.assertEqual(self.strip("www.acme.in Ltd"), "acme.in Ltd")
        self.assertEqual(self.strip("Acme.com Inc"), "Acme Inc")

    def test_domain_shaped_token_in_a_longer_name_is_left_alone(self):
        self.assertEqual(self.strip("St.Louis Hardware"), "St.Louis Hardware")

    def test_none_and_empty(self):
        self.assertEqual(self.strip(None), "")
        self.assertEqual(self.strip(""), "")


class TestDigitLetterSwaps(unittest.TestCase):
    fix = staticmethod(normalize.fix_digit_letter_swaps)

    def test_zero_and_one_between_letters(self):
        self.assertEqual(self.fix("N0LLIE'S SECURE SALON"), "NoLLIE'S SECURE SALON")
        self.assertEqual(self.fix("C1TY Bank"), "ClTY Bank")

    def test_digits_at_word_edges_or_alone_are_untouched(self):
        self.assertEqual(self.fix("Studio 54"), "Studio 54")
        self.assertEqual(self.fix("A1 Auto 100"), "A1 Auto 100")
        self.assertEqual(self.fix("3M Co"), "3M Co")

    def test_clean_name_for_matching_combines_both(self):
        self.assertEqual(normalize.clean_name_for_matching("N0LLIE'S Salon (ID: 999)"), "NoLLIE'S Salon")


class TestHouseNumberParts(unittest.TestCase):
    parts = staticmethod(normalize.house_number_parts)

    def test_unit_box_and_floor_numbers_are_not_house_numbers(self):
        self.assertEqual(self.parts("3 Cross Timber, PMB 776, Edwards, Colorado"), (("3", ""),))
        self.assertEqual(self.parts("Unit 6, 1213 Cross Creek Circle"), (("1213", ""),))
        self.assertEqual(self.parts("1307 Kalahari Drive, Fl 3rd, Village"), (("1307", ""),))
        self.assertEqual(self.parts("PO Box 3175, Jefferson City"), ())
        self.assertEqual(self.parts("12 Main St, 4Th Floor"), (("12", ""),))

    def test_letter_and_bis_ter_suffixes(self):
        self.assertEqual(self.parts("12A Rue Victor Hugo"), (("12", "a"),))
        self.assertEqual(self.parts("12 bis Rue Victor Hugo"), (("12", "b"),))
        self.assertEqual(self.parts("12 ter Rue Victor Hugo"), (("12", "c"),))

    def test_ordinal_is_not_a_suffix(self):
        self.assertEqual(self.parts("1St Joint St, Chennai"), (("1", ""),))

    def test_raw_digits_keep_leading_zeros(self):
        self.assertEqual(self.parts("H.no 02, Bangalore"), (("02", ""),))
        self.assertEqual(normalize.house_numbers("H.no 02, Bangalore"), ("2",))

    def test_postal_code_dropped_once(self):
        self.assertEqual(self.parts("7800 Valburn Drive, Austin 78701", "78701"), (("7800", ""),))


class TestStreetAbbreviationsGeneral(unittest.TestCase):
    key = staticmethod(normalize.street_key)

    def test_general_street_words(self):
        self.assertEqual(self.key("12 Av Victor Hugo, Paris"), "avenue victor hugo paris")
        self.assertEqual(self.key("3 Bd Voltaire"), "boulevard voltaire")
        self.assertEqual(self.key("5 Imp des Roses"), "impasse des roses")
        self.assertEqual(self.key("7 Rte de Lyon"), "route de lyon")
        self.assertEqual(self.key("8 Pl de la Gare"), "place de la gare")
        self.assertEqual(self.key("100 Ave Foch"), "avenue foch")

    def test_guarded_abbreviations_expand_after_a_house_number(self):
        self.assertEqual(self.key("12 R Victor Hugo"), "rue victor hugo")
        self.assertEqual(self.key("5 Ch des Vignes"), "chemin des vignes")
        self.assertEqual(self.key("3 All des Roses"), "allee des roses")

    def test_guarded_abbreviations_left_alone_elsewhere(self):
        self.assertEqual(self.key("R K Puram, New Delhi"), "r k puram new delhi")      # initial, no number before
        self.assertEqual(self.key("All Saints Road"), "all saints road")               # ordinary word
        self.assertEqual(self.key("12 R K Puram"), "r k puram")                         # next token too short
        self.assertEqual(self.key("Plot 4 R"), "plot r")                                # nothing after it
        self.assertEqual(self.key("21 All Saints Rd"), "allee saints road")            # documents the known limit
