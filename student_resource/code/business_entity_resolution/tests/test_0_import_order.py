"""Import-order guard, not a real test.

On the Windows dev machine, importing lightgbm AFTER pandas has loaded makes
LightGBM's native code crash with an access violation (see src/model.py's
docstring and PROJECT_LOG.md). `unittest discover` loads test modules in
alphabetical order, and several of them import pandas; this file sorts first
and imports lightgbm before any of them do, so model tests can train.
"""

import lightgbm  # noqa: F401
import unittest


class TestLightgbmImportedFirst(unittest.TestCase):
    def test_import(self):
        import lightgbm as lgb
        self.assertTrue(hasattr(lgb, "LGBMClassifier"))
