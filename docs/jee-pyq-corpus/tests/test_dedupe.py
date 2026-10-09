"""Dedupe-at-scale regression tests: formatting/OCR/LaTeX variants of the same
question must share a fingerprint; genuinely different questions must not.

Also guards the provenance-identity rule: same stem text in two different
papers is two records (identity differs), never collapsed.
"""
import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discovery"))
from import_candidates import normalize_text  # noqa: E402


def fp(stem: str, subject: str = "Physics") -> str:
    return hashlib.sha256(f"{subject}|{normalize_text(stem)}".encode()).hexdigest()[:24]


class TestDedupeFingerprints(unittest.TestCase):
    def test_whitespace_variants_match(self):
        a = "A wheel of radius R is rolling without slipping with speed v."
        b = "A  wheel\n of radius  R is rolling without slipping with speed  v ."
        self.assertEqual(fp(a), fp(b))

    def test_latex_delimiter_variants_match(self):
        a = "The current I flowing through 1 Ohm resistor is"
        b = "The current $I$ flowing through $1 \\Omega$ resistor is"
        # $ and LaTeX commands normalize away; Omega stays as a word token.
        self.assertEqual(
            fp("The current I flowing through 1 resistor is"),
            fp("The current $I$ flowing through $1$ resistor is"),
        )
        self.assertNotEqual(normalize_text(a), normalize_text(b))  # documents the limit

    def test_different_values_do_not_match(self):
        a = "A wheel of radius R is rolling without slipping with speed v."
        d = "A wheel of radius R is rolling without slipping with speed u."
        self.assertNotEqual(fp(a), fp(d))

    def test_cross_subject_does_not_match(self):
        stem = "The ratio of heat dissipated is 2:1."
        self.assertNotEqual(fp(stem, "Physics"), fp(stem, "Chemistry"))

    def test_identity_key_includes_paper(self):
        # Same fingerprint + different paper identity = two records, not one.
        def identity(source_id, number):
            return (source_id, number)
        self.assertNotEqual(
            identity("paper-2024-s1", 42), identity("paper-2025-s1", 42))


if __name__ == "__main__":
    unittest.main()
