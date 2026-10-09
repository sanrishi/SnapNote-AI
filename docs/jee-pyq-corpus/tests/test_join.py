"""Tests for the candidate-official join (exact/ambiguous/unmatched)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "index"))

from match_join import match_candidates  # noqa: E402


class TestJoin(unittest.TestCase):
    def _official_record(self, number, stem, source="nta-main-2026-apr05-s2"):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discovery"))
        try:
            from import_candidates import normalize_text
        finally:
            sys.path.pop(0)
        return {
            "source_id": source,
            "exam": "JEE Main",
            "year": 2026,
            "session": "April 2026",
            "shift": "Shift 2 (5 Apr 2026)",
            "subject": "Physics",
            "question_number": number,
            "normalized_text_hash": __import__("hashlib").sha256(
                f"Physics|{normalize_text(stem)}".encode()
            ).hexdigest()[:24],
            "stem_text": stem,
        }

    def _candidate(self, text, cid="c1"):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discovery"))
        try:
            from import_candidates import normalize_text
        finally:
            sys.path.pop(0)
        return {
            "candidate_id": cid,
            "subject": "Physics",
            "source_record_id": "x:0",
            "question_text_hash": __import__("hashlib").sha256(
                f"Physics|{normalize_text(text)}".encode()
            ).hexdigest()[:24],
        }

    def test_exact_match_links_source(self):
        stem = "A wheel initially at rest is subjected to a uniform angular acceleration."
        report = match_candidates(
            [self._candidate(stem)], [self._official_record(30, stem)], {}
        )
        self.assertEqual(len(report["MATCHED"]), 1)
        self.assertEqual(
            report["MATCHED"][0]["official"]["question_number"], 30
        )
        self.assertEqual(report["UNMATCHED"], [])

    def test_near_match_is_ambiguous_never_verified(self):
        stem = ("A wheel initially at rest is subjected to a uniform angular acceleration "
                "about its axis through equal angles in successive intervals.")
        report = match_candidates(
            [self._candidate(stem + " options omitted")],
            [self._official_record(30, stem)],
            {"x:0": stem + " options omitted"},
        )
        self.assertEqual(len(report["MATCHED"]), 0)
        self.assertEqual(len(report["AMBIGUOUS"]), 1)

    def test_unrelated_is_unmatched(self):
        report = match_candidates(
            [self._candidate("Heat is supplied to a diatomic gas at constant pressure.")],
            [self._official_record(30, "A wheel initially at rest.")],
            {},
        )
        self.assertEqual(len(report["UNMATCHED"]), 1)

    def test_duplicate_official_fingerprint_is_ambiguous(self):
        stem = "A wheel initially at rest."
        report = match_candidates(
            [self._candidate(stem)],
            [
                self._official_record(30, stem, source="src-a"),
                self._official_record(31, stem, source="src-b"),
            ],
            {},
        )
        self.assertEqual(len(report["AMBIGUOUS"]), 1)
        self.assertIn("2 official records", report["AMBIGUOUS"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
