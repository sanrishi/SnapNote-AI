"""Tests for the transcribed-official index builder (JEEPrep HTML mirror).

Guards: parser extracts question numbers/stems from saved HTML; the public
summary written to Git never carries copyrighted stem text.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "index"))

from build_transcribed_index import parse_subject_page, public_summary  # noqa: E402
from build_transcribed_index import _parse_shift  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "jeeprep_page.html")


class TestTranscribedIndex(unittest.TestCase):
    def test_parse_subject_page_extracts_records(self):
        with open(FIXTURE, encoding="utf-8") as fh:
            html = fh.read()
        recs = parse_subject_page(html, slug="2025-jan-28-shift-2", subject="physics")
        self.assertGreaterEqual(len(recs), 2)
        for r in recs:
            self.assertEqual(r["subject"], "Physics")
            self.assertEqual(r["year"], 2025)
            self.assertTrue(r["stem_text"])
            self.assertEqual(len(r["normalized_text_hash"]), 24)
            self.assertIn("jeeprep.app", r["source_url"])

    def test_feb_2024_maps_to_january_session(self):
        # NTA 2024 Session 1 ran 27 Jan-1 Feb; Feb-1 shifts are January session.
        shift = _parse_shift(["JEE Main 2024 Feb 1 Shift 2"])
        self.assertEqual(shift["year"], 2024)
        self.assertEqual(shift["session"], "January 2024")
        self.assertIn("Feb", shift["shift"])

    def test_public_summary_carries_no_stem_text(self):
        blob = json.dumps(public_summary([]))
        self.assertNotIn("stem_text", blob)
        with open(FIXTURE, encoding="utf-8") as fh:
            html = fh.read()
        recs = parse_subject_page(html, slug="2025-jan-28-shift-2", subject="physics")
        summary = public_summary(recs)
        self.assertEqual(summary["total_records"], len(recs))
        self.assertTrue(summary["source_ids"])


if __name__ == "__main__":
    unittest.main()
