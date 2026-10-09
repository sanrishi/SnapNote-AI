"""Tests for the JEEnify ingestion path (third-party transcription mirror).

Guards (locked from the 2023/january-24-shift-1 pilot):
- chunk parsing extracts qnum / kind / chapter / mirror answer / figure flag;
- subject identity is POSITION-based so dropped-question gaps (Q41/Q78 in
  the pilot) cannot shift a subject boundary;
- the public summary committed to Git never carries copyrighted stem text.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "index"))

from parse_jeenify import parse_paper, subject_split, subjects_by_position  # noqa: E402
from build_jeenify_index import parse_slug, public_summary, session_for  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "jeenify_page.html")


class TestJeenifyParser(unittest.TestCase):
    def test_parse_chunks(self):
        with open(FIXTURE, encoding="utf-8") as fh:
            html = fh.read()
        recs = parse_paper(html)
        self.assertEqual([r["qnum"] for r in recs], [1, 2, 4, 5])
        self.assertEqual(recs[0]["qtype"], "Single correct")
        self.assertEqual(recs[1]["qtype"], "Numerical")
        self.assertEqual(recs[0]["chapter"], "Kinematics")
        self.assertEqual(recs[0]["answer_raw"], "Option 2")
        self.assertEqual(recs[1]["answer_raw"], "5")
        self.assertTrue(recs[0]["stem"])
        self.assertEqual(recs[2]["n_images"], 1)
        self.assertEqual(recs[0]["n_images"], 0)

    def test_subject_split_position_based_despite_gap(self):
        # Fixture skips Q3 (dropped-question gap); positions still map
        # Physics, Physics, Chemistry, Mathematics.
        with open(FIXTURE, encoding="utf-8") as fh:
            html = fh.read()
        recs = parse_paper(html)
        split = subject_split(html, len(recs))
        self.assertEqual(split, [("Physics", 2), ("Chemistry", 1), ("Mathematics", 1)])
        subjects = subjects_by_position(split, len(recs))
        self.assertEqual(subjects, ["Physics", "Physics", "Chemistry", "Mathematics"])

    def test_slug_and_session(self):
        year, mon, day, shift = parse_slug("2023/january-24-shift-1")
        self.assertEqual((year, mon, day, shift), (2023, "January", 24, "1"))
        self.assertEqual(session_for(2023, "January"), "January 2023")
        self.assertEqual(session_for(2024, "February"), "January 2024")
        with self.assertRaises(ValueError):
            parse_slug("not-a-slug")

    def test_public_summary_carries_no_stem_text(self):
        rows = [
            {"year": 2023, "subject": "Physics", "source_id": "s1",
             "stem_text": "secret stem must never commit"},
            {"year": 2023, "subject": "Chemistry", "source_id": "s1",
             "stem_text": "another secret"},
        ]
        blob = json.dumps(public_summary(rows))
        self.assertNotIn("stem_text", blob)
        self.assertNotIn("secret", blob)
        summary = public_summary(rows)
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["by_subject"]["Physics"], 1)


if __name__ == "__main__":
    unittest.main()
