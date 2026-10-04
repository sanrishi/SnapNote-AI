"""Tests for bulk-discovery importer (candidates are never verified facts)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discovery"))

from import_candidates import (  # noqa: E402
    fingerprint,
    import_row,
    normalize_text,
)


class TestImporter(unittest.TestCase):
    def test_normalize_strips_latex_noise(self):
        a = normalize_text("The torque $\\tau = r F$ is __.")
        b = normalize_text("the TORQUE  tau = r  F   is")
        self.assertEqual(a, b)

    def test_fingerprint_deterministic(self):
        self.assertEqual(fingerprint("abc", "Physics"), fingerprint("abc", "Physics"))
        self.assertNotEqual(fingerprint("abc", "Physics"), fingerprint("abc", "Chemistry"))

    def test_no_exam_facts_leak(self):
        record = import_row(
            {
                "question": "Find the moment of inertia.",
                "subject": "Physics",
                "topic": "Rigid Body Dynamics",
                "subtopic": "Moment of inertia",
                "exam": "JEE Main",
                "source_paper": "Some mock paper.docx",
                "difficulty": "Moderate",
                "correct_option": "2",
            },
            "test-dataset",
            "7",
        )
        self.assertIsNone(record["year"])
        self.assertIsNone(record["session"])
        self.assertIsNone(record["shift"])
        self.assertIsNone(record["correct_answer"])
        self.assertIsNone(record["difficulty"])
        self.assertEqual(record["license_status"], "B-discovery-only")
        self.assertEqual(record["provenance_status"], "unverified-candidate")

    def test_dedupe_collapses_identical_text(self):
        import subprocess
        import tempfile
        rows = [
            {"question": "Find the moment of inertia.", "subject": "Physics"},
            {"question": "Find  the  moment of inertia.", "subject": "Physics"},
            {"question": "Find torque.", "subject": "Physics"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.jsonl")
            out = os.path.join(tmp, "out.jsonl")
            with open(src, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            result = subprocess.run(
                [sys.executable, os.path.join(
                    os.path.dirname(__file__), "..", "discovery", "import_candidates.py"),
                 src, "--dataset", "t", "--out", out],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
