"""Regression gate: the deterministic corpus audit must pass.

Covers the 2026-10-06 overnight findings: wrong NTA question IDs, undocumented
null numerical answers, mislabeled third-party provenance, shift mismatches.
"""
import os
import subprocess
import sys
import unittest

CORPUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCorpusAudit(unittest.TestCase):
    def test_audit_corpus_passes(self):
        result = subprocess.run(
            [sys.executable, os.path.join(CORPUS_DIR, "coverage", "audit_corpus.py")],
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("audit clean", result.stdout)


if __name__ == "__main__":
    unittest.main()
