"""Unit tests for candidate triage (recall-first, precision-guarded)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))

from triage import triage_page_text  # noqa: E402


class TestTriage(unittest.TestCase):
    def test_rotational_stem_flagged(self):
        text = ("A wheel initially at rest is subjected to a uniform angular "
                "acceleration about its axis. The ratio theta2/theta1 is __.")
        hits = triage_page_text(text, ["angular acceleration", "torque", "I"])
        self.assertIn("angular acceleration", hits)

    def test_single_letters_never_flag(self):
        text = "Options: I. II. III. The length L and current I are given."
        hits = triage_page_text(text, ["I", "L", "torque"])
        self.assertEqual(hits, [])

    def test_non_rotational_page_clean(self):
        text = ("Heat is supplied to a diatomic gas at constant pressure. "
                "The ratio of delta Q delta U delta W is __.")
        hits = triage_page_text(
            text, ["torque", "angular momentum", "moment of inertia", "rolling"]
        )
        self.assertEqual(hits, [])

    def test_magnetism_torque_still_flagged_for_human(self):
        # Recall-first: magnetic torque mentions get flagged; the human
        # verifier excludes them by chapter discipline, not the filter.
        text = "The torque experienced by the current loop in the magnetic field is __."
        hits = triage_page_text(text, ["torque"])
        self.assertIn("torque", hits)


if __name__ == "__main__":
    unittest.main()
