"""Tests for the systematic chapter taxonomy + mirror topic map."""
import json
import os
import sys
import unittest

CORPUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(os.path.dirname(CORPUS_DIR), "jee-chapter-taxonomy", "data")
sys.path.insert(0, os.path.join(os.path.dirname(CORPUS_DIR), "jee-chapter-taxonomy"))

from build_chapters import build, extract_units  # noqa: E402


class TestChapterTaxonomy(unittest.TestCase):
    def test_53_chapters_unique_ids(self):
        with open(os.path.join(CHAPTER_DIR, "jee-chapters.json"), encoding="utf-8") as fh:
            chapters = json.load(fh)
        self.assertEqual(len(chapters), 53)
        ids = [c["chapter_id"] for c in chapters]
        self.assertEqual(len(ids), len(set(ids)))
        for c in chapters:
            self.assertTrue(c["syllabus_source_url"].startswith("https://"))
            self.assertEqual(c["level"], "chapter")

    def test_rotational_chapter_present(self):
        with open(os.path.join(CHAPTER_DIR, "jee-chapters.json"), encoding="utf-8") as fh:
            ids = {c["chapter_id"] for c in json.load(fh)}
        self.assertIn("jee-physics-rotational-motion-chapter", ids)
        self.assertIn("jee-physics-current-electricity-chapter", ids)

    def test_topic_map_covers_all_labels(self):
        with open(os.path.join(CHAPTER_DIR, "mirror_topic_map.json"), encoding="utf-8") as fh:
            mapping = json.load(fh)
        with open(os.path.join(CHAPTER_DIR, "jee-chapters.json"), encoding="utf-8") as fh:
            valid = {c["chapter_id"] for c in json.load(fh)}
        self.assertGreaterEqual(len(mapping), 62)
        for label, entry in mapping.items():
            self.assertIn(entry["chapter_id"], valid, label)

    def test_extract_units_splits_subjects(self):
        text = ("MATHEMATICS\nUNIT 1: SETS, RELATIONS AND FUNCTIONS\n"
                "PHYSICS\nUNIT 1: Rotational Motion\n"
                "CHEMISTRY\nUNIT I: SOME BASIC CONCEPTS IN CHEMISTRY\n"
                "Syllabus for JEE (Main) Paper 2A (B.Arch.)\nUNIT 1: SETS, RELATIONS AND FUNCTIONS\n")
        units = extract_units(text)
        self.assertEqual(units["Physics"], ["Rotational Motion"])
        self.assertEqual(units["Chemistry"], ["SOME BASIC CONCEPTS IN CHEMISTRY"])
        self.assertEqual(len(units["Mathematics"]), 1)  # no Paper-2A bleed


if __name__ == "__main__":
    unittest.main()
