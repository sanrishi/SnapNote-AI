"""Tests for source provenance classes."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from source_classes import SourceClass, classify_record, label_is_honest  # noqa: E402


class TestSourceClasses(unittest.TestCase):
    def test_official_nta_url(self):
        rec = {"source_url": "https://cdnbbsr.s3waas.gov.in/x.pdf",
               "source_document": "NTA paper", "verification_method": "read"}
        self.assertEqual(classify_record(rec), SourceClass.OFFICIAL)
        self.assertTrue(label_is_honest(rec))

    def test_third_party_transcription(self):
        rec = {"source_url": "https://mirror.example/p",
               "source_document": "x (third-party transcription of the official paper)",
               "verification_method": "read"}
        self.assertEqual(classify_record(rec), SourceClass.THIRD_PARTY_TRANSCRIPTION)
        self.assertTrue(label_is_honest(rec))

    def test_official_url_with_transcription_label_is_dishonest(self):
        rec = {"source_url": "https://cdnbbsr.s3waas.gov.in/x.pdf",
               "source_document": "x (third-party transcription)",
               "verification_method": "read"}
        self.assertFalse(label_is_honest(rec))

    def test_unknown_source_defaults_discovery_only(self):
        rec = {"source_url": "https://mirror.example/p",
               "source_document": "official NTA paper", "verification_method": "read"}
        self.assertEqual(classify_record(rec), SourceClass.DISCOVERY_ONLY)


if __name__ == "__main__":
    unittest.main()
