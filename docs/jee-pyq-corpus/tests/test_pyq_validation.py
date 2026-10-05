"""Validation tests for the verified JEE PYQ corpus (#31).

Guards: schema conformance, taxonomy cross-references, duplicate detection,
registry consistency, and reproducible counts/coverage. Uses only the
standard library so the corpus can be validated without backend dependencies.
"""

import json
import os
import re
import sys
import unittest

CORPUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CORPUS_DIR, "data")
SCHEMA_PATH = os.path.join(CORPUS_DIR, "schema", "pyq-record.schema.json")
TAXONOMY_PATH = os.path.join(
    os.path.dirname(CORPUS_DIR), "jee-concept-taxonomy", "data",
    "jee-physics-rotational-motion-taxonomy.json",
)
CHAPTER_TAXONOMY_PATH = os.path.join(
    os.path.dirname(CORPUS_DIR), "jee-chapter-taxonomy", "data",
    "jee-chapters.json",
)

REQUIRED_FIELDS = [
    "question_id", "exam", "year", "subject", "chapter", "concept_ids",
    "question_type", "paper_question_number", "nta_question_id",
    "source_url", "source_document", "verification_status", "verification_method",
]


def _load_records():
    records = []
    for path in sorted(os.listdir(DATA_DIR)):
        if path.endswith("-pyq.json"):
            with open(os.path.join(DATA_DIR, path), encoding="utf-8") as fh:
                records.extend(json.load(fh))
    return records


def _taxonomy_ids():
    with open(TAXONOMY_PATH, encoding="utf-8") as fh:
        ids = {c["concept_id"] for c in json.load(fh)}
    if os.path.exists(CHAPTER_TAXONOMY_PATH):
        with open(CHAPTER_TAXONOMY_PATH, encoding="utf-8") as fh:
            ids |= {c["chapter_id"] for c in json.load(fh)}
    return ids


class TestPyqCorpus(unittest.TestCase):
    def test_schema_file_is_valid_json(self):
        with open(SCHEMA_PATH, encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertEqual(schema["title"], "Verified JEE PYQ metadata record")

    def test_required_fields_present(self):
        for record in _load_records():
            for field in REQUIRED_FIELDS:
                self.assertIn(field, record, f"{record.get('question_id')} missing {field}")

    def test_no_unverifiable_exam_facts(self):
        """Records must not contain invented statistics (frequency, weightage)."""
        banned = {"frequency", "weightage", "appeared", "difficulty_stats"}
        for record in _load_records():
            overlap = banned & set(record.keys())
            self.assertEqual(overlap, set(), f"{record['question_id']}: {overlap}")

    def test_concept_ids_exist_in_taxonomy(self):
        valid = _taxonomy_ids()
        for record in _load_records():
            for concept_id in record["concept_ids"]:
                self.assertIn(concept_id, valid, f"{record['question_id']}: unknown {concept_id}")

    def test_no_duplicate_question_ids(self):
        ids = [r["question_id"] for r in _load_records()]
        self.assertEqual(len(ids), len(set(ids)), "duplicate question_id")

    def test_no_duplicate_paper_slots(self):
        slots = [
            (r["exam"], r["year"], r.get("session"), r.get("shift"), r["paper_question_number"])
            for r in _load_records()
        ]
        self.assertEqual(len(slots), len(set(slots)), "duplicate paper slot")

    def test_question_id_format(self):
        pattern = re.compile(r"^pyq-jee-(main|advanced)-[0-9]{4}-[a-z0-9-]+$")
        for record in _load_records():
            self.assertRegex(record["question_id"], pattern)

    def test_marks_have_basis(self):
        for record in _load_records():
            if record.get("marks") is not None:
                self.assertTrue(record.get("marks_basis"), f"{record['question_id']}: marks without basis")

    def test_counts_reproducible(self):
        import sys
        sys.path.insert(0, os.path.join(CORPUS_DIR, "queries"))
        try:
            from pyq_queries import count_by_concept, get_by_concept
        finally:
            sys.path.remove(os.path.join(CORPUS_DIR, "queries"))
        concept = "jee-physics-rotational-motion-concept-005"
        first = count_by_concept(concept)
        second = len(get_by_concept(concept))
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 1)

    def _paper_url(self, url):
        """Registry keys on paper-level URLs; records may point at the exact
        subject page (paper URL + /physics|chemistry|mathematics)."""
        import re
        return re.sub(r"/(physics|chemistry|mathematics)$", "", url or "")

    def test_registry_consistency(self):
        """Every record's source_url must exist in the source registry."""
        with open(os.path.join(CORPUS_DIR, "sources", "registry.json"), encoding="utf-8") as fh:
            registry = json.load(fh)["sources"]
        known_urls = {s["source_url"] for s in registry}
        for record in _load_records():
            self.assertIn(self._paper_url(record["source_url"]), known_urls, record["question_id"])

    def test_new_nullable_fields_allowed(self):
        for record in _load_records():
            self.assertIn("difficulty", record)
            self.assertIsNone(record["difficulty"], "difficulty must stay unverified")
            self.assertIn("content_fingerprint", record)
            self.assertTrue(record["content_fingerprint"])

    def test_shift_retrieval_and_explain(self):
        import sys
        sys.path.insert(0, os.path.join(CORPUS_DIR, "queries"))
        try:
            from pyq_queries import explain_match, get_by_shift
        finally:
            sys.path.remove(os.path.join(CORPUS_DIR, "queries"))
        recs = get_by_shift("JEE Main", 2026, "April 2026", "Shift 1 (2 Apr 2026)")
        self.assertEqual(len(recs), 2)
        reasons = explain_match("jee-physics-rotational-motion-concept-005", recs[0])
        self.assertTrue(any("concept =" in r for r in reasons))
        self.assertEqual(explain_match("jee-physics-rotational-motion-concept-009", recs[0]), [])
    def test_coverage_matrix_recomputes(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(CORPUS_DIR, "coverage", "build_matrix.py")],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(os.path.join(CORPUS_DIR, "coverage", "matrix.json"), encoding="utf-8") as fh:
            matrix = json.load(fh)
        self.assertEqual(matrix["totals"]["verified_records"], len(_load_records()))
        self.assertNotIn("complete", {c["status"] for c in matrix["cells"]})

    def _registry_by_url(self):
        with open(os.path.join(CORPUS_DIR, "sources", "registry.json"), encoding="utf-8") as fh:
            registry = json.load(fh)["sources"]
        return registry

    def test_no_duplicate_source_ids(self):
        ids = [s["source_id"] for s in self._registry_by_url()]
        self.assertEqual(len(ids), len(set(ids)), "duplicate source_id in registry")

    def test_record_hash_matches_registry_source(self):
        """Filename-independent identity: record provenance hash must equal
        the registry source hash for the record's source_url."""
        by_url = {s["source_url"]: s for s in self._registry_by_url()}
        for record in _load_records():
            source = by_url.get(self._paper_url(record["source_url"]))
            self.assertIsNotNone(source, f"{record['question_id']}: source not in registry")
            self.assertEqual(
                record["provenance"]["pdf_sha256"], source["pdf_sha256"],
                f"{record['question_id']}: hash mismatch — wrong source attribution",
            )
            self.assertEqual(
                record["shift"], source["shift"],
                f"{record['question_id']}: shift mismatch with registry source",
            )

    def test_q30_q31_belong_to_apr05_s2(self):
        """Regression: Q30/Q31 (wheel, rolling) were once misattributed to
        5 Apr Shift 1. They belong to 5 Apr Shift 2 by NTA ID sequence."""
        by_id = {r["question_id"]: r for r in _load_records()}
        for qid in (
            "pyq-jee-main-2026-apr05-s2-phy-030",
            "pyq-jee-main-2026-apr05-s2-phy-031",
        ):
            record = by_id[qid]
            self.assertEqual(record["shift"], "Shift 2 (5 Apr 2026)")
            self.assertIn("20260409829414602.pdf", record["source_url"])

    def test_apr05_s1_zero_hit_classification(self):
        """5 Apr Shift 1 was fully scanned with no rotational hits: the
        registry says so AND no record references its paper."""
        with open(os.path.join(CORPUS_DIR, "sources", "registry.json"), encoding="utf-8") as fh:
            registry = json.load(fh)["sources"]
        s1 = [s for s in registry if s["source_id"] == "nta-main-2026-apr05-s1"][0]
        self.assertEqual(s1["rotational_hits"], [])
        s1_url = "https://cdnbbsr.s3waas.gov.in/s3f8e59f4b2fe7c5705bf878bbd494ccdf/uploads/2026/04/20260409828731207.pdf"
        for record in _load_records():
            self.assertNotEqual(record["source_url"], s1_url)


if __name__ == "__main__":
    unittest.main()
