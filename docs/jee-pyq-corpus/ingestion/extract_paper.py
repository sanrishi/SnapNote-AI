"""Ingestion pipeline: official NTA paper PDF -> candidate question records.

Produces CANDIDATES (verification_status: needs-review), never verified
records. A candidate carries only structure parsed from the paper
(sections, question numbers, NTA IDs, types, marks arithmetic) plus file
provenance. Concept mapping and verification are separate human steps.

Usage:
    python ingestion/extract_paper.py <paper.pdf> <source_id> [--out candidates/]
Requires: pymupdf (pip install pymupdf). Network fetch is the caller's job.
"""
import hashlib
import json
import os
import re
import sys
from datetime import date

QUESTION_RE = re.compile(
    r"Question Number\s*:\s*(\d+)\s*Question Id\s*:\s*(\d+)\s*"
    r"Question Type\s*:\s*(MCQ|SA)",
    re.IGNORECASE,
)
SECTION_RE = re.compile(
    r"(Mathematics|Physics|Chemistry) Section ([AB])\b.*?"
    r"Number of Questions\s*:\s*(\d+).*?"
    r"Section Marks\s*:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_candidates(pdf_path: str, source_id: str) -> dict:
    """Parse structure from one official paper. No content interpretation."""
    import fitz

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    sections = []
    for match in SECTION_RE.finditer(full_text):
        subject, section, n_questions, section_marks = match.groups()
        sections.append(
            {
                "subject": subject.capitalize(),
                "section": section.upper(),
                "n_questions": int(n_questions),
                "section_marks": int(section_marks),
                "marks_per_question": int(section_marks) // int(n_questions),
            }
        )
    questions = []
    for match in QUESTION_RE.finditer(full_text):
        number, nta_id, qtype = match.groups()
        questions.append(
            {
                "paper_question_number": int(number),
                "nta_question_id": nta_id,
                "question_type": "MCQ" if qtype.upper() == "MCQ" else "Numerical",
            }
        )
    return {
        "source_id": source_id,
        "parser": "extract_paper.py v1",
        "parsed_at": date.today().isoformat(),
        "pdf_sha256": _sha256(pdf_path),
        "n_pages": len(doc),
        "sections": sections,
        "n_questions_parsed": len(questions),
        "questions": questions,
        "verification_status": "needs-review",
        "note": "Structure only. Concept mapping + verification are separate human steps.",
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: extract_paper.py <paper.pdf> <source_id> [--out DIR]")
        return 2
    pdf_path, source_id = argv[1], argv[2]
    out_dir = "candidates"
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
    os.makedirs(out_dir, exist_ok=True)
    result = extract_candidates(pdf_path, source_id)
    out_path = os.path.join(out_dir, f"{source_id}.candidates.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {out_path}: {result['n_questions_parsed']} questions, "
          f"{len(result['sections'])} sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
