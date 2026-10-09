"""Candidate triage: cheap OCR keyword pass over paper pages.

Purpose: avoid deep manual inspection of all 25 physics questions per paper.
 recall-oriented: flag a page if ANY rotational vocabulary appears; a human
verifies flagged pages only. Candidates != verified records.

Vocabulary is built from the canonical taxonomy (aliases + display names)
plus mechanics terms — never invented chapter claims.

Usage:
    python ingestion/triage.py <paper.pdf> [--dpi 130]
Requires: pymupdf, easyocr.
"""
import json
import os
import re
import sys

# Rotational vocabulary: taxonomy-derived + mechanics terms.
# Deliberately broad (recall first); verification happens downstream.
VOCAB = [
    # taxonomy aliases + display names (normalized on load too)
    "torque", "moment of a force", "turning moment",
    "angular momentum", "rotational momentum",
    "moment of inertia", "radius of gyration",
    "parallel axis", "perpendicular axis", "parallel axes",
    "centre of mass", "center of mass",
    "rigid body", "angular velocity", "angular acceleration",
    "angular impulse", "rotational motion", "rotation",
    "rolling", "rolls", "spin",
    "revolution", "revolutions", "rpm", "rotates", "rotating",
    "axis of rotation", "lever arm",
]


def load_taxonomy_terms(corpus_dir: str) -> list[str]:
    """Extend vocabulary with every taxonomy alias + display name."""
    path = os.path.normpath(os.path.join(
        corpus_dir, "..", "jee-concept-taxonomy", "data",
        "jee-physics-rotational-motion-taxonomy.json",
    ))
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        concepts = json.load(fh)
    terms: list[str] = []
    for concept in concepts:
        terms.append(str(concept.get("display_name", "")))
        terms.extend(str(a) for a in concept.get("aliases", []))
    return [t for t in terms if t]


def triage_page_text(text: str, vocab: list[str]) -> list[str]:
    """Return the vocabulary hits in OCR text (case-insensitive).

    Multi-character terms match as phrases; single-character symbols (I, L,
    tau) match as standalone tokens only — substring matching on one letter
    flags every page and destroys precision.
    """
    lowered = text.casefold()
    hits = set()
    for term in vocab:
        if not term:
            continue
        folded = term.casefold()
        if len(folded) <= 1:
            # Single symbols (I, L, tau) match roman numerals, option labels
            # and variables on nearly every page. They destroy precision while
            # adding no recall: real stems always contain a word-level term.
            continue
        if folded in lowered:
            hits.add(term)
    return sorted(hits)


def triage_paper(pdf_path: str, dpi: int = 130) -> dict:
    """Render physics pages, OCR them, flag candidates. Returns a report."""
    import fitz

    try:
        import easyocr
    except ImportError:
        return {"error": "easyocr not installed", "pages": []}
    corpus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab = VOCAB + load_taxonomy_terms(corpus_dir)
    doc = fitz.open(pdf_path)
    reader = easyocr.Reader(["en"], verbose=False)
    pages = []
    for i, page in enumerate(doc):
        text_markers = page.get_text()
        if "Physics Section" not in text_markers and not any(
            f"Question Number : {n}" in text_markers for n in range(26, 51)
        ):
            continue
        pix = page.get_pixmap(dpi=dpi)
        import numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        ocr_text = " ".join(entry[1] for entry in reader.readtext(img))
        hits = triage_page_text(ocr_text, vocab)
        pages.append({"page": i + 1, "flagged": bool(hits), "hits": hits})
    flagged = sum(1 for p in pages if p["flagged"])
    return {
        "pdf": os.path.basename(pdf_path),
        "physics_pages_scanned": len(pages),
        "pages_flagged": flagged,
        "pages": pages,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: triage.py <paper.pdf> [--dpi 130]")
        return 2
    dpi = int(argv[argv.index("--dpi") + 1]) if "--dpi" in argv else 130
    report = triage_paper(argv[1], dpi)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
