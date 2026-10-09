"""Official Question Index builder: official paper PDFs -> question records.

For each physics question: exam metadata (from registry) + OCR stem text +
normalized fingerprint. Stems in NTA papers are images, so OCR is required;
structure (numbers, IDs, types) comes from the PDF text layer via
ingestion/extract_paper.py candidates.

Output: index/official_index.json — the truth side of the join. External
datasets never contribute exam facts; they only match against this index.
"""
import hashlib
import json
import os
import re
import sys

INDEX_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.dirname(INDEX_DIR)

# Share normalization with the importer so both sides fingerprint identically.
sys.path.insert(0, os.path.join(CORPUS_DIR, "discovery"))
from import_candidates import normalize_text  # noqa: E402


def _question_ranges(n_questions=75):
    return {"Mathematics": (1, 25), "Physics": (26, 50), "Chemistry": (51, 75)}


def split_page_questions(ocr_text: str) -> list[tuple[int, str]]:
    """Split OCR page text into (question_number, stem_text) chunks.

    OCR fragments the header ("Question Number" / "30 Question Id" land in
    separate entries and colons are often dropped), so match loosely.
    """
    parts = re.split(r"Question Number\s*:?\s*(\d+)\s*Question Id", ocr_text)
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        try:
            chunks.append((int(parts[i]), parts[i + 1]))
        except ValueError:
            continue
    return chunks


def build_index(pdf_path: str, source: dict, pages_dir: str, dpi: int = 130) -> list[dict]:
    """OCR physics pages and emit official question records."""
    import fitz
    import numpy as np

    try:
        import easyocr
    except ImportError:
        raise SystemExit("easyocr required: pip install easyocr")
    os.makedirs(pages_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    reader = easyocr.Reader(["en"], verbose=False)
    ranges = _question_ranges()
    records = []
    for i, page in enumerate(doc):
        text_layer = page.get_text()
        if "Physics Section" not in text_layer and not re.search(
            r"Question Number\s*:\s*(2[6-9]|3\d|4\d|50)\b", text_layer
        ):
            continue
        png = os.path.join(pages_dir, f"p{i + 1}.png")
        if not os.path.exists(png):
            page.get_pixmap(dpi=dpi).save(png)
        txt_cache = os.path.join(pages_dir, f"p{i + 1}.ocr.txt")
        if os.path.exists(txt_cache):
            with open(txt_cache, encoding="utf-8") as fh:
                ocr_text = fh.read()
        else:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            ocr_text = " ".join(entry[1] for entry in reader.readtext(img))
            with open(txt_cache, "w", encoding="utf-8") as fh:
                fh.write(ocr_text)
        for number, stem in split_page_questions(ocr_text):
            if not 26 <= number <= 50:
                continue  # physics slice only; neighbors belong to other sections
            subject = next(
                (s for s, (lo, hi) in ranges.items() if lo <= number <= hi), "Unknown"
            )
            normalized = normalize_text(stem)
            records.append(
                {
                    "source_id": source["source_id"],
                    "exam": source["exam"],
                    "year": source["year"],
                    "session": source.get("session"),
                    "shift": source.get("shift"),
                    "subject": subject,
                    "question_number": number,
                    "page": i + 1,
                    "stem_text": stem.strip()[:2000],
                    "normalized_text_hash": hashlib.sha256(
                        f"{subject}|{normalized}".encode("utf-8")
                    ).hexdigest()[:24],
                    "stem_chars": len(stem),
                    "source_url": source["source_url"],
                }
            )
        print(f"  page {i + 1}: {len(split_page_questions(ocr_text))} questions", flush=True)
    return records


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: build_index.py <paper.pdf> <source_id> [--pages DIR] [--out PATH]")
        return 2
    with open(os.path.join(CORPUS_DIR, "sources", "registry.json"), encoding="utf-8") as fh:
        sources = {s["source_id"]: s for s in json.load(fh)["sources"]}
    source = sources[argv[2]]
    pages_dir = argv[argv.index("--pages") + 1] if "--pages" in argv else os.path.join(
        INDEX_DIR, "pages", argv[2]
    )
    records = build_index(argv[1], source, pages_dir)
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(
        INDEX_DIR, "official_index.json"
    )
    existing = []
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            existing = json.load(fh)
    kept = [r for r in existing if r["source_id"] != argv[2]]
    kept.extend(records)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, indent=1)
    print(f"index: {len(records)} questions from {argv[2]} ({len(kept)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
