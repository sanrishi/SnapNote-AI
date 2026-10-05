"""Chapter taxonomy builder: NTA syllabus units -> chapter-level concepts.

Source of truth: the official NTA JEE Main syllabus PDF. Chapter names are
facts, not expression; the generated data file carries the official source
URL so anyone can re-derive it. Run:
    python build_chapters.py [--pdf URL] [--out data/jee-chapters.json]

Chapter IDs: jee-<subject>-<slug>-chapter (stable, human-readable).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "jee-chapters.json")
SYLLABUS_URL = ("https://cdnbbsr.s3waas.gov.in/s3f8e59f4b2fe7c5705bf878bbd494ccdf"
                "/uploads/2024/10/2024102841.pdf")

SUBJECT_ORDER = ["Mathematics", "Physics", "Chemistry"]


def slugify(unit: str) -> str:
    s = unit.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return re.sub(r"-+", "-", s)


def title_case(unit: str) -> str:
    small = {"and", "of", "in", "its", "for", "the", "a", "an", "to"}
    words = unit.lower().replace(" - ", "-").split()
    out = []
    for i, w in enumerate(words):
        if i and w in small:
            out.append(w)
        elif "-" in w:
            out.append("-".join(p.capitalize() for p in w.split("-")))
        else:
            out.append(w.capitalize())
    return " ".join(out)


def extract_units(pdf_text: str) -> dict[str, list[str]]:
    math_end = pdf_text.find("PHYSICS")
    chem_start = pdf_text.find("CHEMISTRY")
    # The PDF appends the Paper 2A (B.Arch) syllabus after Chemistry; cut it.
    paper2 = pdf_text.find("Paper 2A")
    chem_end = paper2 if paper2 > chem_start else len(pdf_text)
    spans = {
        "Mathematics": pdf_text[:math_end],
        "Physics": pdf_text[math_end:chem_start],
        "Chemistry": pdf_text[chem_start:chem_end],
    }
    out: dict[str, list[str]] = {}
    for subject, span in spans.items():
        found: list[str] = []
        for m in re.finditer(
            r"(?i)\bUNIT\s*(?:\d{1,2}|[IVX]{1,5})\s*[:.\-–]*\s*([A-Z][A-Za-z ,&()/\-]{2,90})",
            span,
        ):
            u = m.group(1).strip()
            if u not in found:
                found.append(u)
        out[subject] = found
    return out


def fetch_pdf_text(url: str) -> str:
    import fitz
    import requests

    cache = os.path.join(HERE, ".cache", "nta-syllabus.pdf")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(cache, "wb") as fh:
            fh.write(resp.content)
    doc = fitz.open(cache)
    return "\n".join(p.get_text() for p in doc)


def build(pdf_text: str) -> list[dict]:
    units = extract_units(pdf_text)
    chapters = []
    for subject in SUBJECT_ORDER:
        for unit in units[subject]:
            slug = slugify(unit)
            chapters.append({
                "chapter_id": f"jee-{subject.lower()}-{slug}-chapter",
                "subject": subject,
                "display_name": title_case(unit),
                "syllabus_unit": unit,
                "syllabus_source_url": SYLLABUS_URL,
                "level": "chapter",
                "note": "Chapter-level concept derived systematically from the "
                        "official NTA syllabus unit list. Sub-concept taxonomy "
                        "is built per chapter on demand.",
            })
    return chapters


def main(argv: list[str]) -> int:
    url = argv[argv.index("--pdf") + 1] if "--pdf" in argv else SYLLABUS_URL
    out = argv[argv.index("--out") + 1] if "--out" in argv else DATA
    chapters = build(fetch_pdf_text(url))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(chapters, fh, indent=1)
    by_subject: dict[str, int] = {}
    for c in chapters:
        by_subject[c["subject"]] = by_subject.get(c["subject"], 0) + 1
    print(json.dumps({"total": len(chapters), "by_subject": by_subject, "out": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
