"""JEEnify transcribed-index builder: paper slugs -> local index records.

Appends to ``index/transcribed_index_jeenify.local.json`` (gitignored raw
stems) and writes ``index/transcribed_index_jeenify_summary.json`` (public:
structure, hashes, counts only). Dedupes by (source_id, question_number).

Subject assignment is position-based (see ingestion/parse_jeenify.py):
robust to dropped-question numbering gaps.

Usage:
    python index/build_jeenify_index.py 2023/january-24-shift-1 [...]
    python index/build_jeenify_index.py --slugs 2023/january-24-shift-1 --limit 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

INDEX_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.dirname(INDEX_DIR)
sys.path.insert(0, os.path.join(CORPUS_DIR, "ingestion"))
from parse_jeenify import fetch_html, parse_paper, subject_split, subjects_by_position  # noqa: E402

DEFAULT_OUT = os.path.join(INDEX_DIR, "transcribed_index_jeenify.local.json")
DEFAULT_SUMMARY = os.path.join(INDEX_DIR, "transcribed_index_jeenify_summary.json")
DEFAULT_CACHE = os.path.join(CORPUS_DIR, ".cache", "jeenify")

LONG_MONTH = {
    "january": "January",
    "february": "February",
    "april": "April",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
}


def parse_slug(slug: str) -> tuple[int, str, int, str]:
    """Split ``2023/january-24-shift-1`` -> (year, long month, day, shift)."""
    m = re.match(r"(\d{4})/([a-z]+)-(\d+)(?:-shift-(\d+))?$", slug)
    if not m:
        raise ValueError("bad slug " + slug)
    year, mon, day = int(m.group(1)), m.group(2), int(m.group(3))
    shift = m.group(4) or "1"
    if mon not in LONG_MONTH:
        raise ValueError("bad month in slug " + slug)
    return year, LONG_MONTH[mon], day, shift


def session_for(year: int, long_month: str) -> str:
    if year == 2024 and long_month == "February":
        return "January 2024"  # NTA 2024 Session 1 ran 27 Jan-1 Feb.
    if long_month == "January":
        return f"January {year}"
    return f"{long_month} {year}"


def public_summary(records: list[dict]) -> dict:
    by_year: dict[str, int] = {}
    by_subject: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in records:
        by_year[str(r["year"])] = by_year.get(str(r["year"]), 0) + 1
        by_subject[r["subject"]] = by_subject.get(r["subject"], 0) + 1
        by_source[r["source_id"]] = by_source.get(r["source_id"], 0) + 1
    return {
        "total_records": len(records),
        "by_year": by_year,
        "by_subject": by_subject,
        "by_source": by_source,
        "source_ids": sorted(by_source),
    }


def build(slugs: list[str], cache_dir: str) -> tuple[list[dict], list[dict]]:
    """Fetch + parse each slug. Returns (new_records, failures)."""
    records: list[dict] = []
    failures: list[dict] = []
    for slug in slugs:
        try:
            year, long_month, day, shift = parse_slug(slug)
        except ValueError as exc:
            failures.append({"slug": slug, "error": str(exc)})
            continue
        try:
            html = fetch_html(slug, cache_dir)
            parsed = parse_paper(html)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append({"slug": slug, "error": str(exc)})
            continue
        split = subject_split(html, len(parsed))
        subjects = subjects_by_position(split, len(parsed))
        source_id = f"jeenify-main-{year}-{long_month[:3].lower()}-{day:02d}-shift-{shift}"
        source_url = f"https://www.jeenify.com/jee-main-pyq/{slug}"
        for pos, q in enumerate(parsed):
            stem = q["stem"]
            records.append(
                {
                    "source_id": source_id,
                    "exam": "JEE Main",
                    "year": year,
                    "session": session_for(year, long_month),
                    "shift": f"Shift {shift} ({day} {long_month} {year})",
                    "subject": subjects[pos],
                    "question_number": q["qnum"],
                    "page": slug,
                    "stem_text": stem[:4000],
                    "normalized_text_hash": hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16],
                    "stem_chars": q["stem_chars"],
                    "source_url": source_url,
                    "source_dataset": "jeenify",
                    "answer_key_status": "mirror-claimed",
                    "provenance_status": "third-party-transcription",
                    "license_status": "unknown",
                    "topic": q["chapter"],
                    "difficulty": "",
                    "mirror_answer_raw": q["answer_raw"],
                    "question_kind": q["qtype"],
                    "has_figure": q["n_images"] > 0,
                }
            )
    return records, failures


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("slugs", nargs="*", help="slugs like 2023/january-24-shift-1")
    p.add_argument("--slugs", dest="slugs_opt", default="")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    args = p.parse_args(argv[1:])
    slugs = list(args.slugs)
    if args.slugs_opt:
        slugs += [s.strip() for s in args.slugs_opt.split(",") if s.strip()]
    if not slugs:
        print("usage: build_jeenify_index.py <slug> [...]")
        return 2
    try:
        existing: list[dict] = json.load(open(args.out, encoding="utf-8"))
    except FileNotFoundError:
        existing = []
    have = {(e["source_id"], e["question_number"]) for e in existing}
    fresh, failures = build(slugs, args.cache)
    added = [r for r in fresh if (r["source_id"], r["question_number"]) not in have]
    existing.extend(added)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=1, ensure_ascii=False)
    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump({"index": public_summary(existing), "failures": failures}, fh, indent=1)
    print(json.dumps({"parsed": len(fresh), "added": len(added),
                      "total": len(existing), "failures": failures}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
