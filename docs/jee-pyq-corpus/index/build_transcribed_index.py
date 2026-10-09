"""Transcribed-official question index builder.

Source: JEEPrep public pages, which transcribe public NTA PDFs and claim to
align answers to the official final answer key. This is intentionally labelled
third-party transcription, never NTA-authored raw text.

Raw stems are cached locally (gitignored) so the join can run without copying
copyrighted question text into Git. A compact public summary is written next to
it with structure, hashes, and counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag

INDEX_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.dirname(INDEX_DIR)
sys.path.insert(0, os.path.join(CORPUS_DIR, "discovery"))
from import_candidates import normalize_text  # noqa: E402

BASE = "https://jeeprep.app"
SUBJECTS = ["physics", "chemistry", "mathematics"]
SUBJECT_TITLES = {"physics": "Physics", "chemistry": "Chemistry", "mathematics": "Mathematics"}

USER_AGENT = "SnapNoteAI-research-indexer/1.0 (+contact: local-dev)"


def _get(session: requests.Session, url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    path = os.path.join(cache_dir, key + ".html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            html = resp.text
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
            time.sleep(0.15)
            return html
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0 + attempt)
    raise RuntimeError(url)


def _paper_slugs(session: requests.Session, cache_dir: str, years: set[int]) -> list[str]:
    html = _get(session, BASE + "/pyq", cache_dir)
    soup = BeautifulSoup(html, "html.parser")
    slugs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.fullmatch(r"/pyq/(20\d{2}-[a-z]{3}-\d{1,2}-shift-\d)", href)
        if not m:
            continue
        year = int(href.split("/")[2][:4])
        if year in years:
            slugs.add(href.split("/")[2])
    return sorted(slugs)


def _article_text(article: Tag) -> tuple[str, list[str]]:
    metas = []
    header = article.find("div", class_=re.compile("mb-2"))
    if header:
        for span in header.find_all("span"):
            metas.append(span.get_text(" ", strip=True))
    prose = article.find("div", class_=re.compile("prose"))
    if not prose:
        return "", metas
    soup = BeautifulSoup(str(prose), "html.parser")
    for katex in soup.find_all("span", class_="katex"):
        ann = katex.find("annotation", encoding=re.compile("tex", re.I))
        if ann:
            katex.replace_with(ann.get_text(" ", strip=True))
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text, metas


def _parse_shift(metas: list[str]) -> dict:
    joined = " ".join(metas)
    m = re.search(r"JEE Main\s+(\d{4})\s+([A-Za-z]+)\s+(\d{1,2})\s+Shift\s+(\d)", joined)
    if m:
        year, month, day, shift = m.groups()
        # NTA 2024 Session 1 ran 27 Jan–1 Feb: Feb-1 shifts belong to January.
        if year == "2024" and month.lower().startswith("feb"):
            season = "January"
        else:
            season = "January" if month.lower().startswith("jan") else "April" if month.lower().startswith("apr") else month
        return {
            "year": int(year),
            "session": f"{season} {year}",
            "shift": f"Shift {shift} ({day} {month} {year})",
            "exam_label": joined,
        }
    return {"year": None, "session": None, "shift": None, "exam_label": joined}


def parse_subject_page(html: str, *, slug: str, subject: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for article in soup.find_all("article"):
        qnum = None
        header = article.find("div", class_=re.compile("mb-2"))
        if header:
            for span in header.find_all("span"):
                qm = re.match(r"Q\s*(\d+)", span.get_text(" ", strip=True))
                if qm:
                    qnum = int(qm.group(1))
                    break
        if qnum is None:
            continue
        stem, metas = _article_text(article)
        if not stem:
            continue
        shift = _parse_shift(metas)
        if shift["year"] is None:
            # Fallback: derive year/month from the URL slug.
            m = re.match(r"(20\d{2})-([a-z]{3})-(\d{1,2})-shift-(\d)", slug)
            if m:
                y, mon, day, sh = m.groups()
                if y == "2024" and mon == "feb":
                    season = "January"  # 2024 Session 1 ran 27 Jan–1 Feb.
                else:
                    season = "January" if mon == "jan" else "April" if mon == "apr" else mon
                shift = {"year": int(y), "session": f"{season} {y}", "shift": f"Shift {sh} ({day} {mon} {y})", "exam_label": slug}
        normalized = normalize_text(stem)
        out.append(
            {
                "source_id": f"jeeprep-main-{shift['year']}-{slug}",
                "exam": "JEE Main",
                "year": shift["year"],
                "session": shift["session"],
                "shift": shift["shift"],
                "subject": SUBJECT_TITLES[subject],
                "question_number": qnum,
                "page": None,
                "stem_text": stem[:4000],
                "normalized_text_hash": hashlib.sha256(
                    f"{SUBJECT_TITLES[subject]}|{normalized}".encode("utf-8")
                ).hexdigest()[:24],
                "stem_chars": len(stem),
                "source_url": f"{BASE}/pyq/{slug}/{subject}",
                "source_dataset": "jeeprep-transcribed-nta",
                "answer_key_status": "claimed-official-final-answer-key",
                "provenance_status": "third-party-transcription",
                "license_status": "unknown-third-party-mirror",
                "topic": metas[2] if len(metas) >= 3 else None,
                "difficulty": metas[3] if len(metas) >= 4 else None,
            }
        )
    return out


def public_summary(records: list[dict]) -> dict:
    by_year: dict[str, int] = {}
    by_subject: dict[str, int] = {}
    by_shift: dict[str, int] = {}
    for r in records:
        by_year[str(r["year"])] = by_year.get(str(r["year"]), 0) + 1
        by_subject[r["subject"]] = by_subject.get(r["subject"], 0) + 1
        key = f"{r['year']} | {r['session']} | {r['shift']}"
        by_shift[key] = by_shift.get(key, 0) + 1
    return {
        "total_records": len(records),
        "by_year": by_year,
        "by_subject": by_subject,
        "by_shift_count": len(by_shift),
        "shifts": sorted(by_shift),
        "source_ids": sorted({r["source_id"] for r in records}),
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="2024,2025")
    p.add_argument("--out", default=os.path.join(INDEX_DIR, "transcribed_index.local.json"))
    p.add_argument("--summary", default=os.path.join(INDEX_DIR, "transcribed_index_summary.json"))
    p.add_argument("--cache", default=os.path.join(CORPUS_DIR, ".cache", "jeeprep"))
    p.add_argument("--limit-papers", type=int, default=0)
    args = p.parse_args(argv[1:])
    years = {int(y.strip()) for y in args.years.split(",") if y.strip()}
    session = requests.Session()
    slugs = _paper_slugs(session, args.cache, years)
    if args.limit_papers:
        slugs = slugs[: args.limit_papers]
    records = []
    failures = []
    for slug in slugs:
        for subject in SUBJECTS:
            url = f"{BASE}/pyq/{slug}/{subject}"
            try:
                html = _get(session, url, args.cache)
                recs = parse_subject_page(html, slug=slug, subject=subject)
                records.extend(recs)
                print(f"{slug}/{subject}: {len(recs)}", flush=True)
            except Exception as exc:
                failures.append({"url": url, "error": str(exc)})
                print(f"FAIL {url}: {exc}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)
    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump({"index": public_summary(records), "failures": failures}, fh, indent=1)
    print(json.dumps({"records": len(records), "papers": len(slugs), "failures": len(failures), "out": args.out, "summary": args.summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
