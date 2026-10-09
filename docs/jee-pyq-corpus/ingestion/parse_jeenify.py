"""JEEnify paper parser: single-paper HTML -> transcribed-index records.

Source: JEEnify public paper pages (https://www.jeenify.com/jee-main-pyq),
which transcribe NTA papers with JSON-LD ``Question`` + ``acceptedAnswer``
markup, ``Answer: Option N`` HTML, ``Final answer:`` text, and ``qna-image``
figure URLs. Labelled third-party transcription, never NTA-authored text.

Raw HTML is cached locally (gitignored); only structure/hashes are committed.

Layout notes (locked from the 2023/january-24-shift-1 pilot):
- Question chunks split on ``Q<!-- -->NN</span>`` markers.
- Subject identity is POSITION-based (Physics block, then Chemistry, then
  Mathematics) because dropped-question numbering leaves gaps (pilot: Q41
  and Q78 absent, 88 parsed of 90).
- Chapter labels come from the ``ml-auto truncate`` span; coarse labels
  (``Optics``, ``Vectors``, ``Organic/Inorganic/Physical Chemistry`` …)
  stay unmapped and fail loudly in the queue builder.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request

BASE = "https://www.jeenify.com/jee-main-pyq"
USER_AGENT = "SnapNoteAI-research-indexer/1.0 (+contact: local-dev)"

SUBJECT_ORDER = ["Physics", "Chemistry", "Mathematics"]


def fetch_html(slug: str, cache_dir: str) -> str:
    """Fetch one paper page, caching raw HTML locally. Returns the HTML."""
    url = BASE + "/" + slug
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    path = os.path.join(cache_dir, key + ".html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html


def strip_tags(fragment: str) -> str:
    """Strip scripts/styles/tags to plain stem text."""
    text = re.sub(r"<script.*?</script>", " ", fragment, flags=re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("<!-- -->", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_paper(html: str) -> list[dict]:
    """Parse question chunks. Returns dicts with qnum/qtype/chapter/answer/stem."""
    chunks = re.split(r"Q<!-- -->(\d+)</span>", html)
    out: list[dict] = []
    for i in range(1, len(chunks), 2):
        qnum = int(chunks[i])
        body = chunks[i + 1] if i + 1 < len(chunks) else ""
        mtype = re.search(
            r"text-\[11px\] font-semibold uppercase[^>]*>(Single correct|Numerical|[^<]{1,30})</span>",
            body,
        )
        qtype = mtype.group(1).strip() if mtype else "Unknown"
        mchap = re.search(r"ml-auto truncate[^>]*>([^<]{1,80})</span>", body)
        chapter = mchap.group(1).strip() if mchap else ""
        mans = re.search(r"Answer: <!-- -->(Option (\d)|([^<]{1,200}))", body)
        answer_raw = mans.group(1).strip() if mans else ""
        start = mchap.end() if mchap else 0
        mopt = re.search(r"\(1\)", body[start:])
        mshow = body.find("Show Solution", start)
        end = len(body)
        if mopt:
            end = min(end, start + mopt.start())
        if mshow > 0:
            end = min(end, mshow)
        stem_html = body[start:end]
        images = re.findall(r"/api/qna-image\?path=([^\"\s&]+)", stem_html)
        stem = strip_tags(stem_html)
        out.append(
            {
                "qnum": qnum,
                "qtype": qtype,
                "chapter": chapter,
                "answer_raw": answer_raw,
                "stem": stem,
                "stem_chars": len(stem),
                "n_images": len(images),
            }
        )
    return out


def subject_split(html: str, n_questions: int) -> list[tuple[str, int]]:
    """Read the ``Physics (N), Chemistry (N) and Mathematics (N)`` header.

    Falls back to equal thirds. Callers assign subjects by POSITION in
    parse order — never by question number — so dropped-question gaps
    cannot shift a subject boundary.
    """
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    m = re.search(
        r"Physics \(\s*(\d+)\s*\), Chemistry \(\s*(\d+)\s*\) and Mathematics \(\s*(\d+)\s*\)",
        text,
    )
    if m:
        return [
            ("Physics", int(m.group(1))),
            ("Chemistry", int(m.group(2))),
            ("Mathematics", int(m.group(3))),
        ]
    third = n_questions // 3
    return [
        ("Physics", third),
        ("Chemistry", third),
        ("Mathematics", n_questions - 2 * third),
    ]


def subjects_by_position(split: list[tuple[str, int]], n: int) -> list[str]:
    """Expand a subject split into a per-position subject list."""
    order: list[str] = []
    for subject, count in split:
        order += [subject] * count
    if len(order) < n:
        order += ["Unknown"] * (n - len(order))
    return order[:n]
