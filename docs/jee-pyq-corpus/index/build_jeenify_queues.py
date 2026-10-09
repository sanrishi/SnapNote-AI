"""JEEnify candidate queues: local jeenify index -> per-chapter queues.

Writes ``.cache/queues/<Subject>/<chapter>.jeenify.local.json`` (gitignored
stems live only in the local index; queue files carry hashes + triage flags)
and ``index/chapter_queues_jeenify_summary.json`` (public counts only).

Never touches the jeeprep ``*.local.json`` queues. Identity-dupes are
checked against the FULL verified corpus (all ``data/*-pyq.json`` shards).

Status vocabulary matches the factory: HIGH_CONFIDENCE / NEEDS_SOURCE_REVIEW
/ CORRUPTED / DUPLICATE.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

CORPUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CORPUS, "ingestion"))
from prescreen_queue import integrity_flags  # noqa: E402

JEENIFY_INDEX = os.path.join(CORPUS, "index", "transcribed_index_jeenify.local.json")
TAX_MAP = os.path.join(CORPUS, "..", "jee-chapter-taxonomy", "data", "mirror_topic_map.json")
QUEUE_DIR = os.path.join(CORPUS, ".cache", "queues")
SUMMARY = os.path.join(CORPUS, "index", "chapter_queues_jeenify_summary.json")


def verified_identities() -> set[tuple]:
    idents: set[tuple] = set()
    for path in glob.glob(os.path.join(CORPUS, "data", "*-pyq.json")):
        for r in json.load(open(path, encoding="utf-8")):
            idents.add((r.get("year"), str(r.get("session")), str(r.get("shift")),
                        r.get("paper_question_number")))
    return idents


def classify(flags: list[str]) -> str:
    fl = set(flags)
    if "identity_already_verified" in fl:
        return "DUPLICATE"
    if {"has_corruption_marker", "truncated_stem", "stem_too_short",
        "too_few_words", "index_record_missing"} & fl:
        return "CORRUPTED"
    if not fl:
        return "HIGH_CONFIDENCE"
    return "NEEDS_SOURCE_REVIEW"


def main() -> int:
    index: list[dict] = json.load(open(JEENIFY_INDEX, encoding="utf-8"))
    topic_map: dict = json.load(open(TAX_MAP, encoding="utf-8"))
    idents = verified_identities()
    queues: dict[tuple, list] = collections.defaultdict(list)
    unmapped: collections.Counter = collections.Counter()
    for rec in index:
        topic = rec.get("topic")
        entry = topic_map.get(topic or "")
        if entry is None or entry["subject"] != rec["subject"]:
            unmapped["%s | %s" % (rec["subject"], topic or "UNTAGGED")] += 1
            continue
        flags = integrity_flags(rec.get("stem_text", ""))
        for field in ("exam", "year", "session", "shift", "subject", "source_url"):
            if not rec.get(field):
                flags.append("missing_%s" % field)
        if rec.get("has_figure"):
            flags.append("has_figure")
        ident = (rec.get("year"), str(rec.get("session")), str(rec.get("shift")),
                 rec.get("question_number"))
        if ident in idents:
            flags.append("identity_already_verified")
        state = classify(sorted(set(flags)))
        queues[(rec["subject"], entry["chapter_id"])].append({
            "queue_id": "cand-%s-q%d" % (rec["source_id"], rec["question_number"]),
            "source_id": rec["source_id"],
            "year": rec["year"],
            "session": rec["session"],
            "shift": rec["shift"],
            "subject": rec["subject"],
            "question_number": rec["question_number"],
            "normalized_text_hash": rec["normalized_text_hash"],
            "mirror_topic": topic,
            "mirror_difficulty": rec.get("difficulty"),
            "mirror_answer_raw": rec.get("mirror_answer_raw"),
            "question_kind": rec.get("question_kind"),
            "state": state,
            "flags": sorted(set(flags)),
            "source_url": rec["source_url"],
        })
    summary: dict = {"chapters": [], "unmapped_labels": dict(unmapped)}
    total = 0
    for (subject, chapter), items in sorted(queues.items()):
        states = collections.Counter(i["state"] for i in items)
        path = os.path.join(QUEUE_DIR, subject, chapter.split("jee-")[1] + ".jeenify.local.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        old: list[dict] = []
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
        have = {e["queue_id"] for e in old}
        old.extend(e for e in items if e["queue_id"] not in have)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(old, fh, indent=1)
        summary["chapters"].append({"subject": subject, "chapter_id": chapter,
                                    "candidates": len(old), "states": dict(states)})
        total += len(items)
    summary["total_candidates"] = sum(c["candidates"] for c in summary["chapters"])
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"queue_files": len(queues), "candidates": total,
                      "unmapped": dict(unmapped)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
