"""Mass candidate generation: local index -> per-chapter candidate queues.

For every (subject, chapter) in the mirror_topic_map, emits:
  - .cache/queues/<subject>/<chapter>.local.json  (full, gitignored: stems)
  - index/chapter_queues_summary.json              (public: counts only)

Each candidate carries triage-lite flags (integrity, metadata completeness,
identity-dupe vs the verified corpus) but NO verification verdict.
Status vocabulary matches the established pipeline:
CANDIDATE / HIGH_CONFIDENCE / NEEDS_SOURCE_REVIEW / CORRUPTED / DUPLICATE.
"""
import collections
import hashlib
import json
import os
import sys

CORPUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CORPUS, "ingestion"))
from prescreen_queue import integrity_flags  # noqa: E402

LOCAL_INDEX = os.path.join(CORPUS, "index", "transcribed_index.local.json")
TAX_DIR = os.path.join(CORPUS, "..", "jee-chapter-taxonomy", "data")
QUEUE_DIR = os.path.join(CORPUS, ".cache", "queues")
SUMMARY = os.path.join(CORPUS, "index", "chapter_queues_summary.json")


def main() -> int:
    index = json.load(open(LOCAL_INDEX, encoding="utf-8"))
    topic_map = json.load(open(os.path.join(TAX_DIR, "mirror_topic_map.json"), encoding="utf-8"))
    verified = json.load(open(os.path.join(
        CORPUS, "data", "jee-main-physics-rotational-pyq.json"), encoding="utf-8"))
    verified_idents = {
        (r.get("year"), str(r.get("session")), str(r.get("shift")),
         r.get("paper_question_number")) for r in verified
    }

    queues: dict[tuple, list] = collections.defaultdict(list)
    unmapped = collections.Counter()
    for rec in index:
        topic = rec.get("topic")
        entry = topic_map.get(topic)
        if entry is None or entry["subject"] != rec["subject"]:
            unmapped[topic or "UNTAGGED"] += 1
            continue
        flags = integrity_flags(rec.get("stem_text", ""))
        for field in ("exam", "year", "session", "shift", "subject", "source_url"):
            if not rec.get(field):
                flags.append(f"missing_{field}")
        ident = (rec.get("year"), str(rec.get("session")), str(rec.get("shift")),
                 rec.get("question_number"))
        state = "CANDIDATE"
        if ident in verified_idents:
            state = "DUPLICATE"
            flags.append("identity_already_verified")
        elif {"has_corruption_marker", "truncated_stem", "stem_too_short",
              "too_few_words", "index_record_missing"} & set(flags):
            state = "CORRUPTED"
        elif not flags:
            state = "HIGH_CONFIDENCE"
        else:
            state = "NEEDS_SOURCE_REVIEW"
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
            "state": state,
            "flags": sorted(set(flags)),
            "source_url": rec["source_url"],
        })

    summary = {"chapters": [], "unmapped_labels": dict(unmapped)}
    for (subject, chapter), items in sorted(queues.items()):
        states = collections.Counter(i["state"] for i in items)
        path = os.path.join(QUEUE_DIR, subject, chapter.split("jee-")[1] + ".local.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=1)
        summary["chapters"].append({
            "subject": subject, "chapter_id": chapter,
            "candidates": len(items), "states": dict(states),
        })
    summary["total_candidates"] = sum(c["candidates"] for c in summary["chapters"])
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"chapters": len(summary["chapters"]),
                      "total": summary["total_candidates"],
                      "unmapped": dict(unmapped)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
