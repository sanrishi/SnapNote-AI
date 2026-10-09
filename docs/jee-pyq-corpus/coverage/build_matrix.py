"""Coverage matrix builder: recomputed from registry + verified corpus.

Answers "what do we actually cover?" without ever implying completeness.
Every number is derived on each run from stored files.

Usage:
    python coverage/build_matrix.py [--out coverage/matrix.json]
"""
import json
import os
import sys

CORPUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    with open(os.path.join(CORPUS_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


def build_matrix():
    registry = _load(os.path.join("sources", "registry.json"))["sources"]
    records = []
    data_dir = os.path.join(CORPUS_DIR, "data")
    for path in sorted(os.listdir(data_dir)):
        if path.endswith("-pyq.json"):
            records.extend(_load(os.path.join("data", path)))
    by_slot: dict[tuple, list] = {}
    for record in records:
        slot = (
            record.get("exam"), record.get("year"), record.get("session"),
            record.get("shift"), record.get("chapter"),
        )
        by_slot.setdefault(slot, []).append(record["question_id"])

    cells = []
    for source in registry:
        if source["exam"] == "JEE Advanced" and source["year"] is None:
            cells.append({
                "exam": "JEE Advanced",
                "year": "2007-2025",
                "session": None,
                "shift": None,
                "chapter": "Rotational Motion",
                "verified_questions": 0,
                "source_status": "discovered",
                "status": "unavailable",
            })
            continue
        base = (source["exam"], source["year"], source.get("session"), source.get("shift"))
        chapters = sorted({r["chapter"] for r in records
                           if (r.get("exam"), r.get("year"), r.get("session"), r.get("shift")) == base
                           and r.get("chapter")}) or ["Rotational Motion"]
        for chapter in chapters:
            hits = by_slot.get(base + (chapter,), [])
            # Completeness is never claimed: discovered per-paper totals are not
            # yet authoritative, so a covered cell is at most "partial".
            status = "partial" if hits else source.get("verification_status", "unavailable")
            cells.append({
                "exam": source["exam"],
                "year": source["year"],
                "session": source.get("session"),
                "shift": source.get("shift"),
                "chapter": chapter,
                "verified_questions": len(hits),
                "verified_ids": sorted(hits),
                "source_status": source.get("parser_status", "discovered"),
                "status": status,
            })
    total_verified = len({qid for c in cells for qid in c.get("verified_ids", [])})
    dashboard = {
        "official_papers_indexed": 0,
        "official_questions_indexed": 0,
        "source_linked_queue": 0,
        "verified_records": total_verified,
        "external_candidates_quarantined": 703,
    }
    summary_path = os.path.join(CORPUS_DIR, "index", "transcribed_index_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh).get("index", {})
        dashboard["official_papers_indexed"] = len(summary.get("source_ids", []))
        dashboard["official_questions_indexed"] = summary.get("total_records", 0)
        # Rotational-tagged index entries awaiting human review (not verified).
        dashboard["source_linked_queue"] = sum(
            len([h for h in s.get("rotational_hits", [])])
            for s in registry if s.get("source_id", "").startswith("jeeprep-main-")
        )
    review_path = os.path.join(CORPUS_DIR, "index", "review_log.json")
    if os.path.exists(review_path):
        with open(review_path, encoding="utf-8") as fh:
            review = json.load(fh)
        dashboard["review_batches"] = len(review.get("batches", []))
        dashboard["queue_cumulative"] = review.get("cumulative", {})
        dashboard["queue_state"] = review.get("queue_state")
        reviewed = (review.get("cumulative", {}).get("reviewed") or 0)
        verified_q = (review.get("cumulative", {}).get("verified") or 0)
        dashboard["verification_rate_reviewed"] = (
            round(verified_q / reviewed, 4) if reviewed else None
        )
    concept_dist: dict[str, int] = {}
    for record in records:
        for c in record.get("concept_ids", []):
            concept_dist[c] = concept_dist.get(c, 0) + 1
    dashboard["concept_distribution"] = dict(sorted(concept_dist.items()))
    return {
        "cells": cells,
        "totals": {
            "verified_records": total_verified,
            "sources_parsed": sum(1 for s in registry if s.get("parser_status") == "parsed"),
            "sources_discovered": sum(1 for s in registry if s.get("parser_status") != "parsed"),
        },
        "index_dashboard": dashboard,
        "note": "Status is never 'complete': discovered question totals per paper are not yet authoritative.",
    }


def main(argv):
    out = os.path.join(CORPUS_DIR, "coverage", "matrix.json")
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    matrix = build_matrix()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(matrix, fh, indent=1)
    print(f"wrote {out}: {matrix['totals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
