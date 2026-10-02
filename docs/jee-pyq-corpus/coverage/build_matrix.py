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
    records = _load(os.path.join("data", "jee-main-physics-rotational-pyq.json"))
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
        slot = (
            source["exam"], source["year"], source.get("session"),
            source.get("shift"), "Rotational Motion",
        )
        hits = by_slot.get(slot, [])
        # Completeness is never claimed: discovered per-paper totals are not
        # yet authoritative, so a covered cell is at most "partial".
        status = "partial" if hits else source.get("verification_status", "unavailable")
        cells.append({
            "exam": source["exam"],
            "year": source["year"],
            "session": source.get("session"),
            "shift": source.get("shift"),
            "chapter": "Rotational Motion",
            "verified_questions": len(hits),
            "verified_ids": sorted(hits),
            "source_status": source.get("parser_status", "discovered"),
            "status": status,
        })
    total_verified = sum(c["verified_questions"] for c in cells if isinstance(c["verified_questions"], int))
    return {
        "cells": cells,
        "totals": {
            "verified_records": total_verified,
            "sources_parsed": sum(1 for s in registry if s.get("parser_status") == "parsed"),
            "sources_discovered": sum(1 for s in registry if s.get("parser_status") != "parsed"),
        },
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
