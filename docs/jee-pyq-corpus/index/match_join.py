"""Candidate-official join: match external candidates against the Official
Question Index by fingerprint, then similarity.

Statuses: MATCHED (exact fingerprint) / AMBIGUOUS (high similarity, needs
review) / UNMATCHED. AMBIGUOUS never auto-verifies. External metadata is
never trusted: the official record supplies all exam facts.

Usage:
    python index/match_join.py --candidates <dir with cand-*.jsonl> --index <official_index.json>
"""
import difflib
import glob
import json
import os
import sys

INDEX_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(INDEX_DIR, "..", "discovery"))
from import_candidates import normalize_text  # noqa: E402

SIMILARITY_THRESHOLD = 0.85


def _token_overlap(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _load_rows(rows_dirs: list[str]) -> dict[str, str]:
    """Map 'filename:row' -> full question text from source jsonl files."""
    texts: dict[str, str] = {}
    for directory in rows_dirs:
        for name in os.listdir(directory):
            if not (name.endswith(".jsonl") and os.path.isfile(os.path.join(directory, name))):
                continue
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if line.strip():
                        try:
                            texts[f"{name}:{i}"] = json.loads(line).get("question", "")
                        except (json.JSONDecodeError, AttributeError):
                            continue
    return texts


def match_candidates(candidates: list[dict], official: list[dict], row_texts: dict) -> dict:
    """Join candidates to official records. Returns a benchmark report."""
    by_hash: dict[str, list] = {}
    for record in official:
        by_hash.setdefault(record["normalized_text_hash"], []).append(record)
    results = {"MATCHED": [], "AMBIGUOUS": [], "UNMATCHED": []}
    for candidate in candidates:
        fp = candidate.get("question_text_hash")
        hits = by_hash.get(fp, [])
        if len(hits) == 1:
            results["MATCHED"].append(
                {"candidate_id": candidate["candidate_id"], "official": hits[0]}
            )
            continue
        if len(hits) > 1:
            results["AMBIGUOUS"].append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": f"{len(hits)} official records share the fingerprint",
                    "official": hits,
                }
            )
            continue
        # Near-match pass over the same subject only.
        norm = normalize_text(row_texts.get(candidate.get("source_record_id", ""), ""))
        best: list = []
        for record in official:
            if record["subject"] != candidate.get("subject"):
                continue
            sim = _token_overlap(norm, normalize_text(record.get("stem_text", "")))
            if sim >= SIMILARITY_THRESHOLD:
                best.append((sim, record))
        if best:
            best.sort(reverse=True)
            results["AMBIGUOUS"].append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": f"near-match similarity {best[0][0]:.2f}, needs review",
                    "official": [best[0][1]],
                }
            )
        else:
            results["UNMATCHED"].append({"candidate_id": candidate["candidate_id"]})
    return results


def main(argv: list[str]) -> int:
    candidates: list[dict] = []
    skip_next = False
    for pattern in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if pattern in ("--index", "--rows"):
            skip_next = True
            continue
        if pattern.startswith("--"):
            continue
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        candidates.append(json.loads(line))
    index_path = argv[argv.index("--index") + 1] if "--index" in argv else os.path.join(
        INDEX_DIR, "official_index.json"
    )
    with open(index_path, encoding="utf-8") as fh:
        official = json.load(fh)
    row_texts: dict[str, str] = {}
    if "--rows" in argv:
        row_texts = _load_rows(argv[argv.index("--rows") + 1].split(","))
    report = match_candidates(candidates, official, row_texts)
    summary = {k: len(v) for k, v in report.items()}
    print(json.dumps({"summary": summary, "total_candidates": len(candidates),
                      "official_records": len(official)}, indent=1))
    out = os.path.join(INDEX_DIR, "join_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
