"""Bulk discovery importer: external structured datasets -> QuestionCandidates.

Candidates are UNVERIFIED by construction. Exam-history facts (year, shift,
marks, answers, difficulty) from third-party datasets are NEVER promoted
without official-source reconciliation.

Usage:
    python discovery/import_candidates.py <input.jsonl> --dataset eQOURSE-jee-main \
        --out discovery/candidates.jsonl
Input: eQOURSE-style rows (question, options, subject, topic, subtopic,
exam, source_paper, difficulty, question_type, correct_option).
"""
import hashlib
import json
import os
import re
import sys

DISCOVERY_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, drop LaTeX noise for fingerprinting."""
    text = text.casefold()
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\\([a-z]+)", r" \1 ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(normalized: str, subject: str) -> str:
    return hashlib.sha256(f"{subject}|{normalized}".encode("utf-8")).hexdigest()[:24]


def taxonomy_premap(text: str, topic: str, subtopic: str) -> list[dict]:
    """Cheap first-pass classification against the canonical taxonomy.

    Returns [{concept_id, confidence, reasons}] sorted by confidence.
    Confidence is a coarse tier ('high' 3+ signals, 'medium' 2, 'low' 1),
    never a calibrated probability. Ambiguous output is expected input
    for human/LLM review — not a verdict.
    """
    sys.path.insert(0, os.path.join(DISCOVERY_DIR, "..", "..", "..", "backend"))
    try:
        from app.services import jee_service
    finally:
        sys.path.pop(0)
    terms = [topic, subtopic]
    scored = []
    for concept in jee_service._concepts():
        score, evidence = jee_service._score_concept(concept, terms + [text])
        if score > 0:
            scored.append((score, concept, evidence))
    scored.sort(key=lambda item: item[0], reverse=True)
    out = []
    for score, concept, evidence in scored[:3]:
        tier = "high" if score >= 3 else "medium" if score == 2 else "low"
        out.append(
            {
                "concept_id": concept["concept_id"],
                "confidence": tier,
                "reasons": evidence[:4],
            }
        )
    return out


def import_row(row: dict, dataset: str, record_id: str) -> dict:
    question = str(row.get("question", ""))
    normalized = normalize_text(question)
    return {
        "candidate_id": f"cand-{dataset}-{record_id}",
        "source_dataset": dataset,
        "source_record_id": record_id,
        "source_file": os.path.basename(record_id.split(":")[0]) if ":" in record_id else "",
        "question_text_hash": fingerprint(normalized, str(row.get("subject", ""))),
        "normalized_text_hash": fingerprint(normalized, "generic"),
        "subject": row.get("subject"),
        "topic": row.get("topic"),
        "subtopic": row.get("subtopic"),
        "exam": row.get("exam"),
        "year": None,  # NOT present in source; only via official reconciliation
        "session": None,
        "shift": None,
        "source_paper": row.get("source_paper"),
        "correct_answer": None,  # dataset answers are unverified claims
        "difficulty": None,  # dataset labels are inconsistent; never ingested
        "source_reference": row.get("source_paper"),
        "license_status": "B-discovery-only",
        "provenance_status": "unverified-candidate",
        "question_preview": question[:200],
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: import_candidates.py <input.jsonl> --dataset NAME [--out PATH] [--premap]")
        return 2
    dataset = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "external"
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(
        DISCOVERY_DIR, "candidates.jsonl"
    )
    do_premap = "--premap" in argv
    candidates = []
    src_name = os.path.basename(argv[1])
    with open(argv[1], encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            record = import_row(row, dataset, f"{src_name}:{i}")
            if do_premap:
                record["taxonomy_candidates"] = taxonomy_premap(
                    str(row.get("question", "")),
                    str(row.get("topic", "")),
                    str(row.get("subtopic", "")),
                )
            candidates.append(record)
    # Dedupe by fingerprint: same question text = one canonical candidate.
    seen: dict[str, dict] = {}
    for record in candidates:
        key = record["question_text_hash"]
        if key not in seen:
            seen[key] = record
        else:
            seen[key].setdefault("duplicate_of", []).append(record["candidate_id"])
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in seen.values():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"rows={len(candidates)} unique={len(seen)} dupes={len(candidates) - len(seen)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
