"""Read-only retrieval helpers for the verified JEE PYQ corpus.

Every fact returned comes from a stored record. Counts are computed from the
corpus on every call — never cached claims, never model memory.
"""

import json
from pathlib import Path
from typing import cast

CORPUS_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = CORPUS_DIR / "data"


def _load_records() -> list[dict[str, object]]:
    """Load verified PYQ metadata records from every *-pyq.json shard."""
    records: list[dict[str, object]] = []
    for path in sorted(DATA_DIR.glob("*-pyq.json")):
        with path.open(encoding="utf-8") as file:
            records.extend(cast(list[dict[str, object]], json.load(file)))
    return records


def get_by_concept(concept_id: str) -> list[dict[str, object]]:
    """Return verified records mapped to ``concept_id`` in dataset order."""
    return [
        record
        for record in _load_records()
        if concept_id in cast(list[str], record.get("concept_ids", []))
    ]


def get_by_chapter(chapter: str) -> list[dict[str, object]]:
    """Return verified records for a chapter (exact match)."""
    return [
        record
        for record in _load_records()
        if record.get("chapter") == chapter
    ]


def get_by_year(year: int) -> list[dict[str, object]]:
    """Return verified records for an exam year."""
    return [
        record
        for record in _load_records()
        if record.get("year") == year
    ]


def count_by_concept(concept_id: str) -> int:
    """Reproducible count of verified records for a concept."""
    return len(get_by_concept(concept_id))


def get_by_shift(exam: str, year: int, session: str | None, shift: str | None) -> list[dict[str, object]]:
    """Return verified records for one exact exam sitting."""
    return [
        record
        for record in _load_records()
        if (
            record.get("exam") == exam
            and record.get("year") == year
            and record.get("session") == session
            and record.get("shift") == shift
        )
    ]


def explain_match(concept_id: str, record: dict[str, object]) -> list[str]:
    """Human-readable reasons why a record matches a concept.

    Only cites stored fields. Returns [] when the record is not mapped.
    """
    if concept_id not in cast(list[str], record.get("concept_ids", [])):
        return []
    reasons = [
        f"concept = {concept_id} (mapped, primary)"
        if cast(list[str], record.get("concept_ids", []))[:1] == [concept_id]
        else f"concept = {concept_id} (mapped)",
        f"chapter = {record.get('chapter')}",
        f"question type = {record.get('question_type')}",
    ]
    note = record.get("concept_mapping_note")
    if note:
        reasons.append(f"mapping note: {note}")
    return reasons
