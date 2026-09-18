"""Small read-only lookup helpers for the JEE syllabus corpus."""

import json
from pathlib import Path
from typing import cast


CORPUS_DIR: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = CORPUS_DIR / "data" / "jee-main-physics-rotational-motion.json"


def _load_entries() -> list[dict[str, object]]:
    """Load all corpus entries."""
    with DATA_PATH.open(encoding="utf-8") as file:
        return cast(list[dict[str, object]], json.load(file))


def get_by_subtopic(name: str) -> list[dict[str, object]]:
    """Return entries whose subtopic exactly matches ``name``."""
    return [entry for entry in _load_entries() if entry.get("subtopic") == name]


def get_all_in_chapter(chapter: str) -> list[dict[str, object]]:
    """Return all entries belonging to ``chapter``."""
    return [entry for entry in _load_entries() if entry.get("chapter") == chapter]


def get_prerequisites(subtopic_id: str) -> list[str]:
    """Return prerequisite concepts for an entry ID, or an empty list if absent."""
    for entry in _load_entries():
        if entry.get("id") == subtopic_id:
            prerequisites: object = entry.get("prerequisite_concepts", [])
            return cast(list[str], prerequisites)
    return []
