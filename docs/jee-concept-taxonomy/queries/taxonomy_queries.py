"""Read-only lookup helpers for the Rotational Motion concept taxonomy."""

import json
from pathlib import Path
from typing import cast


TAXONOMY_DIR: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = TAXONOMY_DIR / "data" / "jee-physics-rotational-motion-taxonomy.json"


def _load_concepts() -> list[dict[str, object]]:
    """Load all taxonomy concepts."""
    with DATA_PATH.open(encoding="utf-8") as file:
        return cast(list[dict[str, object]], json.load(file))


def get_concept_by_alias(alias: str) -> dict[str, object] | None:
    """Return the concept whose alias exactly matches ``alias``."""
    normalized_alias: str = alias.casefold()
    for concept in _load_concepts():
        aliases: list[str] = cast(list[str], concept.get("aliases", []))
        if any(candidate.casefold() == normalized_alias for candidate in aliases):
            return concept
    return None


def get_children(concept_id: str) -> list[dict[str, object]]:
    """Return direct child concepts for ``concept_id`` in dataset order."""
    return [
        concept
        for concept in _load_concepts()
        if concept.get("parent_concept_id") == concept_id
    ]


def get_prerequisites(concept_id: str) -> list[dict[str, object]]:
    """Return prerequisite concepts for ``concept_id`` in declared order."""
    concepts: list[dict[str, object]] = _load_concepts()
    by_id: dict[str, dict[str, object]] = {
        cast(str, concept["concept_id"]): concept for concept in concepts
    }
    target: dict[str, object] | None = by_id.get(concept_id)
    if target is None:
        return []
    prerequisite_ids: list[str] = cast(
        list[str], target.get("prerequisite_concept_ids", [])
    )
    return [by_id[prerequisite_id] for prerequisite_id in prerequisite_ids]
