"""Validate the Layer B concept taxonomy and its Layer A references."""

import json
from pathlib import Path
from typing import cast

import jsonschema
import pytest


TAXONOMY_DIR: Path = Path(__file__).resolve().parents[1]
SCHEMA_PATH: Path = TAXONOMY_DIR / "schema" / "concept-entry.schema.json"
DATA_PATH: Path = TAXONOMY_DIR / "data" / "jee-physics-rotational-motion-taxonomy.json"
CORPUS_PATH: Path = (
    TAXONOMY_DIR.parent / "jee-syllabus-corpus" / "data" / "jee-main-physics-rotational-motion.json"
)


def load_json(path: Path) -> object:
    """Load a JSON document."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_schema() -> dict[str, object]:
    """Load the taxonomy JSON Schema."""
    return cast(dict[str, object], load_json(SCHEMA_PATH))


def load_concepts() -> list[dict[str, object]]:
    """Load taxonomy concepts."""
    return cast(list[dict[str, object]], load_json(DATA_PATH))


def test_every_concept_matches_schema() -> None:
    """Validate every concept against the Draft 7 schema."""
    validator: jsonschema.Draft7Validator = jsonschema.Draft7Validator(load_schema())
    for concept in load_concepts():
        errors: list[jsonschema.ValidationError] = sorted(
            validator.iter_errors(concept), key=lambda error: list(error.path)
        )
        assert not errors, "; ".join(error.message for error in errors)


def test_dataset_has_exactly_fifteen_unique_concepts() -> None:
    """Keep the requested canonical slice at exactly fifteen nodes."""
    concepts: list[dict[str, object]] = load_concepts()
    ids: list[object] = [concept.get("concept_id") for concept in concepts]
    assert len(concepts) == 15
    assert len(ids) == len(set(ids))


def test_all_references_are_existing_ids() -> None:
    """Reject dangling parent, child, prerequisite, or related references."""
    concepts: list[dict[str, object]] = load_concepts()
    ids: set[str] = {cast(str, concept["concept_id"]) for concept in concepts}
    reference_fields: tuple[str, ...] = (
        "child_concept_ids",
        "prerequisite_concept_ids",
        "related_concept_ids",
    )
    for concept in concepts:
        parent_id: object = concept["parent_concept_id"]
        if parent_id is not None:
            assert parent_id in ids
        for field in reference_fields:
            references: list[str] = cast(list[str], concept[field])
            assert set(references) <= ids


def test_aliases_resolve_to_one_concept() -> None:
    """Every alias must resolve exactly once, case-insensitively."""
    concepts: list[dict[str, object]] = load_concepts()
    resolutions: dict[str, list[str]] = {}
    for concept in concepts:
        concept_id: str = cast(str, concept["concept_id"])
        for alias in cast(list[str], concept["aliases"]):
            resolutions.setdefault(alias.casefold(), []).append(concept_id)
    assert resolutions
    assert all(len(concept_ids) == 1 for concept_ids in resolutions.values())


def test_syllabus_references_match_layer_a_corpus() -> None:
    """Every taxonomy reference must match a real #29 corpus subtopic."""
    corpus: list[dict[str, object]] = cast(list[dict[str, object]], load_json(CORPUS_PATH))
    subtopics: set[str] = {cast(str, entry["subtopic"]) for entry in corpus}
    references: set[str] = {
        cast(str, concept["syllabus_subtopic_ref"]) for concept in load_concepts()
    }
    assert references <= subtopics
    assert len(references) == 15


@pytest.mark.parametrize("concept_number", ["004", "005"])
def test_moment_torque_nodes_are_flagged_for_review(concept_number: str) -> None:
    """Keep the known physical equivalence visible until taxonomy review."""
    concept_id: str = f"jee-physics-rotational-motion-concept-{concept_number}"
    concept: dict[str, object] = next(
        item for item in load_concepts() if item["concept_id"] == concept_id
    )
    assert "open question" in cast(str, concept["note"]).casefold()
