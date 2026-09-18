"""Validate the Rotational Motion corpus against its JSON Schema."""

import json
from pathlib import Path
from typing import cast

import jsonschema
import pytest


CORPUS_DIR: Path = Path(__file__).resolve().parents[1]
SCHEMA_PATH: Path = CORPUS_DIR / "schema" / "syllabus-entry.schema.json"
DATA_PATH: Path = CORPUS_DIR / "data" / "jee-main-physics-rotational-motion.json"


def load_json(path: Path) -> object:
    """Load a JSON document from a path."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_schema() -> dict[str, object]:
    """Load the corpus JSON Schema."""
    return cast(dict[str, object], load_json(SCHEMA_PATH))


def load_entries() -> list[dict[str, object]]:
    """Load corpus entries from the data file."""
    return cast(list[dict[str, object]], load_json(DATA_PATH))


def test_every_entry_matches_schema() -> None:
    """Validate each corpus entry, including URI and date formats."""
    schema: dict[str, object] = load_schema()
    validator: jsonschema.Draft7Validator = jsonschema.Draft7Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )

    for entry in load_entries():
        errors: list[jsonschema.ValidationError] = sorted(
            validator.iter_errors(entry), key=lambda error: list(error.path)
        )
        assert not errors, "; ".join(error.message for error in errors)


def test_all_ids_are_unique() -> None:
    """Reject duplicate entry identifiers."""
    entries: list[dict[str, object]] = load_entries()
    ids: list[object] = [entry.get("id") for entry in entries]

    assert len(ids) == len(set(ids))


def test_data_file_has_expected_entry_count() -> None:
    """Ensure the requested fifteen contributor-derived subtopics are present."""
    assert len(load_entries()) == 15


@pytest.mark.parametrize("field", ["id", "exam", "subject", "chapter", "subtopic"])
def test_schema_rejects_missing_required_fields(field: str) -> None:
    """Ensure required fields cannot be omitted from an entry."""
    schema: dict[str, object] = load_schema()
    validator: jsonschema.Draft7Validator = jsonschema.Draft7Validator(schema)
    entry: dict[str, object] = dict(load_entries()[0])
    del entry[field]

    assert not validator.is_valid(entry)


@pytest.mark.parametrize(
    ("field", "value"),
    [("exam", "JEE Advanced Physics"), ("verification_status", "pending")],
)
def test_schema_rejects_invalid_enum_values(field: str, value: str) -> None:
    """Ensure fixed-category fields accept only their declared enum values."""
    schema: dict[str, object] = load_schema()
    validator: jsonschema.Draft7Validator = jsonschema.Draft7Validator(schema)
    entry: dict[str, object] = dict(load_entries()[0])
    entry[field] = value

    assert not validator.is_valid(entry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_url", 42),
        ("source_last_updated", "2026/07/14"),
        ("verification_date", "18-09-2026"),
    ],
)
def test_schema_rejects_malformed_formats(field: str, value: object) -> None:
    """Ensure malformed URI and date values are rejected."""
    schema: dict[str, object] = load_schema()
    validator: jsonschema.Draft7Validator = jsonschema.Draft7Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    entry: dict[str, object] = dict(load_entries()[0])
    entry[field] = value

    assert not validator.is_valid(entry)
