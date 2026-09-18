# JEE syllabus corpus: Rotational Motion

This folder contains a small, machine-readable corpus for **JEE Main Physics,
Unit 5: Rotational Motion**. It covers one chapter only and is based on the
official NTA syllabus source recorded in each data entry.

## Files

- `schema/syllabus-entry.schema.json` defines the JSON Schema Draft 7 shape.
- `data/jee-main-physics-rotational-motion.json` contains the 15 corpus entries.
- `tests/test_schema_validation.py` validates every entry and checks duplicate IDs,
  enum values, required fields, URI formats, and date formats.
- `queries/sample_queries.py` provides small read-only lookup helpers.

## Schema fields

Each entry records its `id`, `exam`, `subject`, `chapter`, and `subtopic`, plus
optional `subtopic_segmentation` and `prerequisite_concepts` metadata. It also
records whether the topic is in the syllabus (`in_syllabus`), the official
source and publisher, source update and verification dates, and the
`verification_status`.

## Segmentation caveat

The official source presents Rotational Motion as continuous prose. The 15
subtopics in this corpus are a contributor-derived breakdown of that prose;
they are not verbatim official bullet points. Accordingly, every entry uses
`subtopic_segmentation: "derived_by_contributor"`.

## Run the tests

From the repository root, install the test dependency if needed and run:

```bash
python -m pytest docs/jee-syllabus-corpus/tests
```

The tests use `jsonschema` with a format checker so malformed URIs and dates
are rejected in addition to missing fields and invalid enum values.
