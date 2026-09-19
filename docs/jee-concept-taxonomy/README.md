# JEE Physics concept taxonomy: Rotational Motion

This directory is **Layer B (taxonomy)** for JEE Main Physics Rotational
Motion. It is built on top of **Layer A (#29's syllabus corpus)** in
`docs/jee-syllabus-corpus/`. The taxonomy gives the fifteen Layer A subtopics
stable concept IDs and learning relationships without changing the source
syllabus corpus.

## Files

- `schema/concept-entry.schema.json` defines the concept-entry contract.
- `data/jee-physics-rotational-motion-taxonomy.json` contains exactly 15 concepts.
- `tests/test_taxonomy_validation.py` checks schema validity, references,
  aliases, and Layer A subtopic grounding.
- `queries/taxonomy_queries.py` provides alias, child, and prerequisite lookups.

Each concept records its display name, aliases, parent and children,
prerequisites, related concepts, and the exact `syllabus_subtopic_ref` from the
#29 corpus. References are IDs, so consumers do not need to depend on display
names.

## Open taxonomy flag: moment of a force vs torque

Concept 004, **Moment of a force**, and concept 005, **Torque**, are retained as
separate nodes because both labels occur in the Layer A corpus. Physically they
describe the same quantity in this context. Both entries carry an explicit
`note` marking this as an **open question for sanrishi to resolve**. This is
deliberate review metadata, not a silent merge or an unexamined duplication.

## Run validation

From the repository root:

```bash
python -m pytest docs/jee-concept-taxonomy/tests
```
