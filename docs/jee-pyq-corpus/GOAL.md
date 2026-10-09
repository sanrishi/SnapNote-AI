# JEE Intelligence — Charter (locked)

> 10k is NOT the finish line. Historical coverage and all meaningful JEE
> question patterns are the goal. This file is the decision record; any
> agent working this corpus must follow it.

## Product goal

Build a comprehensive, verified JEE historical knowledge base covering the
meaningful ways JEE has actually tested concepts, so a student does not
miss important question patterns.

The earlier "10,000+" number was only a rough scale target. Do NOT optimize
for reaching exactly 10,000. The final corpus could be below or far above
10k depending on the real historical question universe.

## Completion criterion

> Historical coverage + concept coverage + question-pattern coverage +
> trustworthy verification.

Success means: SnapNote has sufficiently broad, verified historical JEE
coverage that a student is unlikely to miss an important way a concept has
actually been tested. Never report success on row counts alone
(600/600 2026, 5k, 10k — milestones, not completion).

## Pipeline (factory, not manual hunting)

source discovery -> bulk ingestion -> normalization -> dedupe ->
taxonomy mapping -> candidate generation -> prescreen -> verification ->
audit -> verified corpus.

Feed the factory with larger historical source universes. Priority order:
2025 gaps, 2024 gaps, 2023, 2022, older years where source quality and
provenance justify the effort.

## Provenance (never silently promote)

Every source carries an explicit class: `OFFICIAL` /
`THIRD_PARTY_TRANSCRIPTION` / `DISCOVERY_ONLY` (plus `PUBLIC_ARCHIVE`
when that adapter lands). Third-party transcription is never official.

## Trust standard (unchanged)

- Never fabricate year/session/shift/number/answer/marks/difficulty.
- Numericals: independent recomputation whenever possible.
- MCQs: `answer_key: null` when independent confirmation is unavailable.
- Reject/demote on mirror-vs-recompute mismatch or truncated stems.
- Raw stems local/gitignored; only hashes/structure committed.

## Pattern intelligence (just-in-time)

Chapter counts are not enough. Target shape:

Chapter -> Sub-concept -> Question pattern / archetype -> Verified PYQs.

Process source material -> observe recurring concepts/patterns ->
formalize taxonomy -> validate. Never invent archetypes without evidence.
Do not let taxonomy work block source expansion.

## Reporting (coverage, not just count)

Every expansion reports: sources discovered / accepted / rejected,
indexed / unique / candidate / verified / rejected-unresolved questions,
verified by subject/chapter/sub-concept, duplicates, new patterns
discovered, remaining historical gaps.

## Product connection

Student screenshot -> StudyNotes -> canonical concept -> JEE syllabus
context -> relevant verified PYQs -> question-pattern coverage -> student
practice. The corpus is the intelligence layer.

## Execution discipline

When one year/source family is exhausted, move to the next. Blocked
sources get recorded, not forced. Standards never drop for numbers.
Sibling tracks (sub-concept taxonomy, corpus QA, pattern intelligence,
SnapNote integration) run incrementally alongside sourcing.
