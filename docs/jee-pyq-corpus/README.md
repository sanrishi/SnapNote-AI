# JEE PYQ Corpus — Rotational Motion (issue #31)

Verified PYQ **metadata + provenance** for JEE Physics → Rotational Motion.
Stores facts about officially published questions. Never stores question text,
options, images, or answer keys.

## What exists now

- `schema/pyq-record.schema.json` — canonical QuestionRecord (nullable
  difficulty/answer_key; provenance + content fingerprint).
- `data/jee-main-physics-rotational-pyq.json` — **9 verified records**, each
  visually read from official NTA 2026 papers (2 Apr S1 Q29/Q31, 4 Apr S1 Q30).
- `sources/registry.json` — 10 known sources: 9 NTA 2026 shifts (3 parsed,
  6 discovered, 1 retrieval-blocked) + the Advanced 2007–2025 archive.
- `ingestion/extract_paper.py` — PDF → candidate structure (75/75 questions
  on both parsed papers). Candidates are `needs-review`, never verified.
- `coverage/build_matrix.py` + `coverage/matrix.json` — recomputed coverage;
  status is never `complete` (UI must say less, not imply all).
- `queries/pyq_queries.py` — by concept/chapter/year/shift, reproducible
  counts, `explain_match` reasons citing stored fields only.

## Honest coverage (recomputed, 2026-09-30)

3 verified records · 3 sources parsed · 7 discovered · 0 complete.
5th Apr S1 scanned end-to-end with **zero** rotational hits — shifts vary,
which is itself a finding: per-shift coverage must be measured, not assumed.

## Verification pipeline

raw official PDF → parsed candidate → human visual read → taxonomy mapping →
validation tests → `verified` record. No record becomes verified by model
judgment. Difficulty/negative-marking/frequency are never stored.

## Growing the corpus

Parse a discovered source (`ingestion/`), visually verify rotational hits,
append records, rebuild the matrix. Scale targets: Milestone A 50–100
verified → Milestone B full recent-shift chapter coverage → C Physics range →
D Chemistry+Maths → E Advanced.
