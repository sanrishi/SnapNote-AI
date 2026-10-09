# JEE PYQ Corpus — Rotational Motion (issue #31)

Verified PYQ **metadata + provenance** for JEE Physics → Rotational Motion.
Stores facts about officially published questions. Never stores question text,
options, images, or answer keys.

## What exists now

- `schema/pyq-record.schema.json` — canonical QuestionRecord (nullable
  difficulty/answer_key; provenance + content fingerprint).
- `data/jee-main-physics-rotational-pyq.json` — **9 verified records**, each
  visually read from official NTA 2026 papers (Apr 2 S1/S2, Apr 4 S1/S2,
  Apr 5 S2, Apr 8 S2).
- `index/build_transcribed_index.py` — third-party transcription indexer
  (JEEPrep HTML mirror); raw stems stay local/gitignored, only
  `index/transcribed_index_summary.json` (structure + counts) is committed.
- `index/verification_queue.json` — 50 rotational-tagged 2024–2025 index
  entries awaiting human review (hashes + provenance, no stems).
- `sources/registry.json` — 10 known sources: 9 NTA 2026 shifts (3 parsed,
  6 discovered, 1 retrieval-blocked) + the Advanced 2007–2025 archive.
- `ingestion/extract_paper.py` — PDF → candidate structure (75/75 questions
  on both parsed papers). Candidates are `needs-review`, never verified.
- `coverage/build_matrix.py` + `coverage/matrix.json` — recomputed coverage;
  status is never `complete` (UI must say less, not imply all).
- `queries/pyq_queries.py` — by concept/chapter/year/shift, reproducible
  counts, `explain_match` reasons citing stored fields only.

## Honest coverage (recomputed, 2026-10-06)

73 verified records (9 from 2026 direct NTA PDFs + 49 rotational + 15 Current
Electricity pilot across 2024 shifts) · 42 sources parsed ·
3 undiscovered/blocked · 0 complete.
A taxonomy recall sweep over all 929 indexed Physics stems found 22 extra
keyword hits, all correctly other-chapter — the 50-queue captured the
rotational population.
2026 official direct-PDF index: 7 shifts, 175 Physics questions (OCR).
2024–2025 transcribed-official index: 35 papers, 2,829 questions
(2024: 1,620; 2025: 1,209; Physics/Chemistry/Mathematics), via a third-party
transcription of official NTA PDFs that claims official final-answer alignment.
Direct 2024–2025 NTA paper PDFs are login-walled, so these are indexed,
unreviewed mirrors — never verified records.

## Index dashboard

- official papers indexed: 35 (2024–2025) + 7 parsed 2026 shifts
- official questions indexed: 2,829 (2024–2025) + 175 (2026 Physics OCR)
- source-linked review queue: 50 rotational-tagged 2024–2025 entries
- verified questions: 73 (rotational 49/50 reviewed; Current Electricity
  pilot 15/15; numerical answers independently recomputed wherever the mirror
  states one; audit clean)
- corpus factory: 2,829 indexed → 53 chapter queues (2,776 non-duplicate
  candidates) → chapter taxonomy (53) → multi-shard verified persistence →
  aggregating retrieval (`/api/jee` serves chapter concepts)
- queue reduction is 2,829 indexed → 50 candidates = **56.6×** (candidates are
  ~1.77% of the index), not 17.6×
- external candidates quarantined: 703 (eQOURSE mocks: 0 matches vs 175, and
  0 matches vs 2,829 — max similarity 0.72, threshold 0.85; disjoint cohorts)

## Verification pipeline

raw official PDF → parsed candidate → human visual read → taxonomy mapping →
validation tests → `verified` record. No record becomes verified by model
judgment. Difficulty/negative-marking/frequency are never stored.

## Growing the corpus

Parse a discovered source (`ingestion/`), visually verify rotational hits,
append records, rebuild the matrix. Scale targets: Milestone A 50–100
verified → Milestone B full recent-shift chapter coverage → C Physics range →
D Chemistry+Maths → E Advanced.
