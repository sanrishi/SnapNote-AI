# JEE Corpus Pipeline — architecture audit (2026-10-06)

End-to-end flow as built, with the bottleneck each stage hits at 10k+ scale.

## Stages

```
1. SOURCE DISCOVERY (manual web research)
   registry.json: exam/year/session/shift, official URL or login-walled note,
   mirror URL, pdf_sha256, parser_status, verification_status
   -> bottleneck at scale: manual per-paper discovery. Needs adapter pattern.

2. EXTRACTION
   - extract_paper.py: NTA PDF text-structure + OCR stems (2026 direct PDFs)
   - build_transcribed_index.py: third-party HTML mirror pages (2024-2025)
   - build_index.py: OCR physics slices -> official_index (local, gitignored)
   -> bottleneck: one adapter per source class; no shared record schema yet.

3. NORMALIZATION + FINGERPRINT
   discovery/import_candidates.normalize_text (shared by importer, index, join)
   normalized_text_hash = sha256(subject | normalized)[0:24]
   content_fingerprint  = sha256(question_id + source_url)[0:16]
   -> shared primitive. GOOD — no change needed.

4. CANDIDATE GENERATION
   - import_candidates.py: external datasets -> quarantined candidates
   - triage.py: OCR page pre-filter (5x reduction, recall-validated)
   - mirror topic_tag filter (rotational: 50 from 2,829)
   -> bottleneck: rotational-only vocabulary; topic_tag trust unmeasured.

5. TRIAGE (prescreen_queue.py)
   integrity / duplicates / metadata / concept-signal / provenance
   -> classes A/B/C/D. Deterministic, tested. Needs generalization to chapters.

6. VERIFICATION QUEUES (verification_queue.json + review_log.json)
   batch-scoped human review; per-record recomputation; answer discipline.
   -> bottleneck at scale: HUMAN/SOURCE REVIEW throughput. The factory must
      maximize clean-candidate precision so review converts at ~95%+.

7. VERIFIED PERSISTENCE (data/jee-main-physics-rotational-pyq.json)
   schema-validated QuestionRecords, provenance + fingerprints.
   -> single-chapter file; needs per-chapter sharding at scale.

8. AUDIT (coverage/audit_corpus.py, wired into pytest via test_audit.py)
   duplicates, provenance honesty, shift coherence, answer formats.
   -> GOOD; extend per new source classes.

9. RETRIEVAL (/api/jee via jee_service.py + queries/pyq_queries.py)
   concept -> records; growth-proofed tests.
   -> needs chapter-level fallback when sub-concept taxonomy is missing.

10. REPORTING (coverage/build_matrix.py -> matrix.json index_dashboard)
    indexed / queued / verified / yield / concept distribution.
    -> extend per subject/chapter/source-class.
```

## What the factory run added (2026-10-06)

- `jee-chapter-taxonomy/`: 53 chapter concepts derived from the official NTA
  syllabus units + a 62-label mirror-topic map (41 exact, 21 explicit
  special-cases; unmapped labels fail loudly).
- `index/build_candidate_queues.py`: all 2,829 indexed questions queued across
  53 chapters with triage-lite states; zero unmapped.
- Multi-shard persistence: `data/*-pyq.json` shards; queries, matrix, audit,
  and `/api/jee` aggregate shards (deduplicated by question_id).
- Chapter-level product loop proven: "Current Electricity" resolves to its
  chapter concept and serves 15 verified PYQs.
- Pilot: Current Electricity 15 reviewed -> 15 verified (73 total verified).
- `ingestion/source_classes.py`: OFFICIAL / THIRD_PARTY_TRANSCRIPTION /
  DISCOVERY_ONLY, enforced by the audit.
- Recall discipline: a taxonomy sweep over all 929 Physics stems found only
  other-chapter hits — queues have measured precision, not assumed recall.

## Bottlenecks ranked (10k+ lens)

1. Trusted candidate acquisition — source universe size, not tooling.
2. Verification throughput — human review is the scarce resource; triage
   precision and recomputation automation are the multipliers.
3. Taxonomy coverage — verified mapping needs concepts; chapter-level
   derivation from the syllabus corpus unblocks non-rotational chapters.
4. Manual per-source scripts — adapter interface + shared record schema.
5. Single-file corpus + single-chapter queries — shard by subject/chapter.

## Non-goals (explicit)

- No Advanced ingestion until Main yield is measured per chapter.
- No Chemistry/Maths verification before chapter taxonomy exists.
- No unverified candidates counted as verified, ever.
```

