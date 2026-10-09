"""JEE concept mapping service (issue #32 vertical slice).

StudyNotes topic + key terms -> canonical taxonomy concept -> official
syllabus node -> verified PYQ records -> JEEConceptContext.

Trust rules (non-negotiable):
- The corpora are the ONLY factual source. Nothing here invents years,
  shifts, marks, frequencies, or weightage.
- Matching is deterministic (exact display-name / alias / token match with
  fixed scoring). No LLM calls, no probabilities-as-facts.
- Uncertain input returns 'ambiguous' (tie) or 'unresolved' (no hit) —
  never a forced nearest node.
"""
import json
import logging
import os
import re
from pathlib import Path

from app.models.schemas import (
    JEEConceptContext,
    JEEConceptRef,
    JEEPatternRef,
    JEEPyQRef,
    JEESyllabusRef,
)

logger = logging.getLogger(__name__)

# Cap the inline PYQ list so one popular chapter cannot bloat the response.
# pyq_count always reports the full total; pyqs_truncated says more exist.
MAX_INLINE_PYQS = 25


def _source_class(record: dict) -> str:
    """Provenance class derived from stored evidence, never assumed.

    OFFICIAL only for NTA-hosted documents; everything else stays
    THIRD_PARTY_TRANSCRIPTION (mirrors are never silently promoted).
    """
    url = str(record.get("source_url", ""))
    document = str(record.get("source_document", ""))
    if "third-party transcription" in document:
        return "THIRD_PARTY_TRANSCRIPTION"
    if "cdnbbsr.s3waas.gov.in" in url or "nta.ac.in" in url:
        return "OFFICIAL"
    return "THIRD_PARTY_TRANSCRIPTION"


def _patterns_for(chapter_id: str, valid_ids: set[str]) -> list[JEEPatternRef]:
    """Evidence-backed patterns for one chapter.

    Only patterns whose every referenced question exists in the loaded
    corpus are published; anything else stays out of the response.
    """
    found: list[JEEPatternRef] = []
    for root in _candidate_roots():
        path = root / "jee-question-patterns.json"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            logger.warning("JEE patterns file unreadable: %s", path)
            continue
        if isinstance(data, list):
            for pattern in data:
                if pattern.get("chapter_id") != chapter_id:
                    continue
                question_ids = [str(q) for q in pattern.get("question_ids", [])]
                if not question_ids or any(q not in valid_ids for q in question_ids):
                    continue
                found.append(
                    JEEPatternRef(
                        pattern_id=str(pattern.get("pattern_id", "")),
                        name=str(pattern.get("name", "")),
                        sub_concept=str(pattern.get("sub_concept", "")),
                        method=str(pattern.get("method", "")),
                        question_ids=question_ids,
                        coverage_note=str(pattern.get("coverage_note", "")),
                    )
                )
        break
    return found

_SLICE_FILES = {
    # Canonical docs/ filenames first, then the flattened Docker image names.
    # *-pyq.json shards aggregate (sorted, deduplicated by question_id).
    "taxonomy": [
        "jee-physics-rotational-motion-taxonomy.json",
        "jee-chapters.json",
        "taxonomy.json",
    ],
    "syllabus": [
        "jee-main-physics-rotational-motion.json",
        "syllabus.json",
    ],
    "pyq": [
        "jee-main-physics-rotational-pyq.json",
        "pyq.json",
        "*-pyq.json",  # shard glob: every verified chapter shard aggregates
    ],
}


def _candidate_roots() -> list[Path]:
    override = os.environ.get("JEE_DATA_DIR", "").strip()
    roots = [Path(override)] if override else []
    roots.append(Path("/app/jee-data"))
    here = Path(__file__).resolve()
    repo = here.parents[3]  # backend/app/services -> repo root
    roots.append(repo / "docs" / "jee-concept-taxonomy" / "data")
    roots.append(repo / "docs" / "jee-chapter-taxonomy" / "data")
    roots.append(repo / "docs" / "jee-syllabus-corpus" / "data")
    roots.append(repo / "docs" / "jee-pyq-corpus" / "data")
    roots.append(repo / "docs" / "jee-pyq-corpus" / "patterns")
    return roots


def _load(name: str) -> list[dict]:
    """Load corpus slices, aggregating every matching shard.

    PyQ shards aggregate across files (deduplicated by question_id) so the
    corpus grows by adding files, never by editing loaders. Taxonomy
    aggregates sub-concept files with the chapter-level taxonomy.
    """
    filenames = _SLICE_FILES[name]
    aggregated: list[dict] = []
    seen: set[str] = set()
    for root in _candidate_roots():
        for filename in filenames:
            direct = root / filename
            paths = [direct] if direct.is_file() else []
            if root.is_dir():
                paths.extend(p for p in sorted(root.rglob(filename)) if p not in paths)
            for path in paths:
                with path.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                logger.debug("JEE %s loaded from %s (%d records)", name, path, len(data))
                if name == "pyq":
                    for record in data:
                        qid = record.get("question_id")
                        if qid not in seen:
                            seen.add(qid)
                            aggregated.append(record)
                elif name == "taxonomy":
                    for concept in data:
                        # Chapter-level concepts carry chapter_id; normalize so
                        # the matcher sees one shape (level marks granularity).
                        if "concept_id" not in concept and "chapter_id" in concept:
                            concept = {**concept, "concept_id": concept["chapter_id"],
                                       "level": "chapter"}
                        aggregated.append(concept)
                else:
                    aggregated.extend(data)
                if name != "pyq" and name != "taxonomy":
                    return aggregated
        if aggregated and name not in ("pyq", "taxonomy"):
            return aggregated
    if not aggregated:
        logger.warning("JEE %s corpus not found; returning empty", name)
    return aggregated


def _concepts() -> list[dict]:
    return _load("taxonomy")


def _syllabus_nodes() -> list[dict]:
    return _load("syllabus")


def _pyq_records() -> list[dict]:
    return _load("pyq")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold()))


def _score_concept(concept: dict, terms: list[str]) -> tuple[int, list[str]]:
    """Fixed scoring: display-name exact +2, alias exact/token +1 each."""
    score = 0
    evidence: list[str] = []
    name = _normalize(str(concept.get("display_name", "")))
    aliases = [_normalize(str(a)) for a in concept.get("aliases", [])]
    for term in terms:
        normalized = _normalize(term)
        if not normalized:
            continue
        if normalized == name:
            score += 2
            evidence.append(f"exact concept-name match: {term.strip()!r}")
            continue
        if normalized in aliases:
            score += 1
            evidence.append(f"exact alias match: {term.strip()!r}")
            continue
        term_tokens = _tokens(normalized)
        for alias in aliases:
            if not alias:
                continue
            if alias == normalized or alias in term_tokens:
                score += 1
                evidence.append(f"alias token match: {alias!r} in {term.strip()!r}")
                break
            if len(alias) > 1 and re.search(r"\b" + re.escape(alias) + r"\b", normalized):
                score += 1
                evidence.append(f"alias phrase match: {alias!r} in {term.strip()!r}")
                break
    return score, evidence


def resolve_concept(topic_title: str, key_terms: list[str]) -> tuple[dict | None, str, list[str], list[dict]]:
    """Resolve candidate terms to one canonical concept.

    Sub-concept level decides first with the existing fixed scoring: a
    single winner matches, a tie is ambiguous, no hits fall through to the
    chapter level. Chapter fallback needs a clear single winner scoring 2+
    (two independent term hits), otherwise ties stay ambiguous and silence
    stays unresolved. Returns (concept | None, match_status, evidence, tied).
    """
    terms = [topic_title, *key_terms]
    concepts = _concepts()
    sub_level = [c for c in concepts if c.get("level") != "chapter"]
    chapters = [c for c in concepts if c.get("level") == "chapter"]

    scored = []
    for concept in sub_level:
        score, evidence = _score_concept(concept, terms)
        if score > 0:
            scored.append((score, concept, evidence))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        tied = [item for item in scored if item[0] == top_score]
        if len(tied) > 1:
            candidates = [
                {"concept_id": c["concept_id"], "name": c["display_name"]} for _, c, _ in tied
            ]
            return None, "ambiguous", [f"tie between {len(tied)} concepts; refusing to guess"], candidates
        _, concept, evidence = tied[0]
        return concept, "matched", evidence, []

    chapter_scored = []
    for concept in chapters:
        score, evidence = _score_concept(concept, terms)
        if score > 0:
            chapter_scored.append((score, concept, evidence))
    if chapter_scored:
        chapter_scored.sort(key=lambda item: item[0], reverse=True)
        top_score = chapter_scored[0][0]
        tied = [item for item in chapter_scored if item[0] == top_score]
        if len(tied) > 1:
            candidates = [
                {"concept_id": c["concept_id"], "name": c["display_name"]} for _, c, _ in tied
            ]
            return None, "ambiguous", [f"tie between {len(tied)} chapters; refusing to guess"], candidates
        if top_score >= 2:
            _, concept, evidence = tied[0]
            evidence = [*evidence, "chapter-level match: finer mapping unsupported by the given terms"]
            return concept, "matched", evidence, []
    return None, "unresolved", [], []


def _ref(concept_id: str, concepts_by_id: dict) -> JEEConceptRef | None:
    target = concepts_by_id.get(concept_id)
    if target is None:
        return None
    return JEEConceptRef(concept_id=concept_id, name=str(target.get("display_name", "")))


def _syllabus_link(concept: dict) -> JEESyllabusRef | None:
    wanted = _normalize(str(concept.get("syllabus_subtopic_ref", "")))
    if not wanted:
        return None
    for node in _syllabus_nodes():
        if _normalize(str(node.get("subtopic", ""))) == wanted:
            return JEESyllabusRef(
                node_id=str(node.get("id", "")),
                subtopic=str(node.get("subtopic", "")),
                source_url=str(node.get("source_url", "")),
            )
    return None


def _pyqs_for(concept_id: str) -> list[JEEPyQRef]:
    refs = []
    for record in _pyq_records():
        if concept_id in record.get("concept_ids", []):
            answer_key = record.get("answer_key")
            answer_text = str(answer_key) if answer_key is not None else None
            refs.append(
                JEEPyQRef(
                    question_id=str(record.get("question_id", "")),
                    exam=str(record.get("exam", "")),
                    year=int(record.get("year", 0)),
                    session=record.get("session"),
                    shift=record.get("shift"),
                    paper_question_number=int(record.get("paper_question_number", 0)),
                    question_type=str(record.get("question_type", "")),
                    marks=record.get("marks"),
                    question_summary=record.get("question_summary"),
                    source_url=str(record.get("source_url", "")),
                    source_document=str(record.get("source_document", "")),
                    source_class=_source_class(record),
                    answer_key=answer_text,
                    answer_status="confirmed" if answer_text is not None else "unconfirmed",
                )
            )
    # Newest first; the inline list is capped (pyq_count keeps the total).
    refs.sort(key=lambda ref: (ref.year, ref.paper_question_number), reverse=True)
    return refs


def map_concept_to_jee(topic_title: str, key_terms: list[str]) -> JEEConceptContext:
    """Build the student-facing JEE context for one lecture concept."""
    concept, status, evidence, tied = resolve_concept(topic_title, key_terms)
    if concept is None:
        candidates = [JEEConceptRef(concept_id=c["concept_id"], name=c["name"]) for c in tied]
        return JEEConceptContext(match_status=status, evidence=evidence, candidates=candidates)

    concepts_by_id = {c["concept_id"]: c for c in _concepts()}
    prerequisites = [
        ref
        for cid in concept.get("prerequisite_concept_ids", [])
        if (ref := _ref(cid, concepts_by_id)) is not None
    ]
    related = [
        ref
        for cid in concept.get("related_concept_ids", [])
        if (ref := _ref(cid, concepts_by_id)) is not None
    ]
    pyqs = _pyqs_for(str(concept["concept_id"]))
    syllabus = _syllabus_link(concept)
    valid_ids = {ref.question_id for ref in pyqs}
    chapter_id = str(concept.get("chapter_id", "") or concept.get("concept_id", ""))
    patterns = _patterns_for(chapter_id, valid_ids)
    truncated = len(pyqs) > MAX_INLINE_PYQS
    inline_pyqs = pyqs[:MAX_INLINE_PYQS]
    parts: list[str] = []
    if syllabus is None:
        parts.append("no syllabus entry is stored for it")
    if not pyqs:
        parts.append(
            "no verified PYQs are stored for this concept yet — that reflects "
            "corpus coverage, not proof JEE never tested it"
        )
    else:
        parts.append(f"{len(pyqs)} verified PYQ(s) stored")
        if patterns:
            parts.append(
                f"across {len(patterns)} observed pattern(s); pattern "
                "discovery is ongoing, not exhaustive"
            )
        else:
            parts.append("question-pattern mapping for this chapter is still incomplete")
    coverage_note = "Concept matched: " + "; ".join(parts) + "."
    return JEEConceptContext(
        concept_id=str(concept["concept_id"]),
        canonical_name=str(concept.get("display_name", "")),
        subject=str(concept.get("subject", "")),
        chapter=str(concept.get("chapter", "")),
        subtopic=str(concept.get("syllabus_subtopic_ref", "") or None),
        syllabus=syllabus,
        match_status=status,
        evidence=evidence,
        prerequisites=prerequisites,
        related=related,
        pyqs=inline_pyqs,
        pyq_count=len(pyqs),
        pyqs_truncated=truncated,
        patterns=patterns,
        coverage_note=coverage_note,
    )
