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
    JEEPyQRef,
    JEESyllabusRef,
)

logger = logging.getLogger(__name__)

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

    Returns (concept | None, match_status, evidence, tied_candidates).
    """
    terms = [topic_title, *key_terms]
    scored = []
    for concept in _concepts():
        score, evidence = _score_concept(concept, terms)
        if score > 0:
            scored.append((score, concept, evidence))
    if not scored:
        return None, "unresolved", [], []
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
                )
            )
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
        pyqs=pyqs,
        pyq_count=len(pyqs),
    )
