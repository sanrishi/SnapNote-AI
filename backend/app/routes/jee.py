"""JEE Intelligence boundary (issue #32 vertical slice).

Free metadata lookup over the stored syllabus/taxonomy/PYQ corpora.
No credits, no auth, no LLM calls — every fact traces to stored records.
"""
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.models.schemas import JEEConceptContext, JEEMapRequest
from app.services.jee_service import _concepts, map_concept_to_jee

logger = logging.getLogger(__name__)
router = APIRouter()


class JEEChapterRef(BaseModel):
    concept_id: str = ""
    display_name: str = ""
    subject: str = ""


@router.post("/map", response_model=JEEConceptContext)
def map_jee_concept(req: JEEMapRequest) -> JEEConceptContext:
    """Map lecture terms to a canonical JEE concept + syllabus + verified PYQs."""
    return map_concept_to_jee(req.topic_title, req.key_terms)


@router.get("/chapters", response_model=list[JEEChapterRef])
def list_jee_chapters() -> list[JEEChapterRef]:
    """All supported chapter-level concepts for honest manual recovery."""
    refs = [
        JEEChapterRef(
            concept_id=str(c.get("concept_id", "")),
            display_name=str(c.get("display_name", "")),
            subject=str(c.get("subject", "")),
        )
        for c in _concepts()
        if c.get("level") == "chapter" and c.get("display_name")
    ]
    refs.sort(key=lambda r: (r.subject, r.display_name))
    return refs


@router.get("/concept/{concept_id}", response_model=JEEConceptContext)
def get_jee_concept(concept_id: str) -> JEEConceptContext:
    """Full stored context for one canonical concept ID (no guessing)."""
    for concept in _concepts():
        if concept.get("concept_id") == concept_id:
            return map_concept_to_jee(str(concept.get("display_name", "")), [])
    return JEEConceptContext(
        match_status="unresolved",
        evidence=[f"unknown concept_id: {concept_id}"],
    )
