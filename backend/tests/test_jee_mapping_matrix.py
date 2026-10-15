"""Corpus-wide mapping regression gate: every mirror label + curated cases.

Positive cases assert the exact expected concept (chapter, or the known
more-precise rotational sub-concept refinements). Single-word 'Thermodynamics'
is documented ambiguous (physics vs chemistry tie — honest, not a miss).
Negative cases must stay unresolved; known ties must stay ambiguous.

Acceptance: zero avoidable unresolved on positives, zero false positives
on negatives/ambiguous.
"""
import asyncio
import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.jee_service import map_concept_to_jee

transport = ASGITransport(app=app)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_MIRROR = json.load(open(os.path.join(
    _REPO, "docs", "jee-chapter-taxonomy", "data", "mirror_topic_map.json"),
    encoding="utf-8"))

# Labels whose correct resolution is a more precise sub-concept (or honest
# ambiguity) rather than the coarse mapped chapter. Everything else must hit
# its mapped chapter exactly.
_SPECIAL = {
    "Angular Momentum": "jee-physics-rotational-motion-concept-006",
    "Rotational Motion": "jee-physics-rotational-motion-concept-003",
    "Rotational motion": "jee-physics-rotational-motion-concept-003",
    "System of Particles and Rotational Motion": "jee-physics-rotational-motion-concept-003",
    "Thermodynamics": "ambiguous",
}

CASES = [(label, _SPECIAL.get(label, entry["chapter_id"]))
         for label, entry in sorted(_MIRROR.items())]

CURATED = [
    # (title, terms, expected concept or "ambiguous"/"unresolved")
    ("Logarithmic Inequality with Variable Base",
     ["log base x+3 of (x squared minus x) less than 1",
      "domain: base positive not equal 1"],
     "jee-mathematics-limit-continuity-and-differentiability-chapter"),
    ("Torque", ["torque"],
     "jee-physics-rotational-motion-concept-005"),
    ("Rotational Dynamics and Angular Speed",
     ["rigid body", "pure rotation", "angular speed"],
     "jee-physics-rotational-motion-concept-014"),
    ("Complex Numbers as Square Vertices",
     ["complex number z and its conjugate plotted on the argand plane"],
     "jee-mathematics-complex-numbers-and-quadratic-equations-chapter"),
    ("Photosynthesis in pondweed", ["chlorophyll bubbles"], "unresolved"),
    ("Pancake recipe with maple syrup", ["flour", "batter"], "unresolved"),
    ("Moment of inertia", ["torque"], "ambiguous"),
]


@pytest.mark.parametrize("label,expected", CASES)
def test_mirror_label_resolves(label, expected):
    ctx = map_concept_to_jee(label, [])
    if expected == "ambiguous":
        assert ctx.match_status == "ambiguous", label
        return
    assert ctx.match_status == "matched", label
    assert ctx.concept_id == expected, (label, ctx.concept_id)
    if ctx.concept_id.endswith("-chapter"):
        assert any("chapter-level" in e for e in ctx.evidence), label


@pytest.mark.parametrize("title,terms,expected", CURATED)
def test_curated_cases(title, terms, expected):
    ctx = map_concept_to_jee(title, terms)
    if expected in ("ambiguous", "unresolved"):
        assert ctx.match_status == expected, title
        assert ctx.pyqs == []
        return
    assert ctx.match_status == "matched", title
    assert ctx.concept_id == expected, (title, ctx.concept_id)


def test_every_chapter_has_coverage():
    covered = {entry["chapter_id"] for entry in _MIRROR.values()}
    chapters = json.load(open(os.path.join(
        _REPO, "docs", "jee-chapter-taxonomy", "data", "jee-chapters.json"),
        encoding="utf-8"))
    missing = [c["chapter_id"] for c in chapters if c["chapter_id"] not in covered]
    assert not missing, missing


def test_chapters_endpoint_lists_all_chapters():
    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/jee/chapters")

    body = asyncio.run(run()).json()
    assert len(body) == 53
    assert all(set(c) == {"concept_id", "display_name", "subject"} for c in body)
    assert any(c["display_name"] == "Limit, Continuity and Differentiability" for c in body)


def test_chapter_match_retrieves_chapter_records():
    ctx = map_concept_to_jee("Logarithmic Inequality with Variable Base",
                             ["logarithmic inequality"])
    assert ctx.match_status == "matched"
    assert ctx.pyq_count >= 1
    import glob
    have = set()
    for path in glob.glob(os.path.join(_REPO, "docs", "jee-pyq-corpus", "data", "*-pyq.json")):
        for record in json.load(open(path, encoding="utf-8")):
            if ctx.concept_id in record.get("concept_ids", []):
                have.add(record["question_id"])
    assert {p.question_id for p in ctx.pyqs} <= have
    assert ctx.match_granularity == "chapter"


# ── ambiguous tie returns per-candidate question groups (never zero help) ──

LCD_ID = "jee-mathematics-limit-continuity-and-differentiability-chapter"
SOL_ID = "jee-chemistry-solutions-chapter"


def _tie(monkeypatch, first, second):
    import app.services.jee_service as jee_service

    def fake_resolve(topic_title, key_terms):
        concepts = {c["concept_id"]: c for c in jee_service._concepts()}
        return None, "ambiguous", ["tie between 2 chapters; refusing to guess"], [
            {"concept_id": first, "name": concepts[first]["display_name"]},
            {"concept_id": second, "name": concepts[second]["display_name"]},
        ]

    monkeypatch.setattr(jee_service, "resolve_concept", fake_resolve)


def _corpus_ids_for(concept_id):
    import glob
    have = set()
    for path in glob.glob(os.path.join(_REPO, "docs", "jee-pyq-corpus", "data", "*-pyq.json")):
        for record in json.load(open(path, encoding="utf-8")):
            if concept_id in record.get("concept_ids", []):
                have.add(record["question_id"])
    return have


def test_ambiguous_returns_groups_with_real_records(monkeypatch):
    _tie(monkeypatch, LCD_ID, SOL_ID)
    ctx = map_concept_to_jee("Logarithmic Inequality with Variable Base",
                             ["find all solutions"])
    assert ctx.match_status == "ambiguous"
    assert [c.concept_id for c in ctx.candidates] == [LCD_ID, SOL_ID]
    assert len(ctx.candidate_results) == 2
    seen: set[str] = set()
    for group in ctx.candidate_results:
        assert group.pyq_count >= 1
        assert group.pyqs, group.concept_id
        assert "exact-topic" in group.coverage_note or "exact-subtopic" in group.coverage_note
        for ref in group.pyqs:
            assert ref.question_id in _corpus_ids_for(group.concept_id), ref.question_id
            assert ref.question_id not in seen, ref.question_id  # deduped across groups
            seen.add(ref.question_id)
            assert ref.answer_status in ("confirmed", "unconfirmed")
            if ref.answer_key is None:
                assert ref.answer_status == "unconfirmed"


def test_ambiguous_group_empty_state(monkeypatch):
    import app.services.jee_service as jee_service

    def fake_resolve(topic_title, key_terms):
        concepts = {c["concept_id"]: c for c in jee_service._concepts()}
        return None, "ambiguous", ["tie"], [
            {"concept_id": LCD_ID, "name": concepts[LCD_ID]["display_name"]},
            {"concept_id": "jee-physics-no-such-chapter", "name": "No Such Chapter"},
        ]

    monkeypatch.setattr(jee_service, "resolve_concept", fake_resolve)
    ctx = map_concept_to_jee("Something", ["something"])
    assert ctx.match_status == "ambiguous"
    assert len(ctx.candidate_results) == 2
    empty = [g for g in ctx.candidate_results if g.concept_id == "jee-physics-no-such-chapter"]
    assert len(empty) == 1
    assert empty[0].pyqs == [] and empty[0].pyq_count == 0
    assert "No stored records" in empty[0].coverage_note
    assert len([g for g in ctx.candidate_results if g.pyqs]) == 1


def test_ambiguous_markdown_renders_previews():
    from app.utils.render_notes import render_jee_section

    ctx = map_concept_to_jee("Thermodynamics", [])
    assert ctx.match_status == "ambiguous"
    assert ctx.candidate_results, "tie must carry question groups"
    md = render_jee_section(ctx)
    assert "chapter not yet certain" in md
    assert "Possible: " in md
    assert "answer unconfirmed" in md or "Answer: " in md
    assert "exact-subtopic relevance has not been confirmed" in md
