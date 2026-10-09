"""Product integration tests for the shipped JEE Intelligence experience.

Covers the real student path (StudyNotes -> JEE context -> verified PYQs),
matching states, provenance and answer discipline, pattern references,
deduplication, honest empty states, extract-route integration, and the
free-lookup credit contract.
"""
import asyncio
import io
import json

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.services import jee_service
from app.services.jee_service import map_concept_to_jee
from app.utils.render_notes import render_jee_section

transport = ASGITransport(app=app)
TEST_DEVICE_ID = "test-jee-product-00000000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def seed_credits():
    from app.utils.credits_store import _get_conn
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO device_credits (device_id, credits_remaining, credits_used) VALUES (?, 999, 0)",
        (TEST_DEVICE_ID,),
    )
    conn.commit()


def _valid_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, format="PNG")
    return buf.getvalue()


def _notes_payload(title, explanation="angular velocity equals linear velocity over radius"):
    return {
        "topic": {"title": title, "is_probable": False},
        "what_you_should_remember": "Something to remember.",
        "key_formulas": [{"formula": "x = y", "explanation": explanation,
                          "uncertain_symbols": [], "confidence": "clear"}],
        "understand_it": ["Some understanding."],
        "common_mistakes": [],
        "thirty_second_revision": ["One bullet."],
        "visual_context": {"present": False, "summary": ""},
        "diagram": {"present": False, "svg": "", "best_effort": False},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }


# 1. Chapter-level fallback resolves with verified PYQs
def test_chapter_level_match_carries_pyqs():
    ctx = map_concept_to_jee("Matrices and Determinants", [])
    assert ctx.match_status == "matched"
    assert ctx.concept_id == "jee-mathematics-matrices-and-determinants-chapter"
    assert ctx.pyq_count >= 1
    assert len({p.question_id for p in ctx.pyqs}) == len(ctx.pyqs)


# 2. Popular chapter truncates inline list but keeps the honest total
def test_large_chapter_truncates_with_total():
    from app.services.jee_service import _pyq_records
    import collections
    counts = collections.Counter()
    for record in _pyq_records():
        for cid in record.get("concept_ids", []):
            counts[cid] += 1
    big = max(counts, key=counts.get)
    assert counts[big] > jee_service.MAX_INLINE_PYQS, "no chapter exceeds the cap; adjust test"
    from app.services.jee_service import _concepts
    concept = next(c for c in _concepts() if c["concept_id"] == big)
    ctx = map_concept_to_jee(str(concept.get("display_name", "")), [])
    assert ctx.match_status == "matched"
    assert ctx.pyq_count == counts[big]
    assert ctx.pyqs_truncated is True
    assert len(ctx.pyqs) == jee_service.MAX_INLINE_PYQS


# 3. Provenance classes and answer discipline on every returned PYQ
def test_pyq_provenance_and_answer_discipline():
    ctx = map_concept_to_jee("Matrices and Determinants", [])
    assert ctx.pyqs
    for pyq in ctx.pyqs:
        assert pyq.source_class in ("OFFICIAL", "THIRD_PARTY_TRANSCRIPTION")
        assert pyq.answer_status in ("confirmed", "unconfirmed")
        if pyq.answer_key is None:
            assert pyq.answer_status == "unconfirmed"
        else:
            assert pyq.answer_status == "confirmed"
        assert "frequency" not in pyq.model_dump() and "weightage" not in pyq.model_dump()


# 4. Patterns reference only existing verified IDs; invalid refs filtered
def test_pattern_references_resolve():
    ctx = map_concept_to_jee("Matrices and Determinants", [])
    assert len(ctx.patterns) >= 1
    inline_ids = {p.question_id for p in jee_service._pyqs_for(ctx.concept_id)}
    for pattern in ctx.patterns:
        assert pattern.question_ids
        assert set(pattern.question_ids) <= inline_ids
        assert pattern.coverage_note
    assert "not exhaustive" in ctx.coverage_note
    assert "verified PYQ(s) stored" in ctx.coverage_note


def test_invalid_pattern_reference_filtered():
    from app.models.schemas import JEEPatternRef
    refs = jee_service._patterns_for("jee-physics-kinematics-chapter", {"no-such-id"})
    assert refs == []
    assert isinstance(refs, list)


# 5. Deduplication: one ref per question even with repeated shards
def test_no_duplicate_pyq_refs():
    for title in ["Torque", "Matrices and Determinants", "Optics", "Thermodynamics"]:
        ctx = map_concept_to_jee(title, [])
        if ctx.match_status == "matched":
            ids = [p.question_id for p in ctx.pyqs]
            assert len(ids) == len(set(ids)), title


# 6. Non-JEE content stays JEE-free
def test_generic_screenshot_unresolved():
    ctx = map_concept_to_jee("Photosynthesis in plants", ["chlorophyll"])
    assert ctx.match_status == "unresolved"
    assert ctx.pyqs == [] and ctx.patterns == []
    assert render_jee_section(ctx) == ""
    assert render_jee_section(None) == ""


# 7. Markdown section states
def test_jee_markdown_section_states():
    matched = map_concept_to_jee("Torque", ["torque"])
    section = render_jee_section(matched)
    assert "## 🎓 JEE Context" in section
    assert "Torque" in section
    ambiguous = map_concept_to_jee("Moment of inertia", ["torque"])
    assert ambiguous.match_status == "ambiguous"
    amb_section = render_jee_section(ambiguous)
    assert "could be more than one JEE concept" in amb_section
    assert "## 🎓 JEE Context" in amb_section
    assert "Torque" in amb_section and "Moment of inertia" in amb_section


# 8. Route: diagram with JEE-relevant notes attaches context + markdown
@pytest.mark.asyncio
async def test_diagram_route_attaches_jee(sample_diagram_image):
    from unittest.mock import AsyncMock
    import unittest.mock as um
    payload = _notes_payload("Torque", "torque links force and rotation")
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(payload)})())
    with um.patch("app.services.vision_service.model", mock_model):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["jee"] is not None
    assert body["jee"]["match_status"] == "matched"
    assert body["jee"]["pyq_count"] >= 1
    assert "## 🎓 JEE Context" in body["markdown"]


# 9. Route: generic notes attach nothing and change no credits behavior
@pytest.mark.asyncio
async def test_diagram_route_generic_no_jee(sample_diagram_image):
    from unittest.mock import AsyncMock
    import unittest.mock as um
    from app.utils.credits_store import get_credits
    payload = _notes_payload("Photosynthesis in pondweed", "chlorophyll bubbles")
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(payload)})())
    before, _ = get_credits(TEST_DEVICE_ID)
    with um.patch("app.services.vision_service.model", mock_model):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["jee"] is None
    assert "JEE Context" not in body["markdown"]
    after, _ = get_credits(TEST_DEVICE_ID)
    from app.config import settings
    assert before - after == settings.DIAGRAM_CREDIT_COST  # JEE lookup added zero cost
