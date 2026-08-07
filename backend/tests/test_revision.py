import json
import io
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.services.vision_service import extract_revision_guide
from app.utils.render_notes import uncertainty_sentence

transport = ASGITransport(app=app)
TEST_DEVICE_ID = "test-revision-00000000-0000-0000-0000-000000000000"


def _valid_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def seed_credits():
    from app.utils.credits_store import _get_conn
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO device_credits (device_id, credits_remaining, credits_used) VALUES (?, 999, 0)",
        (TEST_DEVICE_ID,),
    )
    conn.commit()


def _learning_payload(**overrides):
    payload = {
        "topic": {"title": "Angular Velocity", "is_probable": False},
        "what_you_should_remember": "Project the velocity vector; use sin/cos for components.",
        "key_formulas": [{"formula": "ω_z = ω sinθ", "explanation": "vertical component of angular velocity", "uncertain_symbols": [], "confidence": "clear"}],
        "understand_it": ["Think of a vector projected onto an axis."],
        "common_mistakes": ["Swapping sin and cos while resolving components."],
        "thirty_second_revision": ["Project the vector.", "Use sin/cos for components."],
        "visual_context": {"present": False, "summary": ""},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "Like splitting a diagonal force into horizontal and vertical pulls.",
    }
    payload.update(overrides)
    return payload


# ── Unit: extract_revision_guide happy path ──

@pytest.mark.asyncio
async def test_extract_revision_guide_valid(monkeypatch):
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(_learning_payload())})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    guide = await extract_revision_guide(_valid_png(), context_hint='{"topic": "Angular Velocity"}')
    assert guide.what_you_should_remember
    assert guide.key_formulas
    assert guide.common_mistakes
    assert guide.thirty_second_revision
    assert guide.analogy
    assert mock_model.generate_content_async.call_count == 1


# ── Unit: repair retry on malformed first response ──

@pytest.mark.asyncio
async def test_extract_revision_guide_repair(monkeypatch):
    good = _learning_payload()
    bad_response = type("o", (), {"text": "not json"})()
    good_response = type("o", (), {"text": json.dumps(good)})()
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=[bad_response, good_response])
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    guide = await extract_revision_guide(_valid_png())
    assert guide.what_you_should_remember == good["what_you_should_remember"]
    assert mock_model.generate_content_async.call_count == 2


# ── Unit: double repair failure → UpstreamError ──

@pytest.mark.asyncio
async def test_extract_revision_guide_repair_fails(monkeypatch):
    from app.exceptions import UpstreamError

    bad_response = type("o", (), {"text": "garbage"})()
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=bad_response)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    with pytest.raises(UpstreamError):
        await extract_revision_guide(_valid_png())


# ── Route: revision endpoint charges 1 credit and returns learning sheet ──

@pytest.mark.asyncio
async def test_revision_route_ok(sample_diagram_image):
    from unittest.mock import patch

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(_learning_payload())})())
    with patch("app.services.vision_service.model", mock_model):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/revision",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({"topic": "Architecture"}), "deviceId": TEST_DEVICE_ID},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["creditsUsed"] == 1
    assert body["study_notes"]["what_you_should_remember"]
    assert body["study_notes"]["key_formulas"]


# ── Route: revision failure does not charge credits ──

@pytest.mark.asyncio
async def test_revision_failure_no_charge(sample_diagram_image):
    from unittest.mock import patch
    from app.utils.credits_store import get_credits, init_device

    device = "revision-fail-device-00000000-0000-0000-000000000001"
    init_device(device)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=Exception("Gemini timeout"))
    with patch("app.services.vision_service.model", mock_model):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/revision",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": "{}", "deviceId": device},
            )
    assert resp.status_code == 502
    remaining, _ = get_credits(device)
    assert remaining == 50


# ── Unit: deterministic uncertainty sentence ──

def test_uncertainty_sentence_deterministic():
    s = uncertainty_sentence("θ")
    assert "θ" in s
    assert "cannot be confirmed" in s
    assert "lecture context is missing" in s
    assert s == uncertainty_sentence("θ")  # deterministic


# ── Unit: render includes Verify Before Studying section ──

def test_render_verify_before_studying():
    from app.models.schemas import StudyNotes
    from app.utils.render_notes import render_study_notes

    notes = StudyNotes(
        topic={"title": "Rotational Motion", "is_probable": True},
        key_formulas=[{"formula": "t = Iα", "explanation": "", "uncertain_symbols": [], "confidence": "possible_extraction_issue"}],
        verify_before_studying=["τ = Iα may have been misread as 't = Iα' because the tau was blurry."],
    )
    md = render_study_notes(notes)
    assert "🛡️ Verify Before Studying" in md
    assert "τ = Iα may have been misread" in md
    assert "Verify these against the original lecture" in md


def test_render_verify_empty_omitted():
    from app.models.schemas import StudyNotes
    from app.utils.render_notes import render_study_notes

    notes = StudyNotes(topic={"title": "T", "is_probable": False})
    md = render_study_notes(notes)
    assert "Verify Before Studying" not in md


# ── Unit: render learning-first ordering ──

def test_render_learning_first_order():
    from app.models.schemas import StudyNotes
    from app.utils.render_notes import render_study_notes

    notes = StudyNotes(
        topic={"title": "Rotational Motion", "is_probable": False},
        what_you_should_remember="Remember the takeaway.",
        key_formulas=[{"formula": "ω = v/r", "explanation": "", "uncertain_symbols": [], "confidence": "clear"}],
        understand_it=["Understand this concept."],
        common_mistakes=["A mistake."],
        thirty_second_revision=["Bullet one."],
        visual_context={"present": True, "summary": "Compact context."},
    )
    md = render_study_notes(notes)
    idx_remember = md.index("What You Should Remember")
    idx_formulas = md.index("Key Formulas")
    idx_understand = md.index("Understand It")
    idx_mistakes = md.index("Common Mistakes")
    idx_30 = md.index("30-Second Revision")
    idx_visual = md.index("Visual Context")
    assert idx_remember < idx_formulas < idx_understand < idx_mistakes < idx_30 < idx_visual


# ── Unit: structured text formatting groups content ──

def test_format_structured_text_groups():
    from app.services.ocr_service import format_structured_text

    lines = [
        "Net Angular Velocity",
        "ω_net = ω r̂₁ + ω_z k̂",
        "ω sinθ = ω_z",
        "z-axis",
        "IAR",
        "θ",
        "The cone rolls without slipping on the surface.",
    ]
    md = format_structured_text(lines)
    assert "## Visible Formulas" in md
    assert "## Visible Labels" in md
    assert "## Extracted Text" in md
    assert "ω sinθ = ω_z" in md
    assert "- IAR" in md


def test_format_structured_text_dedupes():
    from app.services.ocr_service import format_structured_text

    md = format_structured_text(["ω = v/r", "ω = v/r", "label"])
    assert md.count("ω = v/r") == 1
