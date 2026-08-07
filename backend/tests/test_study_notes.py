import json
import io
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.services.vision_service import extract_study_notes, _extract_json

transport = ASGITransport(app=app)
TEST_DEVICE_ID = "test-structured-00000000-0000-0000-0000-000000000000"


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


# ── Unit: JSON extraction helper ──

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_raises_on_invalid():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("not json")


# ── Unit: extract_study_notes happy path ──

@pytest.mark.asyncio
async def test_extract_study_notes_valid(monkeypatch):
    payload = {
        "topic": {"title": "Rotational Motion", "is_probable": False},
        "visible_content": {"headings": [], "equations": ["ω = v/r"], "labels": [], "statements": []},
        "study_notes": ["Angular velocity is the rate of change of angle."],
        "simple_explanation": "This appears to relate angular velocity to linear velocity.",
        "formula_box": [{"formula": "ω = v/r", "explanation": "angular velocity equals linear velocity over radius", "uncertain_symbols": []}],
        "diagram_interpretation": {"present": False, "visible_elements": [], "likely_interpretation": []},
        "uncertainties": [],
        "verify_before_studying": ["ω = v/r may have been misread because the omega was blurry."],
        "key_takeaway": "ω = v/r",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    notes = await extract_study_notes(_valid_png())
    assert notes.topic.title == "Rotational Motion"
    assert notes.formula_box[0].formula == "ω = v/r"
    assert notes.verify_before_studying
    assert mock_model.generate_content_async.call_count == 1  # no repair retry


# ── Unit: repair retry on malformed first response ──

@pytest.mark.asyncio
async def test_extract_study_notes_repair(monkeypatch):
    good = {
        "topic": {"title": "T", "is_probable": True},
        "visible_content": {"headings": [], "equations": [], "labels": [], "statements": []},
        "study_notes": [],
        "simple_explanation": "",
        "formula_box": [],
        "diagram_interpretation": {"present": False, "visible_elements": [], "likely_interpretation": []},
        "uncertainties": [],
        "verify_before_studying": [],
        "key_takeaway": "",
    }
    bad_response = type("o", (), {"text": "this is not json"})()
    good_response = type("o", (), {"text": json.dumps(good)})()

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=[bad_response, good_response])
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    notes = await extract_study_notes(_valid_png())
    assert notes.topic.title == "T"
    assert notes.topic.is_probable is True
    assert mock_model.generate_content_async.call_count == 2  # original + one repair


# ── Unit: double repair failure → UpstreamError ──

@pytest.mark.asyncio
async def test_extract_study_notes_repair_fails(monkeypatch):
    from app.exceptions import UpstreamError

    bad_response = type("o", (), {"text": "garbage"})()
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=bad_response)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    with pytest.raises(UpstreamError):
        await extract_study_notes(_valid_png())
    assert mock_model.generate_content_async.call_count == 2


# ── Route: diagram returns structured studyNotes ──

@pytest.mark.asyncio
async def test_diagram_route_structured(sample_diagram_image):
    payload = {
        "topic": {"title": "Architecture", "is_probable": False},
        "visible_content": {"headings": [], "equations": [], "labels": ["Process A"], "statements": []},
        "study_notes": ["Two components linked."],
        "simple_explanation": "This appears to show a system with two linked components.",
        "formula_box": [],
        "diagram_interpretation": {"present": True, "visible_elements": ["Process A"], "likely_interpretation": []},
        "uncertainties": ["One arrow's meaning is unclear."],
        "verify_before_studying": ["The label near the top may have been misread."],
        "key_takeaway": "Remember the link between components.",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch_setter = pytest.importorskip("unittest.mock").patch
    with monkeypatch_setter("app.services.vision_service.model", mock_model):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["studyNotes"]["topic"]["title"] == "Architecture"
    assert body["studyNotes"]["diagram_interpretation"]["present"] is True
    assert body["studyNotes"]["uncertainties"] == ["One arrow's meaning is unclear."]
    assert body["studyNotes"]["verify_before_studying"]
    assert "Verify Before Studying" in body["markdown"]
    assert "## Key Takeaway" in body["markdown"] or "Key takeaway" in body["markdown"]


# ── Route: diagram failure does not charge credits ──

@pytest.mark.asyncio
async def test_diagram_failure_no_double_charge(sample_diagram_image):
    from app.utils.credits_store import get_credits, init_device

    device = "failure-test-device-00000000-0000-0000-000000000001"
    init_device(device)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=Exception("Gemini timeout"))
    import unittest.mock as um
    with um.patch("app.services.vision_service.model", mock_model):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": device},
            )
    assert resp.status_code == 502
    remaining, _ = get_credits(device)
    assert remaining == 50

