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
    from app.utils.credits_store import init_device
    init_device(TEST_DEVICE_ID)


# ── Unit: extract_revision_guide happy path ──

@pytest.mark.asyncio
async def test_extract_revision_guide_valid(monkeypatch):
    payload = {
        "why_it_matters": "This concept underpins how angular velocity splits into components.",
        "intuition": "Think of a vector projected onto an axis.",
        "common_mistakes": ["Swapping sin and cos while resolving components."],
        "thirty_second_revision": "Project the velocity vector; use sin/cos for components.",
        "analogy": "Like splitting a diagonal force into horizontal and vertical pulls.",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    guide = await extract_revision_guide(_valid_png(), context_hint='{"topic": "Angular Velocity"}')
    assert guide.why_it_matters
    assert guide.common_mistakes
    assert guide.thirty_second_revision
    assert mock_model.generate_content_async.call_count == 1


# ── Unit: repair retry on malformed first response ──

@pytest.mark.asyncio
async def test_extract_revision_guide_repair(monkeypatch):
    good = {
        "why_it_matters": "W",
        "intuition": "I",
        "common_mistakes": [],
        "thirty_second_revision": "T",
        "analogy": "",
    }
    bad_response = type("o", (), {"text": "not json"})()
    good_response = type("o", (), {"text": json.dumps(good)})()
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=[bad_response, good_response])
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    guide = await extract_revision_guide(_valid_png())
    assert guide.why_it_matters == "W"
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


# ── Route: revision endpoint charges 1 credit and returns guide ──

@pytest.mark.asyncio
async def test_revision_route_ok(sample_diagram_image):
    from unittest.mock import patch

    payload = {
        "why_it_matters": "Architecture matters for system design.",
        "intuition": "Components exchange data through links.",
        "common_mistakes": ["Assuming the arrow is a power line."],
        "thirty_second_revision": "Two components, one link, data flows left to right.",
        "analogy": "Like two offices connected by a corridor.",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
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
    assert body["revision_guide"]["why_it_matters"]
    assert body["revision_guide"]["thirty_second_revision"]


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
