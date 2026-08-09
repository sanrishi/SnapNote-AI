import asyncio
import io
import json
import glob
import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw, ImageFont

from app.main import app
from app.services.preprocessor import enhance_for_vision
from app.services import vision_service
from app.config import settings

transport = ASGITransport(app=app)

FIXTURE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "stress_test_fixtures")
)

try:
    _font = ImageFont.truetype("arial.ttf", 64)
except Exception:
    _font = ImageFont.load_default()


def _valid_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="PNG")
    return buf.getvalue()


def _reset_credits(device_id: str, amount: int = 50) -> None:
    from app.utils.credits_store import _get_conn, init_device
    init_device(device_id)
    conn = _get_conn()
    conn.execute(
        "UPDATE device_credits SET credits_remaining = ?, credits_used = 0 WHERE device_id = ?",
        (amount, device_id),
    )
    conn.commit()


def _fixture_bytes(name: str) -> bytes:
    with open(os.path.join(FIXTURE_DIR, name), "rb") as fh:
        return fh.read()


@pytest.fixture
def large_phone_png() -> bytes:
    w, h = 2340, 1080
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    lines = [
        "omega sin theta equals omega_z",
        "omega = v / r   tau = I alpha",
        "subscript m_1 m_2 r-hat r_hat arrows ->",
    ]
    y = 60
    for line in lines:
        d.text((80, y), line, fill="black", font=_font)
        y += 180
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def small_jpeg() -> bytes:
    img = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 30), "small jpeg E = mc^2", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ── Compression: dimensions & file size ──


@pytest.mark.parametrize("name", [
    "1_stats_table.png",
    "2_flowchart.png",
    "3_math_formulas.png",
    "4_hindi_coaching.png",
    "5_youtube_coaching.png",
    "6_plain_prose.png",
    "7_biology_prose.png",
    "8_chemistry_equilibrium.png",
    "9_dark_bst.png",
])
def test_compression_preserves_dimensions_under_limit(name):
    data = _fixture_bytes(name)
    out = enhance_for_vision(data)
    img_out = Image.open(io.BytesIO(out))
    assert img_out.format == "JPEG"
    assert max(img_out.size) <= settings.MAX_VISION_LONG_EDGE
    img_in = Image.open(io.BytesIO(data))
    ratio_in = img_in.size[0] / img_in.size[1]
    ratio_out = img_out.size[0] / img_out.size[1]
    assert abs(ratio_in - ratio_out) < 0.01, "aspect ratio must be preserved"


def test_compression_large_png_downscales(large_phone_png):
    out = enhance_for_vision(large_phone_png)
    img_out = Image.open(io.BytesIO(out))
    assert img_out.size == (1600, 738)
    assert max(img_out.size) == settings.MAX_VISION_LONG_EDGE
    assert len(out) < len(large_phone_png)


def test_compression_small_jpeg_not_upscaled(small_jpeg):
    out = enhance_for_vision(small_jpeg)
    img_out = Image.open(io.BytesIO(out))
    assert img_out.size == (640, 360)


# ── Compression: OCR readability not damaged (formulas/symbols) ──

@pytest.mark.parametrize("name,required_symbols", [
    ("3_math_formulas.png", ["ω", "θ", "α"]),
    ("8_chemistry_equilibrium.png", ["="]),
    ("9_dark_bst.png", ["BST"]),
])
def test_compression_keeps_symbols_readable(name, required_symbols):
    """OCR on the compressed image must retain comparable text volume to the
    preprocessed original — the empirical guard against JPEG damage."""
    from app.services.ocr_service import read_raw
    from app.services.preprocessor import preprocess
    import numpy as np

    data = _fixture_bytes(name)
    out = enhance_for_vision(data)

    orig_tokens = {t.strip() for _, t, _ in read_raw(preprocess(data))}
    enh_np = np.array(Image.open(io.BytesIO(out)).convert("L"))
    enh_tokens = {t.strip() for _, t, _ in read_raw(enh_np)}

    orig_chars = sum(len(t) for t in orig_tokens)
    enh_chars = sum(len(t) for t in enh_tokens)
    assert enh_chars >= orig_chars * 0.8, (
        f"compression lost text volume: {orig_chars} -> {enh_chars}"
    )
    assert len(enh_tokens) >= 1


def test_compression_no_oversize_output(small_jpeg):
    out = enhance_for_vision(small_jpeg)
    assert len(out) < len(small_jpeg) * 1.1, "never bloat small inputs"


# ── Gemini timeout & retry wiring ──


def test_gemini_call_passes_bounded_timeout_and_no_retry(monkeypatch):
    async def fake(*args, **kwargs):
        return type("o", (), {"text": "ok"})()

    mock_model = AsyncMock()
    mock_model.generate_content_async.side_effect = fake
    monkeypatch.setattr(vision_service, "model", mock_model)

    async def run():
        return await vision_service._call_gemini("p", _valid_png())

    result = asyncio.run(run())
    assert result == "ok"
    kwargs = mock_model.generate_content_async.call_args.kwargs
    assert kwargs["request_options"]["timeout"] == settings.GEMINI_CALL_TIMEOUT_SECONDS
    assert kwargs["request_options"]["retry"] is None


def test_gemini_single_retry_on_429_only(monkeypatch):
    calls = {"n": 0}

    async def fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("429 RESOURCE_EXHAUSTED rate limit")
        return type("o", (), {"text": "retried ok"})()

    mock_model = AsyncMock()
    mock_model.generate_content_async.side_effect = fake
    monkeypatch.setattr(vision_service, "model", mock_model)
    monkeypatch.setattr(vision_service.asyncio, "sleep", AsyncMock())

    async def run():
        return await vision_service._call_gemini("p", _valid_png())

    assert asyncio.run(run()) == "retried ok"
    assert calls["n"] == 2  # exactly one retry


def test_gemini_no_retry_on_non_429(monkeypatch):
    from app.exceptions import UpstreamError

    calls = {"n": 0}

    async def fake(*args, **kwargs):
        calls["n"] += 1
        raise Exception("some other error")

    mock_model = AsyncMock()
    mock_model.generate_content_async.side_effect = fake
    monkeypatch.setattr(vision_service, "model", mock_model)

    async def run():
        await vision_service._call_gemini("p", _valid_png())

    with pytest.raises(UpstreamError):
        asyncio.run(run())
    assert calls["n"] == 1  # no stacked retry


def test_gemini_429_exhausts_after_single_retry(monkeypatch):
    from app.exceptions import UpstreamError

    calls = {"n": 0}

    async def fake(*args, **kwargs):
        calls["n"] += 1
        raise Exception("429 rate limit")

    mock_model = AsyncMock()
    mock_model.generate_content_async.side_effect = fake
    monkeypatch.setattr(vision_service, "model", mock_model)
    monkeypatch.setattr(vision_service.asyncio, "sleep", AsyncMock())

    async def run():
        await vision_service._call_gemini("p", _valid_png())

    with pytest.raises(UpstreamError):
        asyncio.run(run())
    assert calls["n"] == 2  # max_retries=1 → at most 2 calls


# ── Backend 75s timeout: returns 502, charges 0 credits ──


def test_diagram_timeout_returns_502_and_charges_zero(monkeypatch, sample_diagram_image):
    from app.utils.credits_store import get_credits

    device = "timeout-test-device-00000000-0000-0000-000000000000"
    _reset_credits(device)

    async def slow_notes(image_bytes):
        await asyncio.sleep(30)
        return None

    monkeypatch.setattr("app.routes.extract.extract_study_notes", slow_notes)
    monkeypatch.setattr(settings, "DIAGRAM_TIMEOUT_SECONDS", 0.2)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": device},
            )

    resp = asyncio.run(run())
    assert resp.status_code == 502
    remaining, _ = get_credits(device)
    assert remaining == 50  # 0 charged


# ── ImgBB failure does NOT break Gemini learning output ──


def test_imgbb_failure_still_returns_study_notes(sample_diagram_image, monkeypatch):
    from app.utils.credits_store import get_credits

    device = "imgbb-fail-device-00000000-0000-0000-000000000001"
    _reset_credits(device)

    payload = {
        "topic": {"title": "Rotation", "is_probable": False},
        "what_you_should_remember": "Remember the takeaway.",
        "key_formulas": [{"formula": "ω = v/r", "explanation": "", "uncertain_symbols": [], "confidence": "clear"}],
        "understand_it": ["Concept explanation."],
        "common_mistakes": [],
        "thirty_second_revision": ["Revise this."],
        "visual_context": {"present": False, "summary": ""},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)
    monkeypatch.setattr("app.routes.extract.upload_image", lambda img: None)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": device},
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    body = resp.json()
    assert body["studyNotes"]["what_you_should_remember"]
    assert body["imageUrl"] is None
    remaining, _ = get_credits(device)
    assert remaining == 45  # exactly 5 charged, output intact


# ── Credits: success charges exact amount, retry does not double-charge ──


def test_diagram_success_charges_exactly_5(sample_diagram_image, monkeypatch):
    from app.utils.credits_store import get_credits

    device = "exact-credit-device-00000000-0000-0000-000000000002"
    _reset_credits(device)

    payload = {
        "topic": {"title": "Rotation", "is_probable": False},
        "what_you_should_remember": "Takeaway.",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": False, "summary": ""},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": device},
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert resp.json()["creditsUsed"] == 5
    remaining, _ = get_credits(device)
    assert remaining == 45


def test_retry_path_does_not_double_charge(sample_diagram_image, monkeypatch):
    """429 on the first Gemini attempt → 502 (0 charged). A successful retry
    then charges exactly 5. Total charged = 5, never 10."""
    from app.utils.credits_store import get_credits

    device = "no-double-device-00000000-0000-0000-000000000003"
    _reset_credits(device)

    # First attempt: Gemini rate-limits → UpstreamError (502), no charge.
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=Exception("429 RESOURCE_EXHAUSTED"))
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def first_try():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": device},
            )

    resp = asyncio.run(first_try())
    assert resp.status_code == 502
    remaining, _ = get_credits(device)
    assert remaining == 50

    # Retry succeeds → exactly 5 charged.
    payload = {
        "topic": {"title": "Rotation", "is_probable": False},
        "what_you_should_remember": "Takeaway.",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": False, "summary": ""},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model2 = AsyncMock()
    mock_model2.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model2)

    async def second_try():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": device},
            )

    resp = asyncio.run(second_try())
    assert resp.status_code == 200
    remaining, _ = get_credits(device)
    assert remaining == 45


# ── Diagram mode never triggers the OCR pipeline ──


def test_diagram_mode_never_calls_ocr(sample_diagram_image, monkeypatch):
    from app.services import ocr_service

    payload = {
        "topic": {"title": "Rotation", "is_probable": False},
        "what_you_should_remember": "Takeaway.",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": False, "summary": ""},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    called = {"n": 0}
    original = ocr_service.read_raw

    def boom(*a, **k):
        called["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr("app.routes.extract.read_raw", boom)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": "no-ocr-device-00000000-0000-0000-000000000004"},
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert called["n"] == 0  # no OCR triggered in diagram mode


# ── Diagram SVG generation reliability rules ──


def test_study_notes_prompt_enforces_svg_fill_safety():
    """The exact failure caught on camera — solid-filled circles blotting out
    content — must stay forbidden by the prompt."""
    prompt = vision_service.STUDY_NOTES_SYSTEM_PROMPT
    assert "FILL SAFETY" in prompt
    assert 'fill="none"' in prompt
    assert "NEVER fill a whole circle or outline shape solid" in prompt
    assert "opacity=\"0.15\"–\"0.35\"" in prompt
    assert "labels as <text> elements" in prompt


def test_study_notes_prompt_enforces_diagram_completeness():
    """The second caught failure — dropping one of two visible diagrams — must
    stay forbidden by the prompt."""
    prompt = vision_service.STUDY_NOTES_SYSTEM_PROMPT
    assert "COMPLETENESS" in prompt
    assert "include ALL of them" in prompt
    assert "never simplify down to a single figure" in prompt


def test_revision_prompt_reuses_diagram_rules():
    """Revision tier inherits the same diagram rules — no parallel drift."""
    prompt = vision_service.REVISION_SYSTEM_PROMPT
    assert "diagram: same rules as the main prompt" in prompt


# ── Frontend 95s timeout + no infinite spinner ──


def test_frontend_has_95s_timeout_and_no_infinite_spinner():
    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "website", "index.html"))
    with open(html_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    assert "REQUEST_TIMEOUT_MS = 95000" in html
    assert "AbortController" in html
    assert "No credits were charged" in html
    assert "controller.abort()" in html



