import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.schemas import ExtractionResponse, ExtractionType

transport = ASGITransport(app=app)
TEST_DEVICE_ID = "test-device-00000000-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def seed_credits():
    from app.utils.credits_store import _get_conn, init_device
    init_device(TEST_DEVICE_ID)
    conn = _get_conn()
    conn.execute(
        "UPDATE device_credits SET credits_remaining = 999, credits_used = 0 WHERE device_id = ?",
        (TEST_DEVICE_ID,),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Fixture 1: Clean text extraction ──

@pytest.mark.asyncio
async def test_text_extraction(sample_text_image, mock_vision):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("test.png", sample_text_image, "image/png")},
            data={"context": json.dumps({"title": "Stats Lecture Week 3", "url": "https://youtube.com/watch?v=123"}), "deviceId": TEST_DEVICE_ID},
        )
    assert resp.status_code == 200
    parsed = ExtractionResponse(**resp.json())
    assert parsed.type in (ExtractionType.TEXT, ExtractionType.TABLE)
    assert any(word in parsed.markdown.lower() for word in ["statistics", "mean", "median", "mode"])
    assert "youtube" in parsed.tags
    assert "week" in " ".join(t.lower() for t in parsed.tags)
    assert parsed.creditsUsed == 1
    assert parsed.imageUrl is None
    assert mock_vision.call_count == 0


# ── Fixture 2: Table extraction (normal spacing) ──

@pytest.mark.asyncio
async def test_table_extraction(sample_table_image, mock_vision):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("table.png", sample_table_image, "image/png")},
            data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
        )
    assert resp.status_code == 200
    parsed = ExtractionResponse(**resp.json())
    assert parsed.type == ExtractionType.TABLE
    assert "|" in parsed.markdown
    assert any(word in parsed.markdown for word in ["Name", "Age", "Score", "Alice", "Bob"])
    assert parsed.creditsUsed == 1
    assert mock_vision.call_count == 0


# ── Fixture 3: Tight-spacing table (row-clustering regression test) ──

@pytest.mark.asyncio
async def test_tight_table_extraction(sample_tight_table_image, monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import vision_service

    device = "tight-table-device-00000000-0000-0000-000000000003"
    from app.utils.credits_store import init_device, _get_conn
    init_device(device)
    conn = _get_conn()
    conn.execute(
        "UPDATE device_credits SET credits_remaining = 999, credits_used = 0 WHERE device_id = ?",
        (device,),
    )
    conn.commit()

    # Tight single-char tables trigger the OCR quality gate → escalate to Gemini.
    # Mock Gemini to return a clean table so the pipe formatting is still verified.
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": "| x | y | z |\n|---|---|---|\n| 1 | 5 | 9 |\n| 2 | 6 | 10 |\n| 3 | 7 | 11 |\n| 4 | 8 | 12 |"})())
    monkeypatch.setattr(vision_service, "model", mock_model)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("tight.png", sample_tight_table_image, "image/png")},
            data={"context": json.dumps({}), "deviceId": device},
        )
    assert resp.status_code == 200
    parsed = ExtractionResponse(**resp.json())
    assert parsed.type in (ExtractionType.TABLE, ExtractionType.TEXT)
    assert "|" in parsed.markdown
    assert parsed.creditsUsed == 5  # OCR gate escalated to Gemini


# ── Fixture 4: Diagram extraction (structured study notes) ──

@pytest.mark.asyncio
async def test_diagram_extraction(sample_diagram_image, mock_study_notes):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/diagram",
            files={"image": ("diagram.png", sample_diagram_image, "image/png")},
            data={"context": json.dumps({"title": "Sys Arch Diagram"}), "deviceId": TEST_DEVICE_ID},
        )
    assert resp.status_code == 200
    parsed = ExtractionResponse(**resp.json())
    assert parsed.type == ExtractionType.DIAGRAM
    assert parsed.studyNotes is not None
    assert parsed.studyNotes.topic.title == "System Architecture"
    assert parsed.studyNotes.what_you_should_remember
    assert parsed.studyNotes.key_formulas
    assert parsed.studyNotes.understand_it
    assert parsed.studyNotes.thirty_second_revision
    assert parsed.studyNotes.visual_context.present is True
    assert parsed.studyNotes.uncertainties
    assert "# SYSTEM ARCHITECTURE" in parsed.markdown
    assert "## 🎯 What You Should Remember" in parsed.markdown
    assert "## 📦 Key Formulas" in parsed.markdown
    assert "## 🧠 Understand It" in parsed.markdown
    assert "## ⏱️ 30-Second Revision" in parsed.markdown
    assert parsed.creditsUsed == 5
    assert mock_study_notes.call_count == 1


# ── Fixture 5: Oversized image → ImageTooLargeError ──

@pytest.mark.asyncio
async def test_oversized_image(oversized_image):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("big.png", oversized_image, "image/png")},
            data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
        )
    assert resp.status_code == 413
    body = resp.json()
    assert "error" in body
    assert body["code"] == 413


# ── Exception: InvalidInputError (malformed image) ──

@pytest.mark.asyncio
async def test_invalid_image(invalid_image, mock_vision):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("bad.png", invalid_image, "image/png")},
            data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["code"] == 422


# ── Exception: UpstreamError (Vision LLM failure) ──

@pytest.mark.asyncio
async def test_vision_upstream_error(sample_diagram_image, mock_vision_failure):
    from app.utils.credits_store import init_device

    device = "upstream-test-device-00000000-0000-0000-000000000002"
    init_device(device)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/diagram",
            files={"image": ("diagram.png", sample_diagram_image, "image/png")},
            data={"context": json.dumps({}), "deviceId": device},
        )
    assert resp.status_code == 502
    body = resp.json()
    assert "error" in body
    assert body["code"] == 502


# ── Exception: AuthError (invalid Firebase token) ──

@pytest.mark.asyncio
async def test_auth_error():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/auth/google",
            json={"idToken": "invalid-token"},
        )
    assert resp.status_code == 401
    body = resp.json()
    assert "error" in body
    assert body["code"] == 401


# ── Exception: CreditLimitError ──

TEST_ZERO_CREDIT_DEVICE = "zero-credit-device-00000000-0000-0000-0000-000000000000"


@pytest.mark.asyncio
async def test_credit_limit_error():
    from app.utils.credits_store import _get_conn, init_device
    init_device(TEST_ZERO_CREDIT_DEVICE)
    conn = _get_conn()
    conn.execute("UPDATE device_credits SET credits_remaining = 0 WHERE device_id = ?", (TEST_ZERO_CREDIT_DEVICE,))
    conn.commit()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("test.png", b"dummy", "image/png")},
            data={"context": json.dumps({}), "deviceId": TEST_ZERO_CREDIT_DEVICE},
        )
    assert resp.status_code == 429
    body = resp.json()
    assert "error" in body
    assert body["code"] == 429


# ── Device auth test ──

@pytest.mark.asyncio
async def test_device_auth():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/auth/device",
            json={"deviceId": "new-device-for-test"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["deviceId"] == "new-device-for-test"
    assert data["creditsRemaining"] == 50
    assert data["creditsUsed"] == 0
    assert data["plan"] == "free"


# ── Concurrency test ──

@pytest.mark.asyncio
async def test_concurrent_extraction(sample_text_image, sample_tight_table_image, mock_vision, mock_study_notes, monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import vision_service

    # Tight single-char table escalates to Gemini; mock it with a valid table response.
    async def _fake(prompt, *a, **k):
        if "You extract study notes" in prompt:
            return type("o", (), {"text": "| x | y |\n|---|---|\n| 1 | 5 |"})()
        return type("o", (), {"text": "## Clean notes\n- mean\n- median"})()

    mock_model = AsyncMock()
    mock_model.generate_content_async = _fake
    monkeypatch.setattr(vision_service, "model", mock_model)

    async with AsyncClient(transport=transport, base_url="http://test") as client:

        async def do_text():
            resp = await client.post(
                "/api/extract/text",
                files={"image": ("t.png", sample_text_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
            )
            return resp

        async def do_table():
            resp = await client.post(
                "/api/extract/text",
                files={"image": ("tbl.png", sample_tight_table_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": TEST_DEVICE_ID},
            )
            return resp

        results = await asyncio.gather(do_text(), do_table(), do_text(), do_table(), do_text())
        assert all(r.status_code == 200 for r in results)
        parsed = [ExtractionResponse(**r.json()) for r in results]
        assert all(p.creditsUsed >= 1 for p in parsed)
        assert all(len(p.markdown) > 0 for p in parsed)
