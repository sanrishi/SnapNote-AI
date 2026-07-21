import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.models.schemas import ExtractionResponse, ExtractionType
from app.exceptions import SnapNoteError

transport = ASGITransport(app=app)


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
            data={"context": json.dumps({"title": "Stats Lecture Week 3", "url": "https://youtube.com/watch?v=123"})},
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
            data={"context": json.dumps({})},
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
async def test_tight_table_extraction(sample_tight_table_image, mock_vision):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("tight.png", sample_tight_table_image, "image/png")},
            data={"context": json.dumps({})},
        )
    assert resp.status_code == 200
    parsed = ExtractionResponse(**resp.json())
    assert parsed.type in (ExtractionType.TABLE, ExtractionType.TEXT)
    assert "|" in parsed.markdown
    assert parsed.creditsUsed == 1
    assert mock_vision.call_count == 0


# ── Fixture 4: Diagram extraction (Vision LLM path) ──

@pytest.mark.asyncio
async def test_diagram_extraction(sample_diagram_image, mock_vision):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/diagram",
            files={"image": ("diagram.png", sample_diagram_image, "image/png")},
            data={"context": json.dumps({"title": "Sys Arch Diagram"})},
        )
    assert resp.status_code == 200
    parsed = ExtractionResponse(**resp.json())
    assert parsed.type == ExtractionType.DIAGRAM
    assert "Process A" in parsed.markdown
    # imageUrl is None when R2 is unconfigured (test env); string when configured
    assert parsed.creditsUsed == 5
    assert mock_vision.call_count == 1


# ── Fixture 5: Oversized image → ImageTooLargeError ──

@pytest.mark.asyncio
async def test_oversized_image(oversized_image):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/text",
            files={"image": ("big.png", oversized_image, "image/png")},
            data={"context": json.dumps({})},
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
            data={"context": json.dumps({})},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["code"] == 422


# ── Exception: UpstreamError (Vision LLM failure) ──

@pytest.mark.asyncio
async def test_vision_upstream_error(sample_diagram_image, mock_vision_failure):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/diagram",
            files={"image": ("diagram.png", sample_diagram_image, "image/png")},
            data={"context": json.dumps({})},
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

@pytest.mark.asyncio
async def test_credit_limit_error():
    from app.exceptions import CreditLimitError

    @app.get("/_test_credit")
    async def _test_credit():
        raise CreditLimitError()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/_test_credit")
    assert resp.status_code == 429
    body = resp.json()
    assert "error" in body
    assert body["code"] == 429


# ── Concurrency test ──

@pytest.mark.asyncio
async def test_concurrent_extraction(sample_text_image, sample_tight_table_image, mock_vision):
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        async def do_text():
            resp = await client.post(
                "/api/extract/text",
                files={"image": ("t.png", sample_text_image, "image/png")},
                data={"context": json.dumps({})},
            )
            return resp

        async def do_table():
            resp = await client.post(
                "/api/extract/text",
                files={"image": ("tbl.png", sample_tight_table_image, "image/png")},
                data={"context": json.dumps({})},
            )
            return resp

        results = await asyncio.gather(do_text(), do_table(), do_text(), do_table(), do_text())
        assert all(r.status_code == 200 for r in results)
        parsed = [ExtractionResponse(**r.json()) for r in results]
        assert all(p.creditsUsed == 1 for p in parsed)
        assert all(len(p.markdown) > 0 for p in parsed)
