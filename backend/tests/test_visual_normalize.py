"""Generative payload normalization (production hotfix).

Pollinations serves raw bytes in whatever raster format the model returns
(PNG, JPEG, sometimes WebP) but the storage backends accept PNG/JPEG/GIF
only. Unvalidated bytes used to reach the uploader and fail storage with
"Could not store the generated visual". The pipeline now normalizes to PNG
before gates and storage; these tests pin that contract.
"""
import asyncio
import io

import pytest
from PIL import Image

from app.models.schemas import VisualRenderMode, VisualSpec
from app.services.visual_service import _normalize_to_png, generate_visual
from unittest.mock import AsyncMock


def _img(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buf, format=fmt)
    return buf.getvalue()


def test_normalize_png_passthrough_identity():
    png = _img("PNG")
    assert _normalize_to_png(png) == png  # byte-identical, no re-encode


def test_normalize_jpeg_converts_to_png():
    out = _normalize_to_png(_img("JPEG"))
    assert out is not None and out[:8] == b"\x89PNG\r\n\x1a\n"


def test_normalize_webp_converts_to_png():
    try:
        webp = _img("WEBP")
    except Exception:
        pytest.skip("PIL build without WebP support")
    out = _normalize_to_png(webp)
    assert out is not None and out[:8] == b"\x89PNG\r\n\x1a\n"


def test_normalize_garbage_returns_none():
    assert _normalize_to_png(None) is None
    assert _normalize_to_png(b"") is None
    assert _normalize_to_png(b"not-an-image" * 100) is None


def test_generative_pipeline_serves_png_to_gates(monkeypatch):
    """JPEG provider bytes reach the quality gate as PNG (never raw)."""
    spec = VisualSpec(
        concept="Cell",
        render_mode=VisualRenderMode.GENERATIVE,
        text_required=False,
        visual_form="illustration",
    )
    seen: dict = {}

    def _gate(png: bytes) -> bool:
        seen["magic"] = png[:8]
        return True

    monkeypatch.setattr(
        "app.services.visual_service._render_once", AsyncMock(return_value=_img("JPEG")))
    monkeypatch.setattr("app.services.visual_service._quality_pass", _gate)
    result = asyncio.run(generate_visual(spec))
    assert result is not None and result[0] == "png"
    assert seen["magic"] == b"\x89PNG\r\n\x1a\n"
