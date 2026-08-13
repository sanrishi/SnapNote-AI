"""Unit tests for the hybrid Explain Visually renderer and dispatcher.

Deterministic SVG renderer: exact text/symbols rendered by code, sanitized,
byte-deterministic. Hybrid dispatcher: render_mode routes to SVG (deterministic)
or Pollinations (generative) with a conditional OCR legibility gate.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.models.schemas import DeterministicVisual, VisualEquation, VisualRenderMode, VisualSpec
from app.utils.visual_renderer import render_deterministic_visual


# ── Deterministic renderer ──


def _sample_deterministic() -> DeterministicVisual:
    return DeterministicVisual(
        title="Rotational Dynamics",
        equations=[
            VisualEquation(expression="τ = r × F", meaning="torque = force × lever arm"),
            VisualEquation(expression="L = Iω", meaning="angular momentum = inertia × angular velocity"),
            VisualEquation(expression="ΔL = ∫ τ dt", meaning="impulse-momentum theorem"),
        ],
        steps=["Identify the axis", "Compute torque", "Relate to angular momentum"],
        points=["Torque is the rotational analogue of force", "Angular momentum is conserved without net torque"],
    )


def test_renderer_produces_sanitized_svg_with_exact_content():
    svg = render_deterministic_visual(_sample_deterministic())
    assert svg.startswith("<svg")
    assert svg.strip().endswith("</svg>")
    for token in ["Rotational Dynamics", "τ = r × F", "L = Iω", "ΔL = ∫ τ dt", "torque = force × lever arm"]:
        assert token in svg
    # Sanitized: no scripts, no event handlers, no external refs
    assert "<script" not in svg.lower()
    assert "onerror" not in svg.lower()
    assert "javascript:" not in svg.lower()


def test_renderer_is_byte_deterministic():
    a = render_deterministic_visual(_sample_deterministic())
    b = render_deterministic_visual(_sample_deterministic())
    assert a == b


def test_renderer_empty_spec_returns_empty():
    assert render_deterministic_visual(DeterministicVisual()) == ""


def test_renderer_escapes_html_in_text():
    spec = DeterministicVisual(
        title="a < b & c",
        equations=[VisualEquation(expression="x < y", meaning="means & implies")],
    )
    svg = render_deterministic_visual(spec)
    assert "&lt;" in svg
    assert "&amp;" in svg
    assert "<script" not in svg.lower()


# ── Hybrid dispatcher ──


def test_generate_visual_deterministic_returns_svg(monkeypatch):
    spec = VisualSpec(
        concept="Rotation",
        render_mode=VisualRenderMode.DETERMINISTIC,
        deterministic=_sample_deterministic(),
    )
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=b"PNG"))
    result = asyncio.run(_run_generate(spec))
    assert result is not None
    mode, payload = result
    assert mode == "svg"
    assert isinstance(payload, str)
    assert "τ = r × F" in payload
    # Deterministic must never hit the generative renderer
    from app.services import visual_service
    visual_service._render_once.assert_not_called()


def test_generate_visual_generative_returns_png_when_legible(monkeypatch):
    import io

    from PIL import Image

    spec = VisualSpec(
        concept="Cell",
        render_mode=VisualRenderMode.GENERATIVE,
        text_required=False,
        visual_form="illustration",
    )
    buf = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=png))
    monkeypatch.setattr("app.services.visual_service._quality_pass", lambda p: True)
    result = asyncio.run(_run_generate(spec))
    assert result is not None
    mode, payload = result
    assert mode == "png"
    assert payload == png


def test_generate_visual_generative_rejects_unreadable_text(monkeypatch):
    """OCR legibility gate: when text_required, an unreadable render is rejected."""
    import io

    from PIL import Image

    spec = VisualSpec(
        concept="Cell",
        render_mode=VisualRenderMode.GENERATIVE,
        text_required=True,
        visual_form="labeled diagram",
    )
    buf = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=png))
    monkeypatch.setattr("app.services.visual_service._quality_pass", lambda p: True)
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: True)
    result = asyncio.run(_run_generate(spec))
    assert result is None


def test_generate_visual_generative_skips_ocr_when_not_required(monkeypatch):
    """text_required=false must NOT reject an illustration that has no text."""
    import io

    from PIL import Image

    spec = VisualSpec(
        concept="Cell",
        render_mode=VisualRenderMode.GENERATIVE,
        text_required=False,
        visual_form="illustration",
    )
    buf = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=png))
    monkeypatch.setattr("app.services.visual_service._quality_pass", lambda p: True)
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: True)
    result = asyncio.run(_run_generate(spec))
    assert result is not None
    assert result[0] == "png"


def test_legibility_pass_thresholds(monkeypatch):
    import io

    from PIL import Image
    from app.services import visual_service

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: True)

    # A few high-confidence tokens -> pass
    monkeypatch.setattr(
        "app.services.ocr_service.read_raw",
        lambda arr: [[[[0, 0], [0, 0], [0, 0], [0, 0]], "torque", 0.95], [[[0, 0], [0, 0], [0, 0], [0, 0]], "axis", 0.9]],
    )
    assert visual_service._legibility_pass(png) is True

    # Zero tokens -> fail
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    assert visual_service._legibility_pass(png) is False

    # Too few tokens -> fail
    monkeypatch.setattr(
        "app.services.ocr_service.read_raw",
        lambda arr: [[[[0, 0], [0, 0], [0, 0], [0, 0]], "x", 0.95]],
    )
    assert visual_service._legibility_pass(png) is False

    # OCR unavailable -> pass (safety net, never hard-break the feature)
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: False)
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    assert visual_service._legibility_pass(png) is True


async def _run_generate(spec):
    from app.services.visual_service import generate_visual

    return await generate_visual(spec)