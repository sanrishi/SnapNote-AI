import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.models.schemas import DiagramSpec, PolarBounds, StudyNotes
from app.utils.diagram_validation import validate_diagram_spec
from app.utils.diagram_renderer import render_polar_region
from app.utils.math_normalize import display_math, math_value, normalize_math
from app.utils.svg_safe import sanitize_svg


# ── A. Math normalization ──

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", "1"),
        ("sqrt5", "sqrt(5)"),
        ("sqrt(5)", "sqrt(5)"),
        ("√5", "sqrt(5)"),
        ("sqrt{5}", "sqrt(5)"),
        ("2pi", "2*pi"),
        ("2π", "2*pi"),
        ("2*pi", "2*pi"),
        ("2 π", "2*pi"),
        ("pi/2", "pi/2"),
        ("0", "0"),
        ("3", "3"),
        ("3sqrt(2)", "3*sqrt(2)"),
    ],
)
def test_normalize_math_forms(raw, expected):
    assert normalize_math(raw) == expected


@pytest.mark.parametrize("raw", ["", "x=143", "hello world", "r = 1", "sqrt()", "1..2"])
def test_normalize_math_invalid(raw):
    assert normalize_math(raw) is None


def test_normalize_math_values():
    assert math_value("sqrt(5)") == pytest.approx(2.2360679)
    assert math_value("2*pi") == pytest.approx(6.2831853)
    assert math_value("pi/2") == pytest.approx(1.5707963)
    assert math_value("0") == 0.0


def test_display_math():
    assert display_math("sqrt(5)") == "√5"
    assert display_math("2*pi") == "2π"
    assert display_math("pi/2") == "π/2"
    assert display_math("1") == "1"


# ── B. Semantic validation (before rendering) ──

def _spec(**over):
    defaults = dict(
        present=True,
        diagram_type="polar_region",
        bounds=PolarBounds(inner="1", outer="sqrt(5)", theta_min="0", theta_max="2*pi"),
        labels=["r = 1", "r = sqrt(5)"],
        instruction_text=["region between the circles"],
    )
    defaults.update(over)
    return DiagramSpec(**defaults)


def test_validate_valid_spec():
    result = validate_diagram_spec(_spec())
    assert result.valid
    assert result.canonical is not None
    assert result.canonical.inner == "1"
    assert result.canonical.outer == "sqrt(5)"
    assert result.canonical.inner_value == pytest.approx(1.0)
    assert result.canonical.outer_value == pytest.approx(2.2360679)


def test_validate_equivalent_spec_forms_same_canonical():
    a = validate_diagram_spec(_spec(bounds=PolarBounds(inner="1", outer="√5", theta_min="0", theta_max="2π")))
    b = validate_diagram_spec(_spec(bounds=PolarBounds(inner="1", outer="sqrt5", theta_min="0", theta_max="2pi")))
    assert a.valid and b.valid
    assert a.canonical == b.canonical


def test_validate_present_false_invalid():
    result = validate_diagram_spec(_spec(present=False))
    assert not result.valid
    assert result.reasons


def test_validate_unsupported_diagram_type():
    result = validate_diagram_spec(_spec(diagram_type="flowchart"))
    assert not result.valid
    assert any("not supported" in r for r in result.reasons)


def test_validate_inner_equals_outer():
    result = validate_diagram_spec(_spec(bounds=PolarBounds(inner="2", outer="2", theta_min="0", theta_max="2*pi")))
    assert not result.valid
    assert any("outer radius must be greater" in r for r in result.reasons)


def test_validate_inner_greater_than_outer():
    result = validate_diagram_spec(_spec(bounds=PolarBounds(inner="3", outer="1", theta_min="0", theta_max="2*pi")))
    assert not result.valid


def test_validate_missing_bound():
    result = validate_diagram_spec(_spec(bounds=PolarBounds(inner="", outer="sqrt(5)", theta_min="0", theta_max="2*pi")))
    assert not result.valid
    assert any("inner bound could not be read" in r for r in result.reasons)


def test_validate_negative_inner():
    result = validate_diagram_spec(_spec(bounds=PolarBounds(inner="-1", outer="2", theta_min="0", theta_max="2*pi")))
    assert not result.valid


def test_validate_theta_order():
    result = validate_diagram_spec(_spec(bounds=PolarBounds(inner="1", outer="2", theta_min="2*pi", theta_max="0")))
    assert not result.valid
    assert any("end angle must be greater" in r for r in result.reasons)


def test_validate_theta_beyond_full_revolution():
    result = validate_diagram_spec(_spec(bounds=PolarBounds(inner="1", outer="2", theta_min="0", theta_max="3*pi")))
    assert not result.valid
    assert any("exceeds one full revolution" in r for r in result.reasons)


# ── C. Deterministic rendering ──

def _canonical(**over):
    spec = _spec(**over)
    result = validate_diagram_spec(spec)
    assert result.valid
    return result.canonical


def test_render_deterministic_100_runs():
    canonical = _canonical()
    first = render_polar_region(canonical)
    for _ in range(100):
        assert render_polar_region(canonical) == first


def test_render_is_sanitized_and_idempotent():
    canonical = _canonical()
    svg = render_polar_region(canonical)
    assert svg
    assert "<svg" in svg
    assert "onerror" not in svg.lower()
    assert sanitize_svg(svg) == svg


def test_render_geometry():
    svg = render_polar_region(_canonical())
    assert svg.count("<circle") == 3  # origin dot + inner + outer boundary
    assert 'r="200"' in svg  # outer = sqrt(5) scaled to OUTER_MAX_PX
    assert 'r="89.44"' in svg  # inner = 1 * (200 / 2.236)
    assert "<polygon" in svg  # arrowheads
    assert "<line" in svg  # axes
    assert 'fill="#a78bfa"' in svg and 'opacity="0.2"' in svg  # shaded annulus


def test_render_annulus_has_two_subpaths():
    svg = render_polar_region(_canonical())
    assert svg.count("M ") >= 2  # outer ring + inner ring under nonzero winding


def test_render_labels_derived_from_geometry():
    svg = render_polar_region(_canonical())
    assert "r = 1" in svg
    assert "r = √5" in svg


def test_render_labels_independent_of_gemini_text():
    with_labels = render_polar_region(_canonical(labels=["g1(theta)=1", "g2(theta)=sqrt5", "theta"]))
    no_labels = render_polar_region(_canonical(labels=[]))
    assert with_labels == no_labels  # text is derived from geometry, never from Gemini prose


def test_render_caption_derived_from_geometry():
    svg = render_polar_region(_canonical())
    assert "region between r = 1 and r = √5" in svg


def test_render_full_ring_shows_theta_range():
    svg = render_polar_region(_canonical())
    assert "θ from 0 to 2π" in svg


def test_render_partial_theta_caption():
    canonical = _canonical(bounds=PolarBounds(inner="1", outer="2", theta_min="0", theta_max="pi/2"))
    svg = render_polar_region(canonical)
    assert "θ from 0 to π/2" in svg


def test_render_partial_theta_annotation():
    canonical = _canonical(bounds=PolarBounds(inner="1", outer="2", theta_min="0", theta_max="pi/2"))
    svg = render_polar_region(canonical)
    assert "θ" in svg
    assert 'stroke-dasharray="4 4"' in svg


def test_render_full_theta_no_annotation():
    svg = render_polar_region(_canonical())
    assert 'stroke-dasharray="4 4"' not in svg


def test_render_caption_no_overlap():
    svg = render_polar_region(_canonical(instruction_text=["region between the circles"]))
    assert "region between the circles" not in svg  # Gemini prose never rendered


# ── H. Malformed specs fail safely (through the full pipeline) ──

@pytest.mark.asyncio
async def test_pipeline_spec_success(sample_diagram_image, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DIAGRAM_RENDERER_MODE", "semantic")
    payload = {
        "topic": {"title": "Polar Area", "is_probable": False},
        "what_you_should_remember": "The annulus area is π(b²−a²).",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": True, "summary": "The shaded ring is the region between two circles."},
        "diagram": {"present": False, "svg": ""},
        "diagram_spec": {
            "present": True,
            "diagram_type": "polar_region",
            "bounds": {"inner": "1", "outer": "sqrt(5)", "theta_min": "0", "theta_max": "2*pi"},
            "show_axes": True,
            "labels": ["r = 1", "r = sqrt(5)"],
            "shade_region": True,
            "instruction_text": ["region between the circles"],
            "uncertain": ["outer radius could be sqrt(5) or sqrt(3)"],
        },
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    from app.services.vision_service import extract_study_notes

    notes = await extract_study_notes(sample_diagram_image)
    assert notes.diagram.present is True
    assert "<svg" in notes.diagram.svg
    assert any("outer radius could be sqrt(5)" in v for v in notes.verify_before_studying)


@pytest.mark.asyncio
async def test_pipeline_spec_unsupported_type_empty_fallback_goes_explanation_only(sample_diagram_image, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DIAGRAM_RENDERER_MODE", "semantic")
    primary_payload = {
        "topic": {"title": "Flow", "is_probable": False},
        "what_you_should_remember": "",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": False, "summary": ""},
        "diagram": {"present": False, "svg": ""},
        "diagram_spec": {
            "present": True,
            "diagram_type": "flowchart",
            "bounds": {"inner": "", "outer": "", "theta_min": "", "theta_max": ""},
            "labels": [],
            "instruction_text": [],
            "uncertain": [],
        },
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }

    async def _fake_gemini(prompt, **kwargs):
        text = json.dumps(primary_payload)
        return type("o", (), {"text": text})()

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=_fake_gemini)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    from app.services.vision_service import extract_study_notes

    notes = await extract_study_notes(sample_diagram_image)
    assert notes.diagram.present is False
    assert notes.diagram.svg == ""
    assert notes.diagram.best_effort is False
    # Unsupported types now go explanation-only with no second Gemini call (no best-effort garble)
    assert mock_model.generate_content_async.await_count == 1


@pytest.mark.asyncio
async def test_pipeline_spec_unsupported_type_best_effort_fallback(sample_diagram_image, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DIAGRAM_RENDERER_MODE", "semantic")
    primary_payload = {
        "topic": {"title": "Flow", "is_probable": False},
        "what_you_should_remember": "",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": False, "summary": ""},
        "diagram": {"present": False, "svg": ""},
        "diagram_spec": {
            "present": True,
            "diagram_type": "flowchart",
            "bounds": {"inner": "", "outer": "", "theta_min": "", "theta_max": ""},
            "labels": [],
            "instruction_text": [],
            "uncertain": [],
        },
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }

    async def _fake_gemini(prompt, **kwargs):
        return type("o", (), {"text": json.dumps(primary_payload)})()

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=_fake_gemini)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    from app.services.vision_service import extract_study_notes

    notes = await extract_study_notes(sample_diagram_image)
    # Unsupported types now go explanation-only with no best-effort garble (second Gemini call removed)
    assert notes.diagram.present is False
    assert notes.diagram.best_effort is False
    assert notes.diagram.svg == ""
    assert mock_model.generate_content_async.await_count == 1


@pytest.mark.asyncio
async def test_pipeline_spec_supported_type_calls_gemini_once(sample_diagram_image, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DIAGRAM_RENDERER_MODE", "semantic")
    payload = {
        "topic": {"title": "Polar", "is_probable": False},
        "what_you_should_remember": "",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": False, "summary": ""},
        "diagram": {"present": False, "svg": ""},
        "diagram_spec": {
            "present": True,
            "diagram_type": "polar_region",
            "bounds": {"inner": "1", "outer": "sqrt(5)", "theta_min": "0", "theta_max": "2*pi"},
            "labels": [],
            "instruction_text": [],
            "uncertain": [],
        },
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    from app.services.vision_service import extract_study_notes

    notes = await extract_study_notes(sample_diagram_image)
    assert notes.diagram.present is True
    assert notes.diagram.best_effort is False
    assert mock_model.generate_content_async.await_count == 1  # no second fallback call


@pytest.mark.asyncio
async def test_legacy_fallback_malformed_returns_empty(sample_diagram_image, monkeypatch):
    from app.services.vision_service import _legacy_diagram_fallback

    async def _fake_gemini(prompt, **kwargs):
        return type("o", (), {"text": "this is not json {"})()

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=_fake_gemini)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    rep = await _legacy_diagram_fallback(sample_diagram_image)
    assert rep.present is False
    assert rep.svg == ""
    assert rep.best_effort is False


@pytest.mark.asyncio
async def test_legacy_fallback_present_false_svg_empty(sample_diagram_image, monkeypatch):
    from app.services.vision_service import _legacy_diagram_fallback

    async def _fake_gemini(prompt, **kwargs):
        return type("o", (), {"text": json.dumps({"present": False, "svg": ""})})()

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=_fake_gemini)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    rep = await _legacy_diagram_fallback(sample_diagram_image)
    assert rep.present is False
    assert rep.svg == ""


@pytest.mark.asyncio
async def test_pipeline_spec_missing_bounds_no_diagram(sample_diagram_image, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DIAGRAM_RENDERER_MODE", "semantic")
    payload = {
        "topic": {"title": "Partial", "is_probable": False},
        "what_you_should_remember": "",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": False, "summary": ""},
        "diagram": {"present": False, "svg": ""},
        "diagram_spec": {
            "present": True,
            "diagram_type": "polar_region",
            "bounds": {"inner": "", "outer": "", "theta_min": "0", "theta_max": "2*pi"},
            "labels": [],
            "instruction_text": [],
            "uncertain": [],
        },
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=type("o", (), {"text": json.dumps(payload)})())
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    from app.services.vision_service import extract_study_notes

    notes = await extract_study_notes(sample_diagram_image)
    assert notes.diagram.present is False
    assert notes.diagram.svg == ""
    assert any("could not be read" in u for u in notes.uncertainties)


# ── Backward compatibility ──

def test_legacy_diagram_still_validates():
    notes = StudyNotes(
        topic={"title": "T", "is_probable": False},
        diagram={"present": True, "svg": '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="1" cy="1" r="2"/></svg>'},
    )
    assert notes.diagram.present is True
    assert notes.diagram_spec is None


def test_diagram_spec_excluded_from_serialization():
    spec = _spec()
    notes = StudyNotes(diagram_spec=spec)
    dumped = json.loads(notes.model_dump_json())
    assert "diagram_spec" not in dumped
    assert "diagram" in dumped
