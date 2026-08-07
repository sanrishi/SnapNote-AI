import json
from unittest.mock import AsyncMock

import pytest

from app.models.schemas import StudyNotes
from app.services.vision_service import extract_study_notes, _clean_latex_in_dict
from app.utils.latex_clean import latex_to_unicode


# ── Unit: latex_to_unicode ──

def test_greek_and_times():
    assert latex_to_unicode(r"\tau_0 = r \times F") == "τ₀ = r × F"
    assert latex_to_unicode(r"\omega = \omega_0 + \alpha t") == "ω = ω₀ + α t"


def test_fraction():
    assert latex_to_unicode(r"\frac{a}{b}") == "a/b"
    assert latex_to_unicode(r"\mu = \frac{F}{N}") == "μ = F/N"


def test_dollar_wrapper_stripped():
    assert latex_to_unicode("$x^2$") == "x²"


def test_plain_text_unchanged():
    assert latex_to_unicode("x = 2") == "x = 2"


def test_no_backslash_survives():
    result = latex_to_unicode(
        r"\tau_0 = r \times F + \omega r \sin\theta \hat{r}_1 \frac{\partial v}{\partial t}"
    )
    assert "\\" not in result


def test_mixed_unicode_passthrough():
    assert latex_to_unicode("ω = 3 rad/s") == "ω = 3 rad/s"


# ── Unit: _clean_latex_in_dict recursion ──

def test_clean_dict_recursive():
    raw = {
        "topic": {"title": "Torque", "is_probable": False},
        "what_you_should_remember": "Torque from force at a distance.",
        "key_formulas": [
            {"formula": r"\tau_0 = r \times F", "explanation": "distance times force",
             "uncertain_symbols": [r"\tau"], "confidence": "context_needed"}
        ],
        "verify_before_studying": [r"\tau may be misread as 't'"],
    }
    cleaned = _clean_latex_in_dict(raw)
    assert cleaned["key_formulas"][0]["formula"] == "τ₀ = r × F"
    assert cleaned["key_formulas"][0]["uncertain_symbols"] == ["τ"]
    assert cleaned["verify_before_studying"][0].startswith("τ")
    assert "\\" not in json.dumps(cleaned, ensure_ascii=False)


# ── Integration: pipeline cleans LaTeX before validation ──

@pytest.mark.asyncio
async def test_extract_study_notes_strips_latex(monkeypatch):
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, format="PNG")

    payload = {
        "topic": {"title": "Rotational Dynamics", "is_probable": False},
        "what_you_should_remember": "Torque rotates an object.",
        "key_formulas": [
            {"formula": r"\tau_0 = r \times F", "explanation": "distance times force",
             "uncertain_symbols": [], "confidence": "clear"},
            {"formula": r"\omega = \omega_0 + \alpha t", "explanation": "velocity grows with time",
             "uncertain_symbols": [], "confidence": "clear"},
        ],
        "understand_it": ["Torque is the rotational analogue of force."],
        "common_mistakes": ["General thing to watch for: confusing torque with force."],
        "thirty_second_revision": ["τ = r × F"],
        "visual_context": {"present": False, "summary": ""},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(payload)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    notes: StudyNotes = await extract_study_notes(buf.getvalue())
    assert notes.key_formulas[0].formula == "τ₀ = r × F"
    assert notes.key_formulas[1].formula == "ω = ω₀ + α t"
    assert all("\\" not in f.formula for f in notes.key_formulas)
