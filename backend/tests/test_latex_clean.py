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


def test_verified_superscript_letters():
    assert latex_to_unicode("Kc = [C]^c [D]^d / [A]^a [B]^b") == "Kc = [C]ᶜ [D]ᵈ / [A]ᵃ [B]ᵇ"
    assert latex_to_unicode("x^2 + y^n") == "x² + yⁿ"
    # Unverified coverage stays ASCII rather than risking tofu.
    assert latex_to_unicode("x^q + y^z") == "x^q + y^z"
    # Mixed groups are all-or-nothing.
    assert latex_to_unicode("e^(-z)") == "e^(-z)"


def test_plain_text_unchanged():
    assert latex_to_unicode("x = 2") == "x = 2"


def test_no_backslash_survives():
    result = latex_to_unicode(
        r"\tau_0 = r \times F + \omega r \sin\theta \hat{r}_1 \frac{\partial v}{\partial t}"
    )
    assert "\\" not in result


def test_unknown_command_preserved_verbatim():
    """A misspelled command (e.g. Gemini's \integ for \int) must stay visible
    as \integ — never silently mangled into a wrong word ("integ")."""
    assert latex_to_unicode(r"\integ(t)dt") == "\\integ(t)dt"


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


# ── Unit: normalize_ascii_math (math slots only, explicit allowlist) ──

from app.utils.latex_clean import normalize_ascii_math, normalize_spec_math


def test_ascii_operators_and_functions():
    assert normalize_ascii_math("x = (-b ± sqrt(b² - 4ac)) / (2a)") == "x = (-b ± √(b² - 4ac)) / (2a)"
    assert normalize_ascii_math("Scarcity -> Choices -> Cost") == "Scarcity → Choices → Cost"
    assert normalize_ascii_math("Qc < Kc") == "Qc < Kc"
    assert normalize_ascii_math("a <= b and c != d") == "a ≤ b and c ≠ d"
    assert normalize_ascii_math("Sum i = n(n+1)/2") == "Σ i = n(n+1)/2"


def test_bare_word_greek_with_boundaries():
    assert normalize_ascii_math("x = r cos(theta)") == "x = r cos(θ)"
    assert normalize_ascii_math("P(t) = Kp e(t)") == "P(t) = Kp e(t)"
    assert normalize_ascii_math("2pi r") == "2π r"
    # English words containing the letters are untouched.
    assert normalize_ascii_math("spin the pie in the menu") == "spin the pie in the menu"
    assert normalize_ascii_math("Summary of alpha-beta") == "Summary of α-β"


def test_normalizer_leaves_code_and_prose_alone():
    # Single Latin letters could be variables — never guessed as Greek.
    assert normalize_ascii_math("Normal(u, s)") == "Normal(u, s)"
    # Fused identifiers are left alone rather than split mid-token.
    assert normalize_ascii_math("dtheta") == "dtheta"
    # Parenthesized exponents: partial superscripting would look worse.
    assert normalize_ascii_math("e^(-z²/2)") == "e^(-z²/2)"
    assert normalize_ascii_math("") == ""


def test_normalize_spec_math_slot_scoping():
    from app.models.schemas import (
        CompositionCallout,
        CompositionReasoningStep,
        CompositionResult,
        DeterministicVisual,
        LessonComposition,
        VisualCurve,
        VisualPlot,
        VisualRenderMode,
        VisualScene,
        VisualSpec,
    )

    spec = VisualSpec(
        concept="t",
        render_mode=VisualRenderMode.DETERMINISTIC,
        text_required=True,
        deterministic=DeterministicVisual(
            title="Sum of everything -> done",
            scene=VisualScene(
                scene_kind="plot",
                caption="plain caption",
                plot=VisualPlot(
                    x_min=0, x_max=3, y_min=0, y_max=3,
                    curves=[VisualCurve(label="r = sqrt(2)", expr="sqrt(2)")],
                ),
            ),
            composition=LessonComposition(
                title="t",
                framing="f",
                callouts=[CompositionCallout(id="c1", label="theta", value="0 to 2pi")],
                reasoning=[CompositionReasoningStep(id="r1", expression="dA = r dr d theta", explanation="e")],
                result=CompositionResult(expression="x = (-b ± sqrt(b²-4ac)) / (2a)"),
                takeaway="take",
            ),
        ),
    )
    normalize_spec_math(spec)
    det = spec.deterministic
    assert det.scene.plot.curves[0].label == "r = √(2)"
    # Evaluator code is untouched.
    assert det.scene.plot.curves[0].expr == "sqrt(2)"
    comp = det.composition
    assert comp is not None
    assert comp.reasoning[0].expression == "dA = r dr d θ"
    assert comp.result is not None and comp.result.expression == "x = (-b ± √(b²-4ac)) / (2a)"
    assert comp.callouts[0].label == "θ"
    assert comp.callouts[0].value == "0 to 2π"
    # Prose slots pass through byte-identical.
    assert det.title == "Sum of everything -> done"
    assert det.scene.caption == "plain caption"
