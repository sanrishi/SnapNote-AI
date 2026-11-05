"""Question-first Hinglish explanation: schema, trust gates, rendering, routes.

The feature's contract:
* Hinglish is the default explanation language, English is opt-in.
* The explanation is produced in the SAME Gemini call as the study notes.
* A worked interval inequality is re-solved by the exact solver; a wrong claim
  is repaired once, and otherwise the worked steps are withheld so a wrong
  answer can never be presented as verified.
* The original question stays visible and the formal exam answer stays English.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from httpx import ASGITransport, AsyncClient

from app.exceptions import UpstreamError
from app.main import app
from app.models.schemas import (
    ExplanationLanguage,
    StudyExplanation,
    StudyNotes,
)
from app.services import vision_service
from app.utils.explanation_verify import check_explanation_math
from app.utils.render_notes import render_study_notes

transport = ASGITransport(app=app)


def _explanation_payload(**overrides: object) -> dict:
    payload: dict = {
        "question_text": "Find all x for which x² − 5x + 6 = 0.",
        "question_meaning": "Question ye pooch raha hai ki kaunsi x values equation ko true banati hain.",
        "what_is_asked": ["x ki values find karni hain."],
        "concept": ["Quadratic equation ko factorise karke roots nikalte hain."],
        "steps": [
            {
                "title": "Factorise karo",
                "explanation": "Do numbers dhoondo jinka product 6 aur sum -5 ho.",
                "equation": "(x - 2)(x - 3) = 0",
            }
        ],
        "exam_answer": "x = 2 or x = 3",
        "recap": ["Roots 2 aur 3 hain."],
        "confidence": "clear",
        "problem_expr": "",
        "solution_set": "",
    }
    payload.update(overrides)
    return payload


def _notes_payload(explanation: dict | None) -> dict:
    return {
        "topic": {"title": "Quadratic Equations", "is_probable": False},
        "explanation": explanation,
    }


def _seed_device(device_id: str, amount: int = 50) -> None:
    from app.utils.credits_store import _get_conn, init_device

    init_device(device_id)
    conn = _get_conn()
    conn.execute(
        "UPDATE device_credits SET credits_remaining = ?, credits_used = 0 WHERE device_id = ?",
        (amount, device_id),
    )
    conn.execute("DELETE FROM request_log WHERE device_id = ?", (device_id,))
    conn.commit()


def _balance(device_id: str) -> int:
    from app.utils.credits_store import get_credits

    remaining, _ = get_credits(device_id)
    return remaining


def test_explanation_defaults_to_hinglish_and_coerces_model_shapes():
    explanation = StudyExplanation(
        question_text="Question",
        what_is_asked="Find x",
        concept="A concept",
        steps="Do the operation",
        confidence="needs review",
        language=" HINGLISH ",
    )
    assert explanation.language == ExplanationLanguage.HINGLISH
    assert explanation.what_is_asked == ["Find x"]
    assert explanation.concept == ["A concept"]
    assert explanation.steps[0].explanation == "Do the operation"
    assert explanation.confidence == "review"


def test_empty_explanation_is_dropped_from_study_notes():
    notes = StudyNotes(explanation=StudyExplanation())
    assert notes.explanation is None


def test_null_lists_from_the_model_do_not_fail_the_whole_explanation():
    explanation = StudyExplanation(
        question_text="Question",
        what_is_asked=None,
        concept=None,
        steps=None,
        recap=None,
        question_meaning=None,
        exam_answer=None,
        problem_expr=None,
        solution_set=None,
        confidence=None,
    )
    assert explanation.what_is_asked == []
    assert explanation.concept == []
    assert explanation.steps == []
    assert explanation.recap == []
    assert explanation.question_meaning == ""
    assert explanation.exam_answer == ""
    assert explanation.confidence == "review"


def test_explanation_keeps_source_question_and_notation_exactly():
    source = (
        "At 500 K, H₂(g) + I₂(g) ⇌ 2 HI(g), Kc = 49.0. "
        "Calculate [HI] at equilibrium. Give units."
    )
    explanation = StudyExplanation(question_text=source)
    assert explanation.question_text == source
    assert "⇌" in explanation.question_text
    assert "[HI]" in explanation.question_text


def test_prompts_require_question_first_and_machine_checkable_math():
    for prompt in (
        vision_service.STUDY_NOTES_SYSTEM_PROMPT,
        vision_service.STUDY_NOTES_SEMANTIC_PROMPT,
        vision_service.REVISION_SYSTEM_PROMPT,
    ):
        assert "EXPLANATION BLOCK" in prompt
        assert "alter the question" in prompt
        assert "DIGIT FOR DIGIT" in prompt
        assert "formal answer in ENGLISH" in prompt
        assert "problem_expr" in prompt
        assert "solution_set" in prompt
    assert "natural Indian Hinglish" in vision_service._HINGLISH_LANGUAGE_SUFFIX
    assert "clear simple English" in vision_service._ENGLISH_LANGUAGE_SUFFIX


def test_exact_solver_rejects_the_known_wrong_log_solution():
    explanation = StudyExplanation(
        question_text="log base (x+3) of (x² - x) < 1",
        problem_expr="log_{x+3}(x^2 - x) < 1",
        solution_set="(-3, -2) U (-1, 3)",
    )
    check = check_explanation_math(explanation)
    assert check.status == "mismatch"
    assert check.expected == "(-3, -2) ∪ (-1, 0) ∪ (1, 3)"


def test_exact_solver_accepts_the_correct_log_solution():
    explanation = StudyExplanation(
        question_text="log base (x+3) of (x² - x) < 1",
        problem_expr="log_{x+3}(x^2 - x) < 1",
        solution_set="(-3, -2) U (-1, 0) U (1, 3)",
    )
    check = check_explanation_math(explanation)
    assert check.status == "verified"
    assert check.expected == "(-3, -2) ∪ (-1, 0) ∪ (1, 3)"


def test_exact_solver_accepts_natural_log_base_words_safely():
    """A model paraphrase of the base marker must not bypass verification."""
    explanation = StudyExplanation(
        question_text="log base (x+3) of (x² - x) < 1",
        problem_expr="log base (x+3) of (x² - x) < 1",
        solution_set="(-3, -2) U (-1, 0) U (1, 3)",
    )
    check = check_explanation_math(explanation)
    assert check.status == "verified"
    assert check.expected == "(-3, -2) ∪ (-1, 0) ∪ (1, 3)"


def test_exact_solver_accepts_no_real_solutions_phrase():
    explanation = StudyExplanation(
        question_text="Solve x - x < 0.",
        problem_expr="x - x < 0",
        solution_set="no real solutions",
    )
    check = check_explanation_math(explanation)
    assert check.status == "verified"
    assert check.expected == "∅"


def test_non_inequality_problem_expr_does_not_downgrade_confidence():
    explanation = StudyExplanation(
        question_text="A body moves with v = u + at.",
        problem_expr="v = u + at",
        confidence="clear",
    )
    check = check_explanation_math(explanation)
    assert check.status == "not_applicable"

    chemistry = StudyExplanation(
        question_text="Which direction does the reaction proceed?",
        problem_expr="Q < Kc",
        confidence="clear",
    )
    assert check_explanation_math(chemistry).status == "not_applicable"


def test_unsupported_math_is_never_labelled_verified():
    explanation = StudyExplanation(
        question_text="Solve sin(x) < 1.",
        problem_expr="sin(x) < 1",
        solution_set="(0, 1)",
        confidence="clear",
    )
    result = asyncio.run(_run_verification(explanation))
    assert result.verified is False
    assert result.confidence == "review"
    assert "verify" in result.verification_note.casefold()


async def _run_verification(explanation: StudyExplanation) -> StudyExplanation:
    notes = StudyNotes(explanation=explanation)
    await vision_service._verify_explanation(
        notes,
        b"image",
        ExplanationLanguage.HINGLISH,
        time.monotonic() + 30.0,
    )
    assert notes.explanation is not None
    return notes.explanation


def test_extract_study_notes_defaults_to_hinglish_in_one_paid_call(monkeypatch):
    prompts: list[str] = []

    async def fake_call(prompt: str, image_bytes: bytes, json_mode: bool = False) -> str:
        prompts.append(prompt)
        return json.dumps(_notes_payload(_explanation_payload()))

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    notes = asyncio.run(vision_service.extract_study_notes(b"image"))

    assert len(prompts) == 1
    assert "natural Indian Hinglish" in prompts[0]
    assert notes.explanation is not None
    assert notes.explanation.language == ExplanationLanguage.HINGLISH
    assert notes.explanation.verified is False


def test_extract_study_notes_english_is_opt_in(monkeypatch):
    prompts: list[str] = []

    async def fake_call(prompt: str, image_bytes: bytes, json_mode: bool = False) -> str:
        prompts.append(prompt)
        return json.dumps(_notes_payload(_explanation_payload()))

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    notes = asyncio.run(
        vision_service.extract_study_notes(
            b"image", explanation_language=ExplanationLanguage.ENGLISH
        )
    )

    assert "clear simple English" in prompts[0]
    assert "natural Indian Hinglish" not in prompts[0]
    assert notes.explanation is not None
    assert notes.explanation.language == ExplanationLanguage.ENGLISH


def test_wrong_interval_claim_is_repaired_with_one_focused_call(monkeypatch):
    wrong = _explanation_payload(
        question_text="Solve log base (x+3) of (x² - x) < 1.",
        problem_expr="log_{x+3}(x^2 - x) < 1",
        solution_set="(-3, -2) U (-1, 3)",
        exam_answer="x ∈ (-3, -2) ∪ (-1, 3)",
        steps=[
            {
                "title": "Domain",
                "explanation": "Domain restrictions lagao.",
                "equation": "x² - x > 0",
            }
        ],
    )
    fixed = {
        **_explanation_payload(
            question_text="Solve log base (x+3) of (x² - x) < 1.",
            problem_expr="log_{x+3}(x^2 - x) < 1",
            solution_set="(-3, -2) U (-1, 0) U (1, 3)",
            exam_answer="x ∈ (-3, -2) ∪ (-1, 0) ∪ (1, 3)",
        ),
        "steps": [
            {
                "title": "Domain",
                "explanation": "Argument positive aur base valid rakho.",
                "equation": "x² - x > 0, x + 3 > 0, x + 3 ≠ 1",
            },
            {
                "title": "Cases",
                "explanation": "Base > 1 aur 0 < base < 1 dono cases solve karo.",
                "equation": "x ∈ (-3, -2) ∪ (-1, 0) ∪ (1, 3)",
            },
        ],
    }
    calls: list[str] = []

    async def fake_call(prompt: str, image_bytes: bytes, json_mode: bool = False) -> str:
        calls.append(prompt)
        if "VERIFIED SOLUTION SET" in prompt:
            return json.dumps(fixed)
        return json.dumps(_notes_payload(wrong))

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    notes = asyncio.run(vision_service.extract_study_notes(b"image"))

    assert len(calls) == 2
    assert "VERIFIED SOLUTION SET" in calls[1]
    assert "(-3, -2) ∪ (-1, 0) ∪ (1, 3)" in calls[1]
    assert notes.explanation is not None
    assert notes.explanation.verified is True
    assert notes.explanation.solution_set == "(-3, -2) ∪ (-1, 0) ∪ (1, 3)"
    assert notes.explanation.confidence == "clear"
    assert "(-1, 0)" in notes.explanation.exam_answer


def test_math_repair_is_skipped_when_the_route_budget_is_gone(monkeypatch):
    """The optional repair must never push the request past the route guard."""
    wrong = _explanation_payload(
        question_text="Solve log base (x+3) of (x² - x) < 1.",
        problem_expr="log_{x+3}(x^2 - x) < 1",
        solution_set="(-3, -2) U (-1, 3)",
    )
    calls = 0

    async def fake_call(prompt: str, image_bytes: bytes, json_mode: bool = False) -> str:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("repair must not run without budget")
        return json.dumps(_notes_payload(wrong))

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    monkeypatch.setattr("app.config.settings.DIAGRAM_TIMEOUT_SECONDS", 0.01)

    notes = asyncio.run(vision_service.extract_study_notes(b"image"))

    assert calls == 1
    assert notes.explanation is not None
    assert notes.explanation.steps == []
    assert notes.explanation.verified is True
    assert notes.explanation.exam_answer == "Verified solution: (-3, -2) ∪ (-1, 0) ∪ (1, 3)"


def test_failed_repair_withholds_wrong_steps_and_keeps_verified_result(monkeypatch):
    wrong = _explanation_payload(
        question_text="Solve log base (x+3) of (x² - x) < 1.",
        problem_expr="log_{x+3}(x^2 - x) < 1",
        solution_set="(-3, -2) U (-1, 3)",
        exam_answer="x ∈ (-3, -2) ∪ (-1, 3)",
    )
    calls = 0

    async def fake_call(prompt: str, image_bytes: bytes, json_mode: bool = False) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps(_notes_payload(wrong))
        raise UpstreamError(service="Gemini", detail="repair unavailable")

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    notes = asyncio.run(vision_service.extract_study_notes(b"image"))

    assert calls == 2
    assert notes.explanation is not None
    explanation = notes.explanation
    assert explanation.steps == []
    assert explanation.verified is True
    assert explanation.confidence == "review"
    assert explanation.exam_answer == "Verified solution: (-3, -2) ∪ (-1, 0) ∪ (1, 3)"
    assert "Worked steps" in explanation.verification_note


def test_model_cannot_self_verify():
    notes = StudyNotes(
        explanation=StudyExplanation(
            question_text="Question",
            question_meaning="Meaning",
            exam_answer="Answer",
            verified=True,
            verification_note="trust me",
        )
    )
    vision_service._finalize_explanation(notes, ExplanationLanguage.HINGLISH)
    assert notes.explanation is not None
    assert notes.explanation.verified is False
    assert notes.explanation.verification_note == ""


def _rich_notes(language: ExplanationLanguage = ExplanationLanguage.HINGLISH) -> StudyNotes:
    explanation = StudyExplanation(
        language=language,
        question_text="Find all x for which x² − 5x + 6 = 0.",
        question_meaning="Question ye pooch raha hai ki equation ki roots kya hain.",
        what_is_asked=["x ki values find karni hain."],
        concept=["Factorisation se roots nikalte hain."],
        steps=[
            {
                "title": "Factorise",
                "explanation": "Product 6 aur sum -5 wale numbers dhoondo.",
                "equation": "(x - 2)(x - 3) = 0",
            }
        ],
        exam_answer="x = 2 or x = 3",
        recap=["Roots 2 aur 3 hain."],
        confidence="clear",
        verified=True,
    )
    return StudyNotes(
        topic={"title": "Quadratic Equations"},
        what_you_should_remember="Roots factorisation se milte hain.",
        key_formulas=[{"formula": "x² − 5x + 6 = 0", "explanation": "quadratic"}],
        explanation=explanation,
    )


def test_markdown_puts_question_explanation_before_revision_sheet():
    markdown = render_study_notes(_rich_notes())
    order = [
        "## 📖 Original Question",
        "## 🧠 Question ka matlab",
        "## 🎯 Question humse kya maang raha hai?",
        "## 📚 Concept samjho",
        "## 🪜 Step-by-step solution",
        "## ✍️ Exam mein kaise likhna hai? (formal English answer)",
        "## ⏱️ Quick recap",
        "## 📝 Revision Sheet",
        "## 🎯 What You Should Remember",
    ]
    positions = [markdown.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "x = 2 or x = 3" in markdown
    assert "✓ Final result verified by SnapNote's exact solver." in markdown


def test_markdown_english_labels_and_withheld_note():
    notes = _rich_notes(ExplanationLanguage.ENGLISH)
    assert notes.explanation is not None
    notes.explanation.verified = False
    notes.explanation.verification_note = "The worked steps could not be verified exactly."
    markdown = render_study_notes(notes)
    assert "## 🧠 What the Question Means" in markdown
    assert "## 🎯 What the Question Asks" in markdown
    assert "## 📚 The Concept" in markdown
    assert "## ✍️ How to Write It in the Exam (formal answer)" in markdown
    assert "The worked steps could not be verified exactly." in markdown
    assert "✓ Final result verified" not in markdown


def test_markdown_without_explanation_has_no_question_sections():
    notes = StudyNotes(topic={"title": "Photosynthesis"})
    markdown = render_study_notes(notes)
    assert "Original Question" not in markdown
    assert "Revision Sheet" not in markdown


# ── Route integration ──


def test_invalid_explanation_language_rejected_before_any_charge(
    monkeypatch, sample_diagram_image
):
    device = "hinglish-invalid-language-device-000000000000"
    _seed_device(device)
    called = {"n": 0}

    async def fail_if_called(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("Gemini must not be called for an invalid language")

    monkeypatch.setattr("app.routes.extract.extract_study_notes", fail_if_called)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={
                    "context": json.dumps({}),
                    "deviceId": device,
                    "explanationLanguage": "hindi",
                },
            )

    resp = asyncio.run(run())
    assert resp.status_code == 422
    assert "hinglish" in resp.json()["error"].casefold()
    assert called["n"] == 0
    assert _balance(device) == 50


def test_diagram_route_passes_english_language_and_renders_explanation(
    monkeypatch, sample_diagram_image
):
    device = "hinglish-diagram-route-device-000000000000"
    _seed_device(device)
    captured: dict = {}

    async def fake_notes(image_bytes, explanation_language=ExplanationLanguage.HINGLISH):
        captured["language"] = explanation_language
        return StudyNotes(
            topic={"title": "Quadratics"},
            explanation=StudyExplanation(
                language=explanation_language,
                question_text="Find all x for which x² − 5x + 6 = 0.",
                question_meaning="The question asks for the roots.",
                what_is_asked=["Find x."],
                concept=["Factorise the quadratic."],
                steps=[
                    {
                        "title": "Factorise",
                        "explanation": "Find the two factors.",
                        "equation": "(x - 2)(x - 3) = 0",
                    }
                ],
                exam_answer="x = 2 or x = 3",
                recap=["The roots are 2 and 3."],
                confidence="clear",
            ),
        )

    monkeypatch.setattr("app.routes.extract.extract_study_notes", fake_notes)
    monkeypatch.setattr("app.routes.extract.upload_image", lambda img: None)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={
                    "context": json.dumps({}),
                    "deviceId": device,
                    "explanationLanguage": "english",
                },
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    body = resp.json()
    assert captured["language"] == ExplanationLanguage.ENGLISH
    assert body["creditsUsed"] == 5
    assert body["studyNotes"]["explanation"]["language"] == "english"
    assert "What the Question Means" in body["markdown"]
    assert "x = 2 or x = 3" in body["markdown"]
    assert _balance(device) == 45


def test_revision_route_returns_explanation_markdown_at_one_credit(
    monkeypatch, sample_text_image
):
    device = "hinglish-revision-route-device-000000000000"
    _seed_device(device)
    captured: dict = {}

    async def fake_revision(image_bytes, context_hint="", explanation_language=ExplanationLanguage.HINGLISH):
        captured["language"] = explanation_language
        captured["context"] = context_hint
        return StudyNotes(
            topic={"title": "Linear Equations"},
            explanation=StudyExplanation(
                language=explanation_language,
                question_text="Solve 2x + 3 = 7.",
                question_meaning="Question ye pooch raha hai ki x ki value kya hai.",
                what_is_asked=["x find karo."],
                concept=["Dono sides se 3 subtract karo."],
                steps=[
                    {
                        "title": "Subtract 3",
                        "explanation": "x ko isolate karne ke liye.",
                        "equation": "2x = 4",
                    }
                ],
                exam_answer="x = 2",
                recap=["x = 2."],
                confidence="clear",
            ),
        )

    monkeypatch.setattr("app.routes.extract.extract_revision_guide", fake_revision)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/revision",
                files={"image": ("t.png", sample_text_image, "image/png")},
                data={
                    "context": json.dumps({"topic": "Linear Equations"}),
                    "deviceId": device,
                },
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    body = resp.json()
    assert captured["language"] == ExplanationLanguage.HINGLISH
    assert body["creditsUsed"] == 1
    assert "Question ka matlab" in body["markdown"]
    assert "x = 2" in body["markdown"]
    assert _balance(device) == 49


def test_frontend_hinglish_is_default_and_sent_with_both_generating_flows():
    html_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "website", "index.html")
    )
    with open(html_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    assert 'data-lang="hinglish" aria-pressed="true"' in html
    assert 'data-lang="english" aria-pressed="false"' in html
    assert 'let currentExplanationLanguage' in html
    assert 'localStorage.getItem("snapnote-explanation-lang") === "english" ? "english" : "hinglish"' in html
    assert 'formData.append("explanationLanguage", currentExplanationLanguage)' in html
    assert html.count('formData.append("explanationLanguage", currentExplanationLanguage)') == 2
    assert "function renderExplanation" in html
    assert "Original Question" in html
    assert "exam-answer-card" in html
    assert "q-source-note" in html
    assert "if (result.markdown) {" in html
    assert "Simple Hindi" not in html
