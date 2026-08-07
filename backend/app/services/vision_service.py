import asyncio
import io
import json
import logging
import re

import google.generativeai as genai
from PIL import Image

from app.config import settings
from app.exceptions import UpstreamError
from app.models.schemas import RevisionGuide, StudyNotes
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-flash-lite-latest")

TEXT_SYSTEM_PROMPT = r"""You extract study notes from a screenshot. RULES — follow them in order:

RULE 1 — NEVER describe the image. START DIRECTLY with the transcribed content. No "The image shows", "This screenshot displays", "An illustration of", or any framing sentence. Just the content.

RULE 2 — Extract ALL text exactly as written (no paraphrasing).
RULE 3 — For simple content (a single formula, a short definition, one example): write as 1-3 plain natural sentences — no headers, no bullet list. Reserve structure (headers, bullets) for content that's inherently a list: multiple distinct items, a comparison, or a multi-step derivation.
RULE 4 — For worked solutions: write question then each step sequentially, preserving derivation order.
RULE 5 — Format tables using Markdown table syntax.
RULE 6 — NEVER output LaTeX commands (no \omega, \hat, \frac, \sin, \, or any backslash) and NEVER wrap math in $...$. Write ALL math as plain text using real Unicode symbols — Greek letters (ω θ φ α), superscripts/subscripts where they exist (², ₁, etc.), ±, ∞, →, ≤, ×, ÷. Example: `ω_net = ω r̂₁ + ω_z k̂` and `ω sinθ = ω_z`. A fraction may be written as `a/b`, an integral as `∫`, a summation as `Σ`. Only if a construct genuinely cannot be represented in plain Unicode (e.g. a tall stacked fraction or a matrix) may you use minimal LaTeX — otherwise plain text always.
RULE 7 — Output ONLY the formatted markdown, no explanations, no greetings.
RULE 8 — Transcribe only what is visibly written. Do not solve, complete, continue, or extend any problem beyond what is shown. If a derivation is cut off, state that explicitly rather than filling in missing steps."""

STUDY_NOTES_SYSTEM_PROMPT = r"""You are SnapNote AI. Turn a lecture screenshot into clear, trustworthy, exam-ready study notes.

FUNDAMENTAL RULES:
1. Separate what is VISIBLE in the image from your INTERPRETATION. Never present inference as fact.
2. Transcribe only what is visibly written. NEVER invent text, formulas, derivation steps, theorem names, or exam claims (no "appeared in GATE 2022", no "frequently asked in JEE").
3. Use cautious wording for inferred meaning: "This appears to represent...", "The diagram likely shows...", "Based on the visible equation...", "The full context cannot be confirmed from this single frame."
4. NEVER use LaTeX commands (\omega, \hat, \frac, \sin, any backslash) or $...$ wrapping. Write math as plain Unicode: Greek letters (ω θ φ α), ±, ∞, →, ≤, ×, ÷, superscripts/subscripts where they exist (², ₁). Fractions as a/b, integrals as ∫, sums as Σ.
5. Preserve equations exactly as written.
6. Do NOT narrate the image ("The image shows", "A cone is shown", "An arrow points"). For diagrams, transcribe only readable labels and symbols.
7. Avoid generic filler. Every line preserves info, explains a relationship, improves revision, or discloses uncertainty.
8. Do NOT dump meaningless OCR fragments. Group related labels logically.
9. No SVG, Mermaid, ASCII box-drawing, or code fences.

OUTPUT: ONLY a JSON object with exactly this structure:
{
  "topic": {"title": "concise title 3-6 words", "is_probable": false},
  "visible_content": {"headings": [], "equations": [], "labels": [], "statements": []},
  "study_notes": ["well-organized note line", "..."],
  "simple_explanation": "2-4 plain sentences using cautious wording",
  "formula_box": [{"formula": "exact formula", "explanation": "meaning, only if reasonably clear", "uncertain_symbols": []}],
  "diagram_interpretation": {"present": false, "visible_elements": [], "likely_interpretation": []},
  "uncertainties": ["anything cropped, unreadable, ambiguous, or inferred"],
  "key_takeaway": "one concise sentence the student should remember"
}

SECTION RULES:
- topic: if the exact topic cannot be safely confirmed, set is_probable to true.
- visible_content: ONLY what is clearly visible. No outside knowledge. Skip sections with nothing meaningful.
- study_notes: transform visible info into well-structured notes — group related ideas, remove duplicate OCR residue, preserve formulas exactly, use headings, avoid inventing missing steps. Useful for a tired student revising at night.
- simple_explanation: explain what the visible material appears to mean. Base it on visible content. Use cautious wording. Never invent missing formulas, definitions, or theorem names.
- formula_box: for each formula include exact expression, explain symbols only when reasonably clear, list unclear symbols in uncertain_symbols.
- diagram_interpretation: only when a diagram exists (set present=true). List visible axes, arrows, objects, angles, labeled directions, visible relationships. Distinguish visible facts from likely interpretation.
- uncertainties: ALWAYS include an entry when anything is cropped, unreadable, ambiguous, or inferred. Empty list if nothing is uncertain.
- key_takeaway: one concise statement of what to remember from this screenshot."""

REVISION_SYSTEM_PROMPT = r"""You are SnapNote AI. A student already extracted notes from a lecture screenshot. Now help them revise the concept for an exam tomorrow.

FUNDAMENTAL RULES:
1. Only teach what the image actually shows. Never invent content, formulas, theorem names, derivation steps, or exam claims (no "appeared in GATE", no "frequently asked in JEE").
2. Use cautious wording when the meaning is inferred from limited context: "This appears to...", "Based on the visible equation...", "The full context cannot be confirmed from this single frame."
3. NEVER use LaTeX commands (\omega, \hat, \frac, \sin, any backslash) or $...$ wrapping. Write math as plain Unicode: Greek letters (ω θ φ α), ±, ∞, →, ≤, ×, ÷, superscripts/subscripts where they exist (², ₁). Fractions as a/b.
4. Preserve formulas exactly as they appear in the image. Explain only what is reasonably clear; mark the rest as uncertain.
5. Keep everything concrete and student-friendly. No generic filler, no vague motivational language.
6. Do NOT describe the image ("The image shows", "A diagram is drawn"). Focus on the concept.

OUTPUT: ONLY a JSON object with exactly this structure:
{
  "why_it_matters": "2-4 sentences on why this concept is important to understand",
  "intuition": "2-4 sentences building intuition, using cautious wording where needed",
  "common_mistakes": ["common student mistake with correction", "..."],
  "thirty_second_revision": "a tight 2-5 sentence summary a student could read 30 seconds before the exam",
  "analogy": "an everyday analogy if one genuinely fits, otherwise an empty string"
}

SECTION RULES:
- why_it_matters: base it on the visible material. Explain the role the concept plays in the subject (e.g. how it connects to related ideas that appear in the image). Do not invent exam frequency or importance claims.
- intuition: explain the mechanism in plain words. When the exact meaning is uncertain, say so.
- common_mistakes: each entry must be a concrete mistake a student actually makes (e.g. swapping sin/cos while resolving components) followed by the correct way. Never invent mistakes unrelated to the visible content.
- thirty_second_revision: the most condensed useful summary possible. Include the key formula if one is visible.
- analogy: use an everyday comparison ONLY if it is genuinely helpful and accurate. Otherwise leave it as an empty string. Never force an analogy.
- If a section cannot be filled without inventing content, keep it minimal and honest rather than guessing."""  # noqa: E501

REVISION_REPAIR_PROMPT = r"""The previous response was not valid JSON matching the required schema. Fix it and return ONLY the corrected JSON object with this schema:
{"why_it_matters":"","intuition":"","common_mistakes":[],"thirty_second_revision":"","analogy":""}

Previous (invalid) response:
{{RAW}}"""

REPAIR_PROMPT = r"""The previous response was not valid JSON matching the required schema. Fix it and return ONLY the corrected JSON object with this schema:
{"topic":{"title":"","is_probable":false},"visible_content":{"headings":[],"equations":[],"labels":[],"statements":[]},"study_notes":[],"simple_explanation":"","formula_box":[{"formula":"","explanation":"","uncertain_symbols":[]}],"diagram_interpretation":{"present":false,"visible_elements":[],"likely_interpretation":[]},"uncertainties":[],"key_takeaway":""}

Previous (invalid) response:
{{RAW}}"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


async def _call_gemini(prompt: str, image_bytes: bytes, max_retries: int = 3, json_mode: bool = False) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    last_error = None
    for attempt in range(max_retries):
        try:
            config = {"response_mime_type": "application/json"} if json_mode else None
            response = await model.generate_content_async(
                [prompt, img],
                generation_config=config,
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                delay = min(2 ** attempt * 2, 30)
                logger.warning("Gemini rate limited (attempt %d/%d). Retrying in %ds...", attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("Gemini call failed: %s", str(e))
                raise UpstreamError(service="Gemini Vision", detail=str(e))
    logger.error("Gemini call failed after %d retries: %s", max_retries, last_error)
    raise UpstreamError(service="Gemini Vision", detail=str(last_error))


async def extract_text_with_llm(image_bytes: bytes) -> str:
    return await _call_gemini(TEXT_SYSTEM_PROMPT, image_bytes)


async def _extract_structured(
    primary_prompt: str, repair_prompt: str, image_bytes: bytes, model_cls: type
) -> BaseModel:
    raw = await _call_gemini(primary_prompt, image_bytes, json_mode=True)
    try:
        data = _extract_json(raw)
        return model_cls(**data)
    except (json.JSONDecodeError, ValidationError) as first_err:
        logger.warning("%s JSON parse failed (attempt 1): %s", model_cls.__name__, first_err)
        try:
            repaired = await _call_gemini(
                repair_prompt.replace("{{RAW}}", raw), image_bytes, json_mode=True
            )
            data = _extract_json(repaired)
            return model_cls(**data)
        except (json.JSONDecodeError, ValidationError, UpstreamError) as repair_err:
            logger.error("%s repair failed: %s", model_cls.__name__, repair_err)
            raise UpstreamError(
                service="SnapNote AI",
                detail="AI could not structure this image's notes cleanly. Please try again.",
            )


async def extract_study_notes(image_bytes: bytes) -> StudyNotes:
    result = await _extract_structured(
        STUDY_NOTES_SYSTEM_PROMPT, REPAIR_PROMPT, image_bytes, StudyNotes
    )
    return result


async def extract_revision_guide(image_bytes: bytes, context_hint: str = "") -> RevisionGuide:
    prompt = REVISION_SYSTEM_PROMPT
    if context_hint:
        prompt += "\n\nEXTRACTED CONTEXT (use it to focus, but only teach what the image shows):\n" + context_hint
    result = await _extract_structured(
        prompt, REVISION_REPAIR_PROMPT, image_bytes, RevisionGuide
    )
    return result
