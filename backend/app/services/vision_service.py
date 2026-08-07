import asyncio
import io
import json
import logging
import re

import google.generativeai as genai
from PIL import Image

from app.config import settings
from app.exceptions import UpstreamError
from app.models.schemas import StudyNotes
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

STUDY_NOTES_SYSTEM_PROMPT = r"""You are SnapNote AI. Turn a messy lecture screenshot into exam-ready study material. The student wants to understand and revise what the professor was teaching — NOT a description of what objects the image contains.

FUNDAMENTAL RULES:
1. GROUNDING: Never invent missing lecture content. Clearly distinguish in your mind (and state when relevant):
   - Visible evidence (clearly readable equations, labels, text)
   - Safe inference (what the visible material reasonably implies)
   - Missing context (anything cropped, cut off, or not shown)
   If a derivation or equation is cut off, say so. Do NOT complete the professor's missing steps and present them as the professor's work.
2. Never invent text, formulas, derivation steps, theorem names, or exam claims (no "appeared in GATE 2022", no "frequently asked in JEE").
3. Use cautious wording for inferred meaning: "This appears to...", "Based on the visible equation...", "The full context cannot be confirmed from this single frame."
4. NEVER use LaTeX commands (\omega, \hat, \frac, \sin, any backslash) or $...$ wrapping. Write math as plain Unicode: Greek letters (ω θ φ α), ±, ∞, →, ≤, ×, ÷, superscripts/subscripts where they exist (², ₁). Fractions as a/b, integrals as ∫, sums as Σ.
5. Preserve equations exactly as written.
6. Do NOT narrate or inventory the image. No lists of detected axes, objects, arrows, or labels. The image is the professor's teaching medium, not the subject.
7. Avoid generic filler. Every line preserves info, explains a relationship, improves revision, or discloses uncertainty.
8. No SVG, Mermaid, ASCII box-drawing, or code fences.

OUTPUT: ONLY a JSON object with exactly this structure:
{
  "topic": {"title": "study-note title 2-6 words", "is_probable": false},
  "what_you_should_remember": "one concise, exam-relevant takeaway the student should remember",
  "key_formulas": [{"formula": "exact formula", "explanation": "what each symbol means + what relationship it represents", "uncertain_symbols": [], "confidence": "clear"}],
  "understand_it": ["plain-English explanation of what this concept is teaching", "intuition built from visible evidence or safe inference"],
  "common_mistakes": ["mistake students commonly make with this concept, with the correct way"],
  "thirty_second_revision": ["3-5 short bullets a student could scan 30 seconds before the exam"],
  "visual_context": {"present": false, "summary": "1-2 sentences on how the diagram relates to the concept, only if a diagram exists and it helps understanding"},
  "verify_before_studying": ["specific equation or symbol that may have been misread, with what was ambiguous"],
  "uncertainties": ["anything cropped, unreadable, ambiguous, or missing"],
  "analogy": "an everyday analogy if one genuinely fits, otherwise an empty string"
}

SECTION RULES:
- topic: a normal study-note title. If the exact topic cannot be safely confirmed, set is_probable to true (rendered subtly as "Topic inferred from screenshot").
- what_you_should_remember: THE core payoff. One concise, exam-relevant sentence answering "what am I supposed to remember for my exam?"
- key_formulas: for each formula include the exact expression, explain each symbol, and explain the relationship it represents. Set confidence honestly:
  * "clear" — formula is visually legible and internally consistent
  * "context_needed" — formula is readable but the surrounding lecture context is missing
  * "possible_extraction_issue" — the actual mathematical symbols are ambiguous/unclear
  Only when confidence is "possible_extraction_issue" should the formula also be listed in verify_before_studying.
- understand_it: answer "What is this concept actually teaching me?" Prioritize intuition and understanding. Base it on visible content + safe inference. Never invent missing formulas, definitions, or theorem names. If the derivation is cut off, state that.
- common_mistakes: do NOT fabricate mistakes. Only include a mistake when it is genuinely supported by the visible material, or clearly frame it as "a general thing to watch for with this type of problem." Never pretend a mistake was taught by the professor unless it is visible.
- thirty_second_revision: 3-5 tight bullets. Include the key formula if one is visible.
- visual_context: 1-2 sentences maximum, ONLY if a diagram exists AND explaining it helps understanding. Never list detected objects, axes, arrows, or labels.
- verify_before_studying: ONLY genuinely uncertain equations/symbols (confidence "possible_extraction_issue"). Empty unless truly needed.
- uncertainties: only genuinely ambiguous/missing material. Empty if nothing is uncertain.
- analogy: use an everyday comparison ONLY if it is genuinely helpful and accurate. Otherwise empty string. Never force one.

CRITICAL READING RULES:
- When a handwritten or blurry symbol could be more than one thing, do NOT silently pick one. Record your best guess, set confidence to "possible_extraction_issue", and list it in verify_before_studying.
- Common confusion pairs to watch: τ (tau) vs t, ω (omega) vs w, θ (theta) vs 0/O, μ (mu) vs u, α (alpha) vs a, v vs r.
- If a derivation or equation is partially cut off, transcribe only the visible part and flag the rest as missing context rather than completing it."""

REVISION_SYSTEM_PROMPT = r"""You are SnapNote AI. A student extracted a cheap text snapshot from a lecture screenshot. Now turn it into exam-ready study material so they can understand and revise the concept.

Use the same output schema, grounding, and safety rules as the main study-notes prompt (STUDY_NOTES_SYSTEM_PROMPT). The only difference: you may include an "analogy" when one genuinely fits, because this is the revision step.

GROUNDING RULES:
1. Never invent missing lecture content. Distinguish visible evidence from safe inference from missing context. If a derivation is cut off, say so — do not complete it as if the professor wrote it.
2. Never invent formulas, theorem names, derivation steps, or exam claims.
3. common_mistakes must NOT be fabricated. Only include a mistake when supported by the visible material or clearly framed as "a general thing to watch for with this type of problem."
4. Keep uncertainty proportional. Set each formula's confidence honestly ("clear" | "context_needed" | "possible_extraction_issue"). Only list in verify_before_studying when confidence is "possible_extraction_issue".
5. Use cautious wording for inferred meaning: "This appears to...", "Based on the visible equation...".

OUTPUT: ONLY a JSON object with exactly this structure:
{
  "topic": {"title": "study-note title 2-6 words", "is_probable": false},
  "what_you_should_remember": "one concise, exam-relevant takeaway",
  "key_formulas": [{"formula": "exact formula", "explanation": "symbols + relationship", "uncertain_symbols": [], "confidence": "clear"}],
  "understand_it": ["plain-English explanation", "intuition"],
  "common_mistakes": ["mistake with the correct way, only when genuinely useful"],
  "thirty_second_revision": ["3-5 short bullets"],
  "visual_context": {"present": false, "summary": "1-2 sentences, only if a diagram exists and helps"},
  "verify_before_studying": ["equation or symbol that may have been misread"],
  "uncertainties": ["anything ambiguous or missing"],
  "analogy": "an everyday analogy if one genuinely fits, otherwise an empty string"
}

SECTION RULES:
- what_you_should_remember: THE core payoff. Answer "what am I supposed to remember for my exam?"
- common_mistakes: do NOT fabricate. Frame as "a general thing to watch for" unless the mistake is visibly taught.
- thirty_second_revision: 3-5 tight bullets. Include the key formula if visible.
- visual_context: never list detected objects/axes/labels. 1-2 sentences max, only if helpful.
- analogy: only if genuinely helpful and accurate, otherwise empty string."""  # noqa: E501

REVISION_REPAIR_PROMPT = r"""The previous response was not valid JSON matching the required schema. Fix it and return ONLY the corrected JSON object with this schema:
{"topic":{"title":"","is_probable":false},"what_you_should_remember":"","key_formulas":[{"formula":"","explanation":"","uncertain_symbols":[],"confidence":"clear"}],"understand_it":[],"common_mistakes":[],"thirty_second_revision":[],"visual_context":{"present":false,"summary":""},"verify_before_studying":[],"uncertainties":[],"analogy":""}

Previous (invalid) response:
{{RAW}}"""

REPAIR_PROMPT = r"""The previous response was not valid JSON matching the required schema. Fix it and return ONLY the corrected JSON object with this schema:
{"topic":{"title":"","is_probable":false},"what_you_should_remember":"","key_formulas":[{"formula":"","explanation":"","uncertain_symbols":[],"confidence":"clear"}],"understand_it":[],"common_mistakes":[],"thirty_second_revision":[],"visual_context":{"present":false,"summary":""},"verify_before_studying":[],"uncertainties":[],"analogy":""}

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


async def extract_revision_guide(image_bytes: bytes, context_hint: str = "") -> StudyNotes:
    prompt = REVISION_SYSTEM_PROMPT
    if context_hint:
        prompt += "\n\nEXTRACTED CONTEXT (use it to focus, but only teach what the image shows):\n" + context_hint
    result = await _extract_structured(
        prompt, REVISION_REPAIR_PROMPT, image_bytes, StudyNotes
    )
    return result
