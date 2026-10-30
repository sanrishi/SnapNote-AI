import asyncio
import io
import json
import logging
import re

import google.generativeai as genai
from PIL import Image

from app.config import settings
from app.exceptions import UpstreamError
from app.models.schemas import (
    DiagramRep,
    DiagramSpec,
    StudyNotes,
    VisualComparison,
    VisualSpec,
)
from app.utils.diagram_validation import SUPPORTED_DIAGRAM_TYPES, validate_diagram_spec
from app.utils.diagram_renderer import render_polar_region
from app.utils.latex_clean import latex_to_unicode
from app.utils.svg_safe import sanitize_svg
from app.utils.table_quality import normalize_visual_spec
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-flash-lite-latest")

_fallback_model = None


def _get_fallback_model():
    """Lazily construct the capacity-fallback model (once per process)."""
    global _fallback_model
    if _fallback_model is None and settings.GEMINI_FALLBACK_MODEL.strip():
        _fallback_model = genai.GenerativeModel(settings.GEMINI_FALLBACK_MODEL.strip())
    return _fallback_model


def _is_capacity_error(err_str: str) -> bool:
    """503/504-style errors mean the serving pool is sick, not the request.

    A different model usually serves from different capacity, so exactly one
    fallback attempt is justified. 429/quota and auth errors are excluded:
    retrying those elsewhere burns budget without helping.
    """
    return any(
        token in err_str
        for token in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "overloaded", "high demand")
    )


_DAILY_QUOTA_RETRY_IN_RE = re.compile(r"retry in \d+h", re.IGNORECASE)
_QUOTA_RETRY_DELAY_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)")


def _is_daily_quota_error(err_str: str) -> bool:
    """True when a 429 is a per-DAY quota bucket, not a per-minute burst.

    The daily reset is hours away, so the short 429 retry loop cannot help —
    it only delays the honest "high demand" mapping at the route. Per-minute
    429s (small retry_delay) keep their retry.
    """
    if "PerDay" in err_str:
        return True
    if _DAILY_QUOTA_RETRY_IN_RE.search(err_str):
        return True
    m = _QUOTA_RETRY_DELAY_RE.search(err_str)
    return bool(m and int(m.group(1)) > 300)

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
4. NEVER use LaTeX commands (\omega, \hat, \frac, \sin, any backslash) or $...$ wrapping. Write math as plain Unicode: Greek letters (ω θ φ α), ±, ∞, →, ≤, ×, ÷, superscripts/subscripts where they exist (², ₁). Fractions as a/b, integrals as ∫, sums as Σ. Every formula string MUST contain zero backslash (\) characters.
5. Preserve equations exactly as written.
6. Do NOT narrate or inventory the image. No lists of detected axes, objects, arrows, or labels. The image is the professor's teaching medium, not the subject.
7. Avoid generic filler. Every line preserves info, explains a relationship, improves revision, or discloses uncertainty.

OUTPUT: ONLY a JSON object with exactly this structure:
{
  "topic": {"title": "study-note title 2-6 words", "is_probable": false},
  "what_you_should_remember": "one concise, exam-relevant takeaway the student should remember",
  "key_formulas": [{"formula": "exact formula", "explanation": "what each symbol means + what relationship it represents", "uncertain_symbols": [], "confidence": "clear"}],
  "understand_it": ["plain-English explanation of what this concept is teaching", "intuition built from visible evidence or safe inference"],
  "common_mistakes": ["mistake students commonly make with this concept, with the correct way"],
  "thirty_second_revision": ["3-5 short bullets a student could scan 30 seconds before the exam"],
  "visual_context": {"present": false, "summary": "1-2 sentences teaching what the diagram MEANS conceptually for the concept, only if a diagram exists and it helps understanding"},
  "diagram": {"present": false, "svg": ""},
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
- visual_context: 1-2 sentences maximum, ONLY if a diagram exists AND explaining it helps understanding. Explain what the diagram MEANS conceptually and why it matters for the concept — never list what objects, axes, arrows, or labels are visible. It must teach the relationship, not describe the picture. Example (GOOD): "The diagram represents a closed-loop control system: the reference input is compared with feedback to form an error signal, which the controller uses to drive the plant toward the desired output." Example (BAD): "Block diagrams illustrate closed-loop control systems with reference inputs, summing junctions, controllers, processes."
- diagram: REBUILD the visible diagram as a CLEAN vector SVG (present=true + svg) whenever the screenshot contains a meaningful diagram, flowchart, block diagram, graph, geometric figure, or schematic. This gives the student a readable, non-messy version of what was on the board. Rules:
  * COMPLETENESS: If the source contains MULTIPLE distinct diagram sketches (e.g. two separate circles/figures, a graph AND a schematic), include ALL of them in the SVG — never simplify down to a single figure and never drop any visible figure. Every distinct diagram in the screenshot must appear in the output.
  * Ground the SVG strictly in what is visible. Recreate the same boxes, arrows, shapes, connections, and labels — do not add, remove, or "improve" the structure, and never invent content that is not in the image.
  * FILL SAFETY: Open shapes (circles, ellipses, rectangles, polygons that represent outlines/regions) MUST use fill="none" with a visible stroke (e.g. stroke="#000" stroke-width="2" for light backgrounds, or stroke="#e5e5e5" for dark backgrounds). NEVER fill a whole circle or outline shape solid — a solid fill blots out everything inside. Only use fill when shading a genuinely meaningful area (like the region between two boundary curves), and use a low-opacity fill (opacity="0.15"–"0.35") so text and lines beneath stay readable. Labels must always be rendered as <text> elements with readable contrast — never obscured by a filled shape.
  * Readable and clean: straight lines, generous spacing, labels as <text> elements, no hand-drawn wobble, no background photo, no watermark.
  * Keep it simple and flat — rectangles, circles, lines, arrows (markers), text. A student must be able to read it at a glance.
  * Escape text properly (&lt; &gt; &amp;). Use viewBox with a sane aspect ratio and width 100%.
  * Only include <svg>…</svg>. No markdown fences, no comments.
  * If the screenshot is pure prose/equations with no real diagram (no shapes/connections/structure), set present=false and svg="".
- verify_before_studying: ONLY genuinely uncertain equations/symbols (confidence "possible_extraction_issue"). Empty unless truly needed.
- uncertainties: only genuinely ambiguous/missing material. Empty if nothing is uncertain.
- analogy: ALWAYS provide a concise 2-4 sentence everyday analogy that builds intuition for the concept (e.g., for torque: opening a door far from the hinge; for integrals: accumulating area like filling a shape). Only leave empty if the content is purely definitional and no analogy would help. Never invent unsupported facts.

CRITICAL READING RULES:
- When a handwritten or blurry symbol could be more than one thing, do NOT silently pick one. Record your best guess, set confidence to "possible_extraction_issue", and list it in verify_before_studying.
- Common confusion pairs to watch: τ (tau) vs t, ω (omega) vs w, θ (theta) vs 0/O, μ (mu) vs u, α (alpha) vs a, v vs r.
- If a derivation or equation is partially cut off, transcribe only the visible part and flag the rest as missing context rather than completing it."""

SEMANTIC_DIAGRAM_RULE = """- diagram_spec: REBUILD the visible diagram as a STRUCTURED SEMANTIC SPEC, NOT as SVG. The student needs the region's exact boundaries so our renderer can draw a clean diagram. You output MATH (radii and angles), never pixels. Never output SVG geometry — no <circle>, no cx/cy, no pixel coordinates.
  * Set diagram_spec.present=true ONLY when the screenshot contains a real polar-coordinate integration region: two concentric circular boundaries, the area between them shaded, radial labels and/or a theta arc. Otherwise present=false and leave bounds empty.
  * diagram_spec.diagram_type: "polar_region" when the figure is a polar-coordinate region. If the visible diagram is a different kind of figure, still return the spec with present=true but set diagram_type to that figure's honest type name (only "polar_region" is rendered today; anything else is reported as unsupported rather than guessed).
  * bounds.inner / bounds.outer: the two RADII exactly as written, as plain math strings — e.g. "1", "sqrt(5)", "3". NEVER pixel values. Use "" if a radius is not clearly readable. Never invent radii that are not visible.
  * bounds.theta_min / bounds.theta_max: the angular sweep as plain math strings — "0", "pi", "2*pi", "pi/2". If the shaded region is a COMPLETE ring (the shading goes all the way around with no gap) use theta_min="0" and theta_max="2*pi" — a closed ring is visible evidence of a full revolution. Leave a field empty only if the region is cut off or the sweep is genuinely ambiguous.
  * labels: only text visibly written in the figure (e.g. "r = 1", "r = sqrt(5)", "θ"). Empty if none is readable.
  * show_axes: true unless the figure clearly has no axes.
  * shade_region: true when the area between the two boundaries is shaded.
  * instruction_text: 1 short phrase teaching what region is being integrated, grounded in what is visible (e.g. "region between r = 1 and r = sqrt(5)"). Empty if unclear.
  * uncertain: list anything you could not read with high confidence (e.g. "outer radius could be sqrt(5) or sqrt(3)"). Empty if confident.
  * If a boundary is cut off, leave its field empty and note the missing part in uncertain — never complete it.
  * Leave the top-level diagram field (present + svg) as present=false, svg="" — our renderer draws the SVG from diagram_spec."""

_SEMANTIC_OUTPUT_LINE = '  "diagram": {"present": false, "svg": ""},'
_SEMANTIC_REPLACEMENT = (
    '  "diagram": {"present": false, "svg": ""},\n'
    '  "diagram_spec": {"present": false, "diagram_type": "polar_region", '
    '"bounds": {"inner": "", "outer": "", "theta_min": "", "theta_max": ""}, '
    '"show_axes": true, "labels": [], "shade_region": true, "instruction_text": [], "uncertain": []},'
)
_LEGACY_DIAGRAM_BULLET = "- diagram: REBUILD the visible diagram as a CLEAN vector SVG"
_VERIFY_ANCHOR = "\n- verify_before_studying:"


def _build_semantic_prompt() -> str:
    """Derive the semantic-mode prompt from the base prompt (fail fast if anchors drift)."""
    base = STUDY_NOTES_SYSTEM_PROMPT
    if (
        _SEMANTIC_OUTPUT_LINE not in base
        or _LEGACY_DIAGRAM_BULLET not in base
        or _VERIFY_ANCHOR not in base
    ):
        raise RuntimeError("STUDY_NOTES_SYSTEM_PROMPT anchors changed; update _build_semantic_prompt")
    head, tail = base.split(_VERIFY_ANCHOR, 1)
    head, _legacy = head.split(_LEGACY_DIAGRAM_BULLET, 1)
    base = head.rstrip() + f"\n{SEMANTIC_DIAGRAM_RULE}" + _VERIFY_ANCHOR + tail
    base = base.replace(_SEMANTIC_OUTPUT_LINE, _SEMANTIC_REPLACEMENT, 1)
    return base


STUDY_NOTES_SEMANTIC_PROMPT = _build_semantic_prompt()

REVISION_SYSTEM_PROMPT = r"""You are SnapNote AI. A student extracted a cheap text snapshot from a lecture screenshot. Now turn it into exam-ready study material so they can understand and revise the concept.

Use the same output schema, grounding, and safety rules as the main study-notes prompt (STUDY_NOTES_SYSTEM_PROMPT). Always include a concise analogy when it helps intuition (most physics/math concepts benefit).

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
  "visual_context": {"present": false, "summary": "1-2 sentences teaching what the diagram MEANS conceptually, only if a diagram exists and helps"},
  "diagram": {"present": false, "svg": ""},
  "verify_before_studying": ["equation or symbol that may have been misread"],
  "uncertainties": ["anything ambiguous or missing"],
  "analogy": "an everyday analogy if one genuinely fits, otherwise an empty string"
}

SECTION RULES:
- what_you_should_remember: THE core payoff. Answer "what am I supposed to remember for my exam?"
- common_mistakes: do NOT fabricate. Frame as "a general thing to watch for" unless the mistake is visibly taught.
- thirty_second_revision: 3-5 tight bullets. Include the key formula if visible.
- visual_context: 1-2 sentences max, only if a diagram exists and helps. Explain what the diagram MEANS conceptually (teach the relationship), never list what objects/axes/labels are visible.
- diagram: same rules as the main prompt — rebuild the visible diagram as a clean, readable vector SVG (present=true + svg) when the material contains one; else present=false, svg="".
- analogy: ALWAYS provide a concise 2-4 sentence analogy when it helps intuition (most concepts); only leave empty if no analogy would help."""  # noqa: E501

REVISION_REPAIR_PROMPT = r"""The previous response was not valid JSON matching the required schema. Fix it and return ONLY the corrected JSON object with this schema:
{"topic":{"title":"","is_probable":false},"what_you_should_remember":"","key_formulas":[{"formula":"","explanation":"","uncertain_symbols":[],"confidence":"clear"}],"understand_it":[],"common_mistakes":[],"thirty_second_revision":[],"visual_context":{"present":false,"summary":""},"diagram":{"present":false,"svg":""},"verify_before_studying":[],"uncertainties":[],"analogy":""}

Previous (invalid) response:
{{RAW}}"""

REPAIR_PROMPT = r"""The previous response was not valid JSON matching the required schema. Fix it and return ONLY the corrected JSON object with this schema:
{"topic":{"title":"","is_probable":false},"what_you_should_remember":"","key_formulas":[{"formula":"","explanation":"","uncertain_symbols":[],"confidence":"clear"}],"understand_it":[],"common_mistakes":[],"thirty_second_revision":[],"visual_context":{"present":false,"summary":""},"diagram":{"present":false,"svg":""},"verify_before_studying":[],"uncertainties":[],"analogy":""}

Previous (invalid) response:
{{RAW}}"""

_REPAIR_ANCHOR = '{"present":false,"svg":""},"verify_before_studying"'
_REPAIR_SEMANTIC = (
    '{"present":false,"svg":""},'
    '"diagram_spec":{"present":false,"diagram_type":"polar_region",'
    '"bounds":{"inner":"","outer":"","theta_min":"","theta_max":""},'
    '"show_axes":true,"labels":[],"shade_region":true,"instruction_text":[],"uncertain":[]},'
    '"verify_before_studying"'
)

REPAIR_PROMPT_SEMANTIC = REPAIR_PROMPT.replace(_REPAIR_ANCHOR, _REPAIR_SEMANTIC, 1)

LEGACY_DIAGRAM_ONLY_PROMPT = r"""You are SnapNote AI. A student uploaded a lecture screenshot. Our deterministic renderer does not support this diagram type yet, so draw the visible diagram as a best-effort vector SVG — the student still deserves a readable version of what was on the board.

Output ONLY a JSON object with exactly this structure:
{"present": false, "svg": ""}

RULES:
- COMPLETENESS: If the source contains MULTIPLE distinct diagram sketches, include ALL of them — never drop a visible figure.
- Ground strictly in what is visible. Recreate the same boxes, arrows, shapes, connections, and labels — do not add, remove, or "improve" structure, and never invent content not in the image.
- FILL SAFETY: Open shapes (circles, ellipses, rectangles, polygons that are outlines/regions) MUST use fill="none" with a visible stroke. NEVER fill a whole outline shape solid. Only use fill when shading a genuinely meaningful area, at low opacity (0.15–0.35) so text beneath stays readable. Labels must be <text> elements with readable contrast.
- Readable and clean: straight lines, generous spacing, labels as <text>, no hand-drawn wobble, no photo, no watermark.
- Keep it simple and flat — rectangles, circles, lines, arrows, text. A student must read it at a glance.
- Escape text (&lt; &gt; &amp;). Use viewBox with a sane aspect ratio and width 100%.
- Only include <svg>…</svg>. No markdown fences, no comments.
- If there is no real diagram (pure prose/equations, no shapes/connections/structure), set present=false and svg="".
- This is a BEST-EFFORT output: if the image is ambiguous, prefer a simpler faithful sketch over a confident wrong one."""

# Hard budget for the unsupported-diagram fallback call. The diagram route allows
# 75s total and the primary call may take up to 60s, so the fallback is capped at
# 25s to stay inside the window; on any failure we degrade to explanation-only.
LEGACY_FALLBACK_TIMEOUT_SECONDS = 25.0
_FALLBACK_NOTE = (
    "This diagram is a best-effort reconstruction. This diagram type isn't fully "
    "supported by the deterministic renderer yet, so it was redrawn as-is from the "
    "screenshot and may not be perfectly accurate — verify it against the original image."
)


async def _legacy_diagram_fallback(image_bytes: bytes) -> DiagramRep:
    """Best-effort legacy SVG for an unsupported diagram type, or an empty state on failure."""
    try:
        raw = await _call_gemini(
            LEGACY_DIAGRAM_ONLY_PROMPT,
            image_bytes,
            json_mode=True,
            timeout_seconds=LEGACY_FALLBACK_TIMEOUT_SECONDS,
        )
        data = _clean_latex_in_dict(_extract_json(raw))
        data = _sanitize_svg_in_dict(data)
        present = bool(data.get("present"))
        svg = str(data.get("svg") or "")
        if present and svg:
            return DiagramRep(present=True, svg=sanitize_svg(svg), best_effort=True)
    except Exception as e:
        logger.warning("Legacy diagram fallback failed (will degrade to explanation-only): %s", e)
    return DiagramRep(present=False, svg="")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


async def _call_gemini(
    prompt: str,
    image_bytes: bytes,
    max_retries: int = 1,
    json_mode: bool = False,
    timeout_seconds: float | None = None,
) -> str:
    """Call Gemini with a bounded per-call timeout and at most one controlled retry.

    The SDK default timeout for generate_content is 600s and it internally retries
    on 503 — together with any outer retry loop that stacks unboundedly. We pin a
    sane per-call timeout and retry only on 429/RESOURCE_EXHAUSTED (where a short
    wait genuinely helps). On 503/504 capacity errors we try the configured
    fallback model exactly once (different capacity pool), then fail fast.
    Everything else fails fast.

    `timeout_seconds` overrides the configured default so a secondary fallback call
    can be hard-budgeted (keeps the total under the diagram route timeout).
    """
    img = Image.open(io.BytesIO(image_bytes))
    call_timeout = timeout_seconds if timeout_seconds is not None else settings.GEMINI_CALL_TIMEOUT_SECONDS
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            config = {"response_mime_type": "application/json"} if json_mode else None
            response = await model.generate_content_async(
                [prompt, img],
                generation_config=config,
                request_options={
                    "timeout": call_timeout,
                    "retry": None,
                },
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            if attempt < max_retries and ("RESOURCE_EXHAUSTED" in err_str or "429" in err_str):
                if _is_daily_quota_error(err_str):
                    logger.warning("Gemini daily quota exhausted; failing fast (reset is hours away).")
                    break
                delay = 2 * (attempt + 1)
                logger.warning("Gemini rate limited (attempt %d/%d). Retrying in %ds...", attempt + 1, max_retries + 1, delay)
                await asyncio.sleep(delay)
            elif attempt == 0 and _is_capacity_error(err_str):
                # Transient capacity dip: one short backoff on the primary
                # before failing over. 5s is enough to ride out spikes without
                # materially extending the worst case; 504s are included
                # because an expired server deadline is also capacity talking.
                logger.warning("Gemini capacity dip; pausing 5s before failover...")
                await asyncio.sleep(5)
                try:
                    config = {"response_mime_type": "application/json"} if json_mode else None
                    response = await model.generate_content_async(
                        [prompt, img],
                        generation_config=config,
                        request_options={
                            "timeout": call_timeout,
                            "retry": None,
                        },
                    )
                    return response.text.strip()
                except Exception as e2:
                    last_error = e2
                    err_str = str(e2)
                    break
            else:
                break
    assert last_error is not None
    err_str = str(last_error)
    fallback = _get_fallback_model()
    if fallback is not None and _is_capacity_error(err_str):
        logger.warning(
            "Gemini primary model hit a capacity error; trying fallback model once: %s",
            settings.GEMINI_FALLBACK_MODEL,
        )
        try:
            config = {"response_mime_type": "application/json"} if json_mode else None
            response = await fallback.generate_content_async(
                [prompt, img],
                generation_config=config,
                request_options={
                    "timeout": call_timeout,
                    "retry": None,
                },
            )
            logger.info("Gemini fallback model served the call: %s", settings.GEMINI_FALLBACK_MODEL)
            return response.text.strip()
        except Exception as e:
            logger.warning("Gemini fallback model also failed: %s", str(e)[:200])
    logger.error("Gemini call failed: %s", err_str)
    raise UpstreamError(service="Gemini Vision", detail=err_str)


async def extract_text_with_llm(image_bytes: bytes) -> str:
    return await _call_gemini(TEXT_SYSTEM_PROMPT, image_bytes)


def _clean_latex_in_dict(value: object) -> object:
    if isinstance(value, str):
        return latex_to_unicode(value)
    if isinstance(value, list):
        return [_clean_latex_in_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_latex_in_dict(item) for key, item in value.items()}
    return value


def _sanitize_svg_in_dict(value: object) -> object:
    if isinstance(value, dict):
        if value.get("svg"):
            value["svg"] = sanitize_svg(value["svg"])
        return {key: _sanitize_svg_in_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_svg_in_dict(item) for item in value]
    return value


async def _extract_structured(
    primary_prompt: str, repair_prompt: str, image_bytes: bytes, model_cls: type
) -> BaseModel:
    raw = await _call_gemini(primary_prompt, image_bytes, json_mode=True)
    try:
        data = _clean_latex_in_dict(_extract_json(raw))
        data = _sanitize_svg_in_dict(data)
        return model_cls(**data)
    except (json.JSONDecodeError, ValidationError) as first_err:
        logger.warning("%s JSON parse failed (attempt 1): %s", model_cls.__name__, first_err)
        try:
            repaired = await _call_gemini(
                repair_prompt.replace("{{RAW}}", raw), image_bytes, json_mode=True
            )
            data = _clean_latex_in_dict(_extract_json(repaired))
            data = _sanitize_svg_in_dict(data)
            return model_cls(**data)
        except (json.JSONDecodeError, ValidationError, UpstreamError) as repair_err:
            logger.error("%s repair failed: %s", model_cls.__name__, repair_err)
            raise UpstreamError(
                service="SnapNote AI",
                detail="AI could not structure this image's notes cleanly. Please try again.",
            )


async def extract_study_notes(image_bytes: bytes) -> StudyNotes:
    semantic = settings.DIAGRAM_RENDERER_MODE == "semantic"
    primary = STUDY_NOTES_SEMANTIC_PROMPT if semantic else STUDY_NOTES_SYSTEM_PROMPT
    repair = REPAIR_PROMPT_SEMANTIC if semantic else REPAIR_PROMPT
    result = await _extract_structured(primary, repair, image_bytes, StudyNotes)

    if semantic and result.diagram_spec is not None:
        await _apply_diagram_spec(result, image_bytes)
    _normalize_confidence(result)
    return result


def _normalize_confidence(notes: StudyNotes) -> None:
    """Canonicalize formula confidence labels to the three known states.

    Gemini occasionally emits near-miss spellings (observed:
    "possibleₑxtraction_issue" — U+2091 in place of 'e'; also capitalization
    drift). The frontend keys its subtle-note / Verify Before Studying styling
    off the EXACT string, so a near-miss silently renders as "clear" and hides
    a real warning. NFKC folds the confusable subscript; containment matching
    then maps to the canonical state. Unknown strings are left untouched —
    guessing "clear" would hide evidence, guessing a warning would invent one.
    """
    import unicodedata

    for formula in notes.key_formulas or []:
        folded = unicodedata.normalize("NFKC", str(formula.confidence)).casefold()
        if "extraction" in folded:
            formula.confidence = "possible_extraction_issue"
        elif "context" in folded:
            formula.confidence = "context_needed"
        elif "clear" in folded:
            formula.confidence = "clear"


async def _apply_diagram_spec(notes: StudyNotes, image_bytes: bytes) -> None:
    """Route the semantic spec: deterministic render, or safe per-image fallback.

    Supported type -> deterministic renderer (never touches Gemini again).
    Unsupported type -> bounded legacy best-effort SVG, marked best_effort.
    Anything else (missing/contradictory geometry, no diagram) -> explanation-only.
    A fallback failure never fabricates a diagram — it degrades to explanation-only.
    """
    spec = notes.diagram_spec
    if spec is None:
        return

    for item in spec.uncertain:
        if item.strip() and item.strip() not in notes.verify_before_studying:
            notes.verify_before_studying.append(item.strip())

    if spec.diagram_type not in SUPPORTED_DIAGRAM_TYPES:
        notes.diagram = DiagramRep(present=False, svg="")
        notes.diagram_spec = None
        return

    validation = validate_diagram_spec(spec)
    if validation.valid and validation.canonical is not None:
        notes.diagram = DiagramRep(present=True, svg=render_polar_region(validation.canonical))
    else:
        notes.diagram = DiagramRep(present=False, svg="")
        for reason in validation.reasons:
            text = reason.strip()
            if not text:
                continue
            # The spec leg says "No diagram was detected" when it has no
            # geometry — but the extraction leg may have SEEN a diagram and
            # described it in visual_context. Printing both contradicts the
            # trust layer (observed: case 3 described the Argand diagram and
            # still warned that no diagram was detected). Reword, never drop
            # the uncertainty.
            if "no diagram was detected" in text.casefold() and notes.visual_context.present:
                text = (
                    "The source diagram could not be rebuilt exactly — "
                    "the visual context above describes it."
                )
            if text not in notes.uncertainties:
                notes.uncertainties.append(text)

    notes.diagram_spec = None


async def extract_revision_guide(image_bytes: bytes, context_hint: str = "") -> StudyNotes:
    prompt = REVISION_SYSTEM_PROMPT
    if context_hint:
        prompt += "\n\nEXTRACTED CONTEXT (use it to focus, but only teach what the image shows):\n" + context_hint
    result = await _extract_structured(
        prompt, REVISION_REPAIR_PROMPT, image_bytes, StudyNotes
    )
    _normalize_confidence(result)
    return result


VISUAL_SPEC_PROMPT = r"""You are SnapNote AI. A student uploaded a messy lecture screenshot, and we already extracted study notes from it. We will now build ONE clean, educational visual that helps the student understand this concept. Your job: decide WHAT that visual should be and HOW it must be rendered, and describe it as a STRUCTURED SPEC — not as pixels, not as an SVG, not as a vague prompt.

THE RENDERING RULE (the most important decision you make):
Choose "render_mode": "deterministic" or "generative" based on INFORMATION PRECISION, not the subject name:
  - "deterministic" — when correctness of text, symbols, equations, relationships, or exact structure is CENTRAL to understanding. Our deterministic renderer then draws a clean SVG with your exact equations and steps. Use this for: formulas, derivations, definitions, labeled flows, diagrams where labels matter, physics/math/engineering content, chemistry mechanisms, anything where garbled text would be harmful.
  - "generative" — ONLY when the value is a rich conceptual illustration where exact text is NOT the main payload: conceptual scenes, biology/cell processes, analogy-like images, mechanisms you can convey with pictures. Never use generative mode for content whose understanding depends on reading exact symbols or labels.

Worked mathematics (inequalities, domains, formulas, derivations) is ALWAYS "deterministic".

If in doubt, choose "deterministic". A physics topic like "what is angular momentum?" may still benefit from a generative conceptual illustration — decide by precision, not by topic.

TASK PLANNING (do this FIRST, before choosing any scene):
Decide the teaching job in a "task" object. The task determines the representation — our validator rejects any scene that cannot do the task's job, so plan honestly:
   - "task_type": exactly one of "compare_variants" (lesson contrasts 2-3 alternatives across shared features — motion variants, cell types, reaction mechanisms, market structures), "multi_step_solution" (ordered solution steps combining into one conclusion), "graph_relationship" (a plotted relationship IS the lesson), "vector_arrangement" (bodies, forces/velocities, angles, pivots), "connected_process" (ordered causal/process chain with dependencies/feedback), "coordinate_construction" (points/polygons/dimensions placed on axes), "interval_solution" (worked inequality or domain problem whose answer is a union of intervals — ALWAYS when the lesson solves an inequality), "hierarchy_structure" (whole/part, classification, taxonomy, parent/child or tree structure), "cyclic_process" (repeating stages or a loop), "chronology" (events/phases in time order), "definition_overview" (only when nothing structural fits — last resort, never for comparable/hierarchical/cyclic/chronological/graphable/vector content).
   - "learning_outcome": one sentence stating what the student will be able to do after seeing the visual.
   - "required_elements": every entity the visual MUST show (bodies, points, axes, steps, variants), written as CONCRETE grounded strings — the actual symbols/values/names the scene will render ("x² − x", "F", "domain x > −3", "lens with F and F′"), never placeholder slot nouns ("problem", "domain", "solution", "diagram", "concept"). A required element is a content claim the validator checks against the rendered visual: a placeholder noun that appears nowhere in the scene fails the whole spec. Each must be satisfiable by the chosen family's primitives — do not list what no family can draw.
   - "required_relationships": the connections/transformations/constraints between them.
   - "exact_claims": verbatim equations/values/labels from the image or notes that MUST appear (e.g. domain restrictions, final results). Copy them character-for-character; the validator checks their presence. Only declare claims you will actually PRINT in the visual (scene labels, table cells, composition result/callouts) — do not copy every formula from the notes. Keep these to equations, values, and SHORT labels (max ~60 chars) — NEVER whole prose sentences. A sentence-length claim can only be satisfied by printing that sentence verbatim in the visual, which a clean diagram should not do. For a prose/definition lesson use the term labels instead ("Cell Theory", "Prokaryotic cells"); for a comparison use column/row labels and short values.
     - Every exact claim MUST be rendered where the scene shows it: concept_map node label/detail, comparison table cell/column/row label, panel vector/arc label, process node, plot label, or the composition result/callout. A formula that appears only in required_elements or the study notes is invisible and fails validation.
     - If a claim carries an explanation ("Search: O(h) where h = height of tree"), place the printable core in the scene ("Search: O(h)") and exact-claim only that core. Do not exact-claim explanatory clauses.
   - "acceptable_families": SCENE KINDS allowed for this task, chosen ONLY from this registry — compare_variants: ["comparison"]; multi_step_solution: ["process_flow"]; graph_relationship: ["plot"]; vector_arrangement: ["force_diagram"]; connected_process: ["process_flow"]; coordinate_construction: ["plot"]; interval_solution: ["number_line"]; hierarchy_structure: ["concept_map"]; cyclic_process: ["concept_map"]; chronology: ["concept_map"]; definition_overview: any family. These entries are scene_kind values ("force_diagram" | "process_flow" | "plot" | "generic" | "comparison" | "concept_map" | "number_line") — NEVER task_type names like "definition_overview". Your "scene.scene_kind" MUST appear in acceptable_families, and acceptable_families must never contradict the registry row above.
   - "assumptions": anything you had to assume to draw (empty when nothing assumed). "missing": evidence you needed but could not find.
   - A simple concept gets a simple task (definition_overview with 1-2 elements). Never inflate a simple lesson to justify a fancy scene; never shrink a structured lesson into generic boxes.

THE SCENE PRIMITIVES (deterministic mode):
Inside "deterministic.scene" you describe WHAT must be shown using universal educational primitives. Python decides HOW to draw everything — you NEVER give coordinates, pixel positions, or SVG. You decide only the SEMANTICS: which objects, which vectors, their directions in degrees, and which relationships to draw.
  - "scene_kind": choose the representation from the INFORMATION STRUCTURE, not the subject name:
      * "force_diagram" — bodies, forces, velocities, angles, pivots, lever arms, fields.
      * "process_flow" — a causal/ordered chain where each step produces the next (control loops, pipelines, mechanisms stated as steps).
      * "plot" — a quantitative relationship y=f(x), trajectory, curve, or ray/axis construction.
      * "comparison" — 2-3 alternatives judged against shared features: use "panels" for motion/geometry variants, or "table" for conceptual/feature matrices.
      * "concept_map" — hierarchy/taxonomy/whole-part ("types of", classification, parent/child, tree structure), repeating cycle, or chronology (choose its "layout").
      * "number_line" — worked inequality/domain with an interval-union answer.
      * "generic" — ONLY when none of the above can honestly represent the material; never use it as a default for prose.
  - "concept_map" fields (the relational grammar for classification, cycles and chronology):
      "layout": "tree" | "cycle" | "timeline"
      "nodes": 3-12 items, each {"id": "ascii-id", "label": "short exact label", "detail": "optional one-line teaching note", "parent": "parent-id (tree only; empty for the single root)", "marker": "date/phase tag (timeline only)"}
        - tree: exactly ONE root (empty parent); every other node names an existing parent; depth <= 4. Use it for taxonomies, whole/part breakdowns, classification, tree data structures, "types of X".
          * Nodes at the same level must be the SAME KIND of thing (all cell types, all tree parts, all phases). Never mix a theory, a person, a formula, or a historical event into a type hierarchy — put those in the detail or in composition, or choose another layout.
          * If the lesson mainly CONTRASTS 2-3 alternatives (prokaryote vs eukaryote, SN1 vs SN2, Qc < Kc vs Qc > Kc), use a comparison table instead of a tree. A tree that puts the alternatives in one row is a weak visual.
        - cycle: 3-8 stages in list order; the renderer closes the loop automatically. Use it for repeating biological/chemical cycles and feedback loops (the TCA cycle has 8 intermediates).
        - timeline: 3-6 events in chronological order; put the date/phase in "marker" (<= 28 chars; prefer compact units such as "0.0044 Bya" or "1665"). Use it for history, discovery timelines, evolution, project phases.
        - HARD CAP: a cycle accepts at most 8 nodes, a timeline at most 6. If the source contains more events, choose the most important ones up to that cap and put ONLY those in "task.required_elements" and "task.exact_claims" — never claim an event you do not draw. A tree may use up to 12 nodes.
        - Do NOT put a paragraph in a node. Labels are short (<= 48 chars); details are one clause (<= 90 chars).
      "edges": optional labeled relationships between node ids, e.g. {"source": "root", "target": "branch", "label": "includes"}. For tree layouts the parent field already draws the hierarchy; use edges only for meaningful cross-links.
      "relation": optional exact relationship/answer to highlight.
  - "comparison" fields:
      A scene_kind of "comparison" REQUIRES one of the two payloads below. Emitting scene_kind "comparison" with neither "panels" nor "table" is invalid and will be refused. Use a "table" when the lesson contrasts alternatives across shared features or maps conditions to outcomes; use "panels" only for motion/geometry variants.
      EITHER motion/geometry panels (2-3 variants of one physical idea, e.g. pure translation | pure rotation | combined rolling):
        "panels": exactly 2-3 objects, each {"label": "Pure translation", "object": {"kind": "block"|"disk", "label": "CM"}, "points": [...], "vectors": [...], "arcs": [...], "note": "one short line"}.
          - The object's own label names the centre point (usually "CM"). Put OTHER material points in "points" — do not duplicate CM.
          - "points": [{"label": "P'", "at": "top"}, {"label": "P", "at": "bottom"}] — labeled material points on the body. "at" is one of center|top|bottom|left|right and is REQUIRED for every point.
          - "vectors": [{"label": "v = Rω", "angle_deg": 0, "length": 1.0, "at": "top", "color": "red"}] — "at" anchors the vector to a material point (top = P', center = CM, bottom = P). A vector with "length": 0.0 is REAL PHYSICS: the renderer draws a point marker + label and NEVER an arrow. Use 0.0 for a point that is instantaneously at rest (pure rotation at CM, rolling at the contact point); never use 0.4 or a tiny arrow there. Set "color": "red" on velocity vectors so motion reads distinctly from the body; rotation arcs render green.
          - "arcs": [{"label": "ω", "direction": "ccw"|"cw"}] — one rotation arc ONLY for variants that actually rotate.
          - "note": one short line under the panel (optional). Put exact equations on the vector labels, not only in the note.
      OR a feature matrix (alternatives compared across shared dimensions - cell types, SN1/SN2, market structures, data structures):
        "table": {"rows": [{"label": "Condition"}, {"label": "Direction"}], "columns": [{"label": "Qc < Kc", "values": ["excess reactants", "forward"]}, {"label": "Qc > Kc", "values": ["excess products", "backward"]}], "corner": "FEATURE", "verdict": "optional one-line takeaway", "verdict_basis": [{"column": "Qc < Kc", "row": "Direction"}, {"column": "Qc > Kc", "row": "Direction"}]}
          - 2-6 rows and 2-3 columns are REQUIRED; every column has EXACTLY one value per row, in the same order. A table with one row or one column is invalid.
          - EVERY column must add information. NEVER emit a column whose values merely repeat the row labels (the live failure was a "Condition" column containing "Qc < Kc"/"Qc > Kc" while those exact labels were already the rows) or whose values duplicate another column. A redundant axis is removed; if that leaves only one real dimension the table is refused.
          - "corner" names the row axis in the top-left cell: "FEATURE" when rows are shared features, "DISTRIBUTION" / "ALTITUDE" / "STAGE" when the rows are the alternatives themselves, etc. It must describe the rows truthfully.
          - A matrix may be oriented EITHER way: columns can be the alternatives (rows = shared features, e.g. Qc < Kc | Qc > Kc) OR rows can be the alternatives (columns = shared features, e.g. rows Binomial/Poisson/Normal/Exponential × columns Mean/Variance/MGF). Pick the orientation that keeps rows at 2-6 and columns at 2-3. If the source contrasts MORE than 3 alternatives, transpose so the alternatives are the 4-6 ROWS and the shared metrics are the 2-3 columns — never drop an alternative the lesson is about.
          - "verdict" is the table's SYNTHESIS (a decision rule or takeaway), never a glossary definition of one cell. It must be traceable: emit "verdict_basis" naming at least TWO real printed cells (column label + row label) that support it, and never put a number in the verdict that is absent from the table. The server verifies the cited cells and drops an untraceable verdict; omit both fields when no synthesis follows from the data.
          - Every cell must be non-empty and short (<= 80 chars); use "-" only when the source truly does not apply.
          - Use the exact terms/numbers from the image, not generic paraphrases. A condition→outcome contrast belongs in a table, never in a tree.
  - "force_diagram" fields:
      "object": {"kind": "pivot"|"disk"|"point"|"block", "label": "O" (a short name), "caption": "what it is"}
      "ground": true|false — draw the surface the body rolls/rests on (baseline + contact point). ALWAYS true for a rolling disk, a block sliding on a floor/incline, or any body whose contact with the ground is part of the lesson; false otherwise. Without it a rolling body reads as floating.
      "vectors": [{"label": "r", "angle_deg": 55, "length": 1.0, "tail": "r" (label of the vector this one starts from; omit/empty to start at the object), "color": ""|"accent"|"red"|"green", "caption": "what this vector represents"}]
        - "angle_deg": direction in degrees, 0 = pointing right, 90 = straight up, 180 = left, 270 = down. Use the direction that matches the real diagram.
        - "length": relative length (0.4 to 1.6). 1.0 is the default.
      "angles": [{"label": "θ", "between": ["r", "F"] (labels of exactly two vectors), "caption": "what the angle is"}]
      "arcs": [{"label": "τ", "around": "O" (object label), "direction": "ccw"|"cw", "caption": "e.g. rotation direction"}]
      "relation": {"expression": "τ = r × F" (exact Unicode equation, NO LaTeX, NO backslash), "caption": "one line meaning"}
        - PHYSICS CORRECTNESS (non-negotiable): velocity arrows show instantaneous direction of motion. For rotation about an axis, velocity is TANGENTIAL (perpendicular to the radius at that point) — never radial, never pointing at or away from the centre. A rotation arc shows the turning sense; a velocity vector on a rotating body must be tangent to its circular path. If the source does not fix the tangent direction, omit the vector rather than guessing.
  - "process_flow" fields:
      "nodes": [{"label": "P: Kp e(t)"}, {"label": "I: Ki ∫e dt"}, ...] (2-6 nodes)
        - One node per distinct block visible on the board, in its real order. Keep labels TIGHT: block name + its own formula inline ("P: Kp e(t)", "Σ: sum", "Plant: G(s)"). Do NOT merge parallel blocks (P, I, D side by side) into one node and do NOT serialize a summing junction — a summation point is its own node labeled with Σ. Never transliterate symbols: write "Σ: sum" — never "Summation S" or "Sum"; write "∫ e(t)dt" — never "Integ e(t)dt"; × stays ×, Greek letters stay Greek.
      "connectors": [{"source": 0, "target": 1, "label": "u(t)" (optional), "feedback": false}]
        - "feedback": true on the connector that loops the output back to the input (drawn as a curved dashed return arrow).
        - Do not repeat a node's own label as a connector label (the renderer drops the echo anyway, but the spec should be clean).
      "relation": {"expression": "...", "caption": "..."} (optional)
  - "plot" fields:
      "plot": {"x_label": "x", "y_label": "y", "x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5, "show_grid": true,
               "curves": [{"label": "y = x²", "expr": "x**2", "x_min": 0, "x_max": 2, "style": "solid", "color": "accent"},
                          {"label": "trajectory", "points": [[0,0],[1,1.2],[2,1.8]], "style": "solid", "color": ""}],
               "markers": [{"x": 2.5, "y": 2.5, "label": "F", "color": "accent"}],
               "segments": [{"x1": 2.5, "y1": 0.5, "x2": 2.5, "y2": 4.5, "label": "lens", "style": "dashed", "arrow": false}]}
        - each curve: EITHER "expr" (safe math in x: x, sin(x), cos(x), sqrt(x), exp(x), log(x), pi) with its own x_min/x_max, OR explicit "points" [[x,y],...]. Prefer expr for accuracy. Colors ""|accent|red|green.
        - "markers": labeled points (focal points F, F′, vertices, intercepts) — drawn as a dot + label. "segments": straight primitives with optional "style": "solid"|"dashed", "arrow": true to put an arrowhead at (x2,y2). USE THESE for ray/lens diagrams on axes: the lens is a vertical dashed segment, F and F′ are markers, the object and image are vertical arrow segments, and rays are curves made of explicit points. Never fake a lens/arrow with a text label or a decorative curve.
        - discrete points and polygons: use explicit "points" with a "label" per vertex (e.g. an Argand-plane square: one closed loop [[-2,-2],[2,-2],[2,2],[-2,2],[-2,-2]] plus per-vertex labels like "A z = x + iy"). Vertices belong on the geometry as labels, never only in prose.
        - dimensions: express side lengths as short curve/vertex labels (e.g. "side = 4"), never as decorative text.
        - EQUATION FIDELITY: equation and expression strings must be copied verbatim from the image or notes — especially transformed expressions (e.g. write "z - 2Re(z) = -x + iy" exactly as shown; never rewrite Re(z) as Re(conjugate(z)) or otherwise "simplify"). The renderer prints these strings byte-identical; any mistranscription you emit becomes the student's wrong answer.
   - "number_line" (worked interval mathematics — solving ONE inequality: log inequalities, polynomial/rational inequalities, domains):
     Use this WHENEVER the lesson works through an inequality or a domain whose answer is an interval or union of intervals (log_{x+3}(x² − x) < 1, (x − 2)/(x + 3) ≤ 0, x² − 5x + 6 > 0, "find the domain of ..."). NEVER force such content into "plot", "comparison" or "generic".
       "problem": the ONE inequality exactly as written (plain Unicode, no LaTeX, no words around it), e.g. "log_{x+3}(x² − x) < 1". Copy every symbol/number verbatim.
       "domain": {"label": "x² − x > 0, x + 3 > 0, x + 3 ≠ 1", "intervals": [{"lo": "-3", "hi": "-2"}, {"lo": "-2", "hi": "0"}, {"lo": "1", "hi": "inf"}], "points": ["-2"]}
         - the FULL final domain: apply every restriction (log argument > 0; log base > 0 and ≠ 1; denominators ≠ 0), intersect them, and list the resulting intervals. Excluded single values (like the base = 1 point) go in "points". The verifier REQUIRES a domain track whenever any restriction exists (any log, any denominator) AND requires every excluded value to be visibly represented (as a listed point or as an open endpoint of the domain intervals); OMIT "domain" only for pure polynomial inequalities with no restrictions — omitting it elsewhere is a verification failure.
       "cases": up to 3 branches, each {"condition", "transformed", "solution"}:
         - "condition": exactly "BASE > 1" form (e.g. "x + 3 > 1") for the increasing regime, or exactly "0 < BASE < 1" form (e.g. "0 < x + 3 < 1") for the decreasing regime. NEVER write a bare "BASE < 1" — dropping the positivity bound 0 < BASE makes the condition incorrect and the verifier rejects it.
         - "transformed": REQUIRED for every branch — never leave it empty. The inequality AFTER applying log monotonicity (direction kept for base > 1, REVERSED for 0 < base < 1), e.g. "x² − x < x + 3". Write the SAME inequality derived from monotonicity: bring terms to one side instead of rewriting it into a different form, and never re-solve.
         - "solution": the case's final set with the domain already applied, e.g. "(-1, 0) ∪ (1, 3)".
         - For a CONSTANT base (log_2(...), log_3(...)) there is no case split: OMIT "cases".
       "solution": {"label": "Final answer", "intervals": [{"lo": "-3", "hi": "-2"}, {"lo": "-1", "hi": "0"}, {"lo": "1", "hi": "3"}], "points": []}
         - the COMPLETE union over all cases. Interval endpoints: lo/hi are "-inf"/"inf" or exact numbers ("-3", "1/2", "1.5"); lo_closed/hi_closed (default false = open) say whether the endpoint is included. If the inequality has NO solutions at all, emit empty "intervals" and "points" (the visual renders "No real solutions" explicitly — that is a correct, complete answer, not a failure).
     HARD MATH RULES — a server-side verifier re-solves the problem and REFUSES any wrong domain, case, or final answer (honest refusal beats a wrong visual):
       - Work the problem correctly BEFORE writing the JSON: domain first (argument > 0; base > 0 and ≠ 1), then case split (base > 1 keeps direction; 0 < base < 1 reverses), then intersect each case with the domain, then union.
       - Worked reference example (study the full completeness): for log_{x+3}(x² − x) < 1 the domain is (-3, -2) ∪ (-2, 0) ∪ (1, inf) — x² − x > 0 AND x + 3 > 0 AND x + 3 ≠ 1, so the base=1 point x = -2 is excluded. Case x + 3 > 1 (x > -2): x² − x < x + 3 gives (-1, 3), intersected with the domain: (-1, 0) ∪ (1, 3). Case 0 < x + 3 < 1 (-3 < x < -2): x² − x > x + 3 gives (-3, -2). FINAL ANSWER: (-3, -2) ∪ (-1, 0) ∪ (1, 3). A union like (-3, -2) ∪ (-1, 3) that drops the domain intersection is WRONG.
       - Double-check every endpoint closure (strict <, > = open; ≤, ≥ = closed).
       - If you cannot derive the exact solution, do NOT emit a number_line scene with a guessed answer — refuse honestly by choosing a different honest representation instead.
   - "caption": ONE sentence under "WHAT THE VISUAL SHOWS" teaching what the diagram means conceptually (never an inventory of parts). Example: "Torque magnitude depends on the lever arm r and the angle θ between r and F."

  When the screenshot shows a graph/plot/trajectory (parabola, sine wave, polar curve, projectile), prefer "plot" with a safe expr. When it shows vectors/forces, prefer "force_diagram". When it shows boxes/arrows, prefer "process_flow". When it SOLVES an inequality or finds a domain (answer = interval union), use "number_line" — never generic, never a forced comparison. When it classifies/types/branches (taxonomy, whole/part, tree), uses a repeating cycle, or narrates a chronology, use "concept_map" with the matching layout. When it contrasts 2-3 alternatives across shared features, use "comparison" with a "table". Use "generic" only when the material truly has no structure to draw.

THE COMPOSITION BLOCK (explicit lesson content):
Inside "deterministic.composition" you describe the LESSON the visual teaches — title, framing, callout chips, reasoning chain, highlighted result, takeaway. The renderer draws this block verbatim around the scene geometry, so every word must already be grounded.
  - "title": short title, max 80 chars (usually repeats deterministic.title).
  - "framing": one-line concept framing, max 200 chars.
  - "callouts": 2-4 items, each {"id": "vec-r" (plain ASCII only: lowercase a-z, digits, _ and - only, max 32 chars, referencing what it describes — e.g. "vec-r", "node-err", NEVER Unicode subscripts like "node₀"), "label": "r" (max 24 chars), "value": "55° lever" (max 80 chars)}. Labels/values may use Unicode math (θ, τ, ×); ids MUST be plain ASCII.
  - "reasoning": up to 4 steps, each {"id": "rs-theta" (same plain-ASCII id rule, e.g. "rs-1", "rs-tau"), "expression": "θ = 90° − 55° = 35°" (max 120 chars, exact Unicode math, NO LaTeX), "explanation": "angle between r and F" (max 200 chars)}.
  - "result": {"expression": "τ = r × F" (max 120 chars), "emphasis": true}.
  - "takeaway": one sentence, max 200 chars.
  COMPOSITION GROUNDING (never invent to make the visual richer):
  - Every callout label/value must come from a scene label you already emitted or verbatim text in the image/notes. Never invent new symbols, numbers, or names.
  - Every reasoning expression must use ONLY equations/symbols visible in the image, the notes, or the scene you emitted. Arithmetic combining values you already emitted (e.g. θ = 90° − 55°) is allowed; new factual claims are forbidden.
  - The result expression must EQUAL the scene's own relation expression when a relation exists; otherwise it must be an equation verbatim from the image/notes.
  - If you cannot fill callouts/reasoning/result from visible evidence, OMIT the entire "composition" block. The renderer falls back to derived content automatically. Never hallucinate missing reasoning merely to fill the template.

GROUNDING RULES:
1. The screenshot is the source of truth. The study notes below are helpful context, but never invent content that conflicts with what the image actually shows.
2. Never invent formulas, quantities, or relationships that are not in the image or the notes. If something is cut off or ambiguous, leave it out of the visual rather than guessing.
3. Keep the visual clean and minimal: white/light background, dark high-contrast text, flat vector style, no photo-realism, no watermark, no decoration. Prefer a real information structure (hierarchy, cycle, timeline, feature matrix, process) over a card grid of sentences. A student should understand MORE from the composition than from a plain text summary.
4. Write labels as short, plain, readable text. Preserve mathematical symbols VERBATIM: Σ, ∫, ×, ±, √, π, θ, ω, τ and other Greek letters stay exactly as written — never transliterate them into ASCII words or single letters ("Summation S", "Integ", "sqrt"). A student must read the same mathematics they saw on the board.
5. In "scene", only emit primitives you are confident about. A torque/force concept should use a force_diagram (pivot + r vector + F vector + θ angle + τ rotation). A control-loop concept should use a process_flow. A graph/trajectory should use a plot. An inequality/domain solution should use a number_line. A taxonomy/classification/whole-part lesson should use concept_map (tree); a repeating cycle should use concept_map (cycle); a chronology should use concept_map (timeline). A conceptual contrast across features should use comparison (table). Only when none of those structures honestly fits should you use generic. When the lesson explicitly compares 2-3 motion variants of one idea (e.g. pure translation vs pure rotation vs combined motion), use "comparison" with one panel per variant: each panel has a short label, an object (block for sliding, disk for rolling/spinning), the motion's own velocity vectors, a rotation arc ONLY where the variant actually rotates, and a one-line note. Velocity vectors on rotating bodies must be TANGENTIAL (perpendicular to the radius), never radial. For a rolling-motion board (pure translation vs pure rotation vs combined rolling), the comparison MUST have exactly 3 panels and every disk MUST carry the material points P' (top), CM (center), and P (bottom):
       • Pure translation: vectors at top/CM/bottom all point right and are labeled "v_CM"; no arc.
       • Pure rotation: top vector right labeled "v = Rω"; CM vector length 0.0 labeled "v = 0" (a dot, never an arrow); bottom vector left labeled "v = Rω"; one rotation arc.
       • Combined rolling: top vector right labeled "v = v_CM + Rω = 2v_CM"; CM vector right labeled exactly "v = v_CM" (the equality is required — bare "v_CM" belongs only in the pure-translation panel); bottom vector length 0.0 labeled "v = 0" (a dot, never an arrow); one rotation arc.
       The exact claims v = v_CM + Rω = 2v_CM, v = v_CM, and v = 0 MUST be printed on these panel vector labels and included in task.exact_claims; putting a motion claim only in equations, a note, or a composition callout leaves the diagram physically incomplete. Never omit the third panel or a zero-velocity endpoint. Never use "generic" for a comparison lesson — use the feature table when the contrast is conceptual rather than motion-based; a central box with formula callouts teaches nothing. Only when no real structure fits may you use "generic" with central_label (2-4 words) and 3-5 callouts, each written as "Label: value" — one idea per callout, max ~52 chars, carrying BOTH the symbol/formula and its meaning (e.g. "Σ i = n(n+1)/2: sum of first n integers") so the visual is self-contained. Never leave scene empty if a real structure can honestly be drawn.

OUTPUT: ONLY a JSON object with exactly this structure:
{
  "concept": "one phrase naming the concept the visual explains",
  "render_mode": "deterministic | generative",
  "text_required": true,
  "deterministic": {
    "title": "short title (2-6 words)",
    "task": {"task_type": "compare_variants", "learning_outcome": "one sentence", "required_elements": ["each entity the visual must show"], "required_relationships": [], "exact_claims": ["verbatim equations/values"], "acceptable_families": ["comparison"], "assumptions": [], "missing": []},
    "scene": {
      "scene_kind": "force_diagram | process_flow | plot | generic | comparison | concept_map | number_line",
      "caption": "one teaching sentence about what the diagram means",
      "force": { "object": {...}, "ground": false, "vectors": [...], "angles": [...], "arcs": [...], "relation": {...} },
      "flow": { "nodes": [...], "connectors": [...], "relation": {...} },
      "plot": { "x_label": "x", "y_label": "y", "x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5, "show_grid": true, "curves": [{"label": "y=x²", "expr": "x**2", "x_min": 0, "x_max": 2}], "markers": [{"x": 2, "y": 4, "label": "F"}], "segments": [{"x1": 2, "y1": 0, "x2": 2, "y2": 5, "label": "lens", "style": "dashed"}] },
      "generic": { "central_label": "Main idea", "callouts": ["F = ma: force equals mass times acceleration", "Σ i = n(n+1)/2: sum of first n integers"] },
      "comparison": { "panels": [{"label": "Pure translation", "object": {"kind": "disk", "label": "CM"}, "points": [{"label": "P'", "at": "top"}, {"label": "P", "at": "bottom"}], "vectors": [{"label": "v_CM", "angle_deg": 0, "length": 1.0, "at": "top"}, {"label": "v_CM", "angle_deg": 0, "length": 1.0, "at": "center"}, {"label": "v_CM", "angle_deg": 0, "length": 1.0, "at": "bottom"}], "arcs": [], "note": "all points share v_CM"}], "relation": {"expression": "v = ωr", "caption": "rolling link"} },
      "concept_map": { "layout": "tree", "nodes": [{"id": "root", "label": "Cell", "detail": "basic unit of life", "parent": ""}, {"id": "prok", "label": "Prokaryotic", "detail": "no true nucleus", "parent": "root"}, {"id": "euk", "label": "Eukaryotic", "detail": "true nucleus", "parent": "root"}], "edges": [{"source": "root", "target": "prok", "label": "type"}, {"source": "root", "target": "euk", "label": "type"}], "relation": null },
      "number_line": {
        "problem": "log_{x+3}(x² − x) < 1",
        "domain": {"label": "x² − x > 0, x + 3 > 0, x + 3 ≠ 1", "intervals": [{"lo": "-3", "hi": "-2"}, {"lo": "-2", "hi": "0"}, {"lo": "1", "hi": "inf"}], "points": ["-2"]},
        "cases": [
          {"condition": "x + 3 > 1", "transformed": "x² − x < x + 3", "solution": "(-1, 0) ∪ (1, 3)"},
          {"condition": "0 < x + 3 < 1", "transformed": "x² − x > x + 3", "solution": "(-3, -2)"}
        ],
        "solution": {"label": "Final answer", "intervals": [{"lo": "-3", "hi": "-2"}, {"lo": "-1", "hi": "0"}, {"lo": "1", "hi": "3"}], "points": []}
      }
    },
    "equations": [{"expression": "exact formula in Unicode, no LaTeX, no backslash", "meaning": "one line: what each symbol means and what the relationship represents"}],
    "steps": ["ordered steps, each a short phrase"],
    "points": ["key points, each a short phrase"],
    "composition": {
      "title": "short title, max 80 chars",
      "framing": "one-line concept framing, max 200 chars",
      "callouts": [{"id": "vec-r", "label": "r", "value": "55° lever"}],
      "reasoning": [{"id": "rs-theta", "expression": "θ = 90° − 55° = 35°", "explanation": "angle between r and F"}],
      "result": {"expression": "τ = r × F", "emphasis": true},
      "takeaway": "one sentence, max 200 chars"
    }
  },
  "visual_form": "the chosen visual form, e.g. 'force vector diagram' or 'labeled block diagram' or 'step-by-step flowchart'",
  "key_elements": ["every box/axis/label/marker the visual must contain, verbatim text in quotes"],
  "key_relationships": ["each relationship the visual must show, e.g. 'torque = cross product of r and F'"],
  "must_show": ["the essential things that must be visually present so the student understands"],
  "avoid": ["anything to leave out: unsupported quantities, decorative objects, unrelated details"]
}

FIELD RULES:
- "task": ALWAYS emit it (plan first, then scene). The validator enforces that "scene.scene_kind" is in "task.acceptable_families", that every "required_elements" entry is satisfiable by the scene, and that every "exact_claims" string appears verbatim. A spec whose scene cannot do its task is refused.
- "text_required": true when readable text/symbols are essential to the visual (always true in deterministic mode). false only for a purely conceptual illustration that conveys meaning through pictures alone.
- "deterministic": ALWAYS populate it. Prefer providing a "scene" (a real diagram) when the material has ANY honest structure — force_diagram, process_flow, plot, comparison (panels or table), concept_map (tree/cycle/timeline), or number_line; the renderer draws it. IMPORTANT — Explain Visually is a VISUAL ARTIFACT, not a second study sheet: when you emit a scene, keep it structure-first (objects, relationships, nodes, cells, vectors, labels) and do NOT fill equations/steps/points with the same material that already lives in the study notes. Leave equations/steps/points EMPTY when a scene is present; the only equation allowed inside the visual is the scene's own "relation" (e.g. "τ = r × F") plus the one-line caption. Fill equations/steps/points ONLY when there is no scene (the renderer then falls back to a card layout). In generative mode, still include at least title and any one exact relationship you do not want a generative model to garble (the renderer ignores it, but it keeps the exact content available). If nothing exact applies, keep deterministic.title set and leave the lists empty.
- Keep each list concise (2-6 items). Ground every item in the image and notes. If the material genuinely cannot benefit from a visual (e.g. pure prose with no structure worth drawing), set render_mode to "generative", concept to the topic, visual_form to "simple illustration", key_elements to one broad item like "the central idea shown as a simple icon", and avoid anything ungrounded — never invent a diagram the material doesn't support.
- "composition": emit it ONLY when you have enough structured evidence (scene labels + visible equations) to fill callouts/reasoning/result honestly. Omit the block entirely when evidence is thin — the renderer falls back gracefully. Never emit a composition whose values you invented."""


def _minimal_notes_context(study_notes: StudyNotes) -> str:
    """Compact grounded context for the repair leg: topic + formulas only.

    The full notes JSON (uncertainties, analogy, visual context, ...) is
    unnecessary for reconciling one expression pair and only slows the
    repair call. Formulas carry confidence so the repair prefers clear
    extractions.
    """
    topic = ""
    try:
        if study_notes.topic is not None:
            topic = study_notes.topic.title or ""
    except AttributeError:
        topic = ""
    formulas = []
    for f in study_notes.key_formulas or []:
        formulas.append({"formula": f.formula, "confidence": f.confidence})
    return json.dumps({"topic": topic, "formulas": formulas}, ensure_ascii=False)


def _normalize_task_enums(data: dict) -> dict:
    """Recover near-miss LLM enum spellings before strict validation.

    Models emit confusable Unicode (observed: "vectorₐrrangement" with U+2090
    and a missing underscore). NFKC folds the confusables; the remainder is
    matched against known values with non-alphanumerics stripped. Only exact
    post-strip matches are repaired — anything else passes through untouched
    so validation still refuses genuinely unknown kinds. Applied ONLY to the
    closed enum fields task_type/scene_kind — equation and label strings stay
    byte-identical (math is case- and glyph-significant).
    """
    import re
    import unicodedata

    from app.models.schemas import VisualTask

    valid_tasks = list(VisualTask.model_fields["task_type"].annotation.__args__)  # type: ignore[union-attr]
    valid_kinds = ["force_diagram", "process_flow", "plot", "generic", "comparison", "concept_map", "number_line"]

    def _recover(value: str, valid: list[str]) -> str:
        folded = unicodedata.normalize("NFKC", value).strip()
        if folded in valid:
            return folded
        bare = re.sub(r"[^a-z0-9]", "", folded.lower())
        for candidate in valid:
            if re.sub(r"[^a-z0-9]", "", candidate.lower()) == bare:
                return candidate
        return value

    det = data.get("deterministic")
    if isinstance(det, dict):
        task = det.get("task")
        if isinstance(task, dict) and isinstance(task.get("task_type"), str):
            task["task_type"] = _recover(task["task_type"], valid_tasks)
        scene = det.get("scene")
        if isinstance(scene, dict) and isinstance(scene.get("scene_kind"), str):
            scene["scene_kind"] = _recover(scene["scene_kind"], valid_kinds)
    return data


def _normalize_composition_ids(data: dict) -> dict:
    """Fold model-authored composition ids to the ASCII contract (pure).

    Ids are internal anchors, not student-visible content. The model still
    emits Unicode subscripts occasionally ("eqₐ", "rs₁"); the pydantic pattern
    then crashed the repair leg and turned a recoverable spec into a refusal.
    NFKC folds the confusables (ₐ→a, ₁→1); any remaining disallowed character
    becomes "-". Equation/label text is untouched. Duplicate ids after folding
    still fail honest composition validation downstream.
    """
    import re
    import unicodedata

    det = data.get("deterministic")
    if not isinstance(det, dict):
        return data
    comp = det.get("composition")
    if not isinstance(comp, dict):
        return data
    for section in ("callouts", "reasoning"):
        items = comp.get(section)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            raw = unicodedata.normalize("NFKC", item["id"]).strip().casefold()
            norm = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
            if not norm:
                norm = f"item-{index + 1}"
            item["id"] = norm[:32]
    return data


def _sanitize_task_claims(spec: VisualSpec) -> int:
    """Drop prose sentences from the planner's exact-claim list (pure).

    The planner occasionally copies full sentences into ``exact_claims``
    ("If Qc < Kc, reaction proceeds forward."). Those are summaries, not exact
    claims: they cannot be printed verbatim in a clean diagram without turning
    it into a paragraph, and failing the whole visual because of one such
    entry is a false refusal. Dropping the claim changes no visible content -
    the grounded condition still lives in the scene/reasoning - and everything
    that IS a short equation/label/value passes through untouched.
    """
    task = spec.deterministic.task
    if task is None:
        return 0
    stopwords = {
        "the", "a", "an", "is", "are", "of", "to", "in", "for", "with",
        "where", "when", "which", "that", "this", "and", "or", "as", "by",
        "from", "on", "at", "it", "its", "be", "can", "will", "must",
        "should", "may", "proceeds", "occurs", "means", "represents",
    }
    kept: list[str] = []
    dropped = 0
    for claim in task.exact_claims:
        norm = " ".join(str(claim).split())
        if not norm:
            continue
        words = re.findall(r"[A-Za-z]+", norm)
        sentence = norm.endswith((".", "!", "?")) and len(words) >= 5
        prose = len(words) >= 8 and sum(w.casefold() in stopwords for w in words) >= 3
        if sentence or prose or len(words) >= 14:
            dropped += 1
            continue
        kept.append(claim)
    task.exact_claims = kept
    return dropped


def _apply_table_quality(spec: VisualSpec) -> VisualSpec:
    """Deterministically gate comparison axes/verdicts at the model boundary.

    The renderer and validator apply the same normalization defensively; doing
    it here means repair calls receive the cleaned spec and logs name the
    exact dimension/verdict the model got wrong.
    """
    gated, report = normalize_visual_spec(spec)
    if report.dropped_columns:
        logger.info(
            "VisualSpec removed redundant comparison column(s): %s",
            ", ".join(report.dropped_columns),
        )
    if report.dropped_verdict:
        logger.warning(
            "VisualSpec dropped untraceable comparison verdict (%s): %r",
            report.verdict_reason,
            report.dropped_verdict,
        )
    if report.corrected_corner:
        logger.info(
            "VisualSpec corrected comparison corner %r -> 'FEATURE'",
            report.corrected_corner,
        )
    return gated


async def build_visual_spec(image_bytes: bytes, study_notes: StudyNotes | None) -> VisualSpec:
    """Produce the structured VisualSpec that drives the Explain Visually image.

    Stage 1 of the pipeline: Gemini reads the screenshot (ground truth) plus the
    already-extracted study notes (context) and decides what educational visual
    to draw — as a semantic spec, never as pixels. This keeps the image
    generation grounded even though the raster provider (Pollinations) is
    text-to-image only.
    """
    prompt = VISUAL_SPEC_PROMPT
    if study_notes is not None:
        notes_json = study_notes.model_dump_json(exclude={"diagram_spec", "diagram"})
        prompt += "\n\nSTUDY NOTES (context — do not contradict the screenshot):\n" + notes_json
    raw = await _call_gemini(prompt, image_bytes, json_mode=True)
    try:
        data = _normalize_composition_ids(
            _normalize_task_enums(_clean_latex_in_dict(_extract_json(raw)))
        )
        from app.utils.latex_clean import normalize_spec_math

        spec = normalize_spec_math(VisualSpec(**data))
        spec = _apply_table_quality(spec)
        dropped_claims = _sanitize_task_claims(spec)
        if dropped_claims:
            logger.info(
                "VisualSpec dropped %d prose sentence(s) from exact_claims",
                dropped_claims,
            )
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("VisualSpec parse failed: %s", e)
        raise UpstreamError(
            service="SnapNote AI",
            detail="Could not plan an educational visual for this material.",
        )
    # One-shot contradiction repair (v3 composition): the scene relation and
    # the composition result both render as the visual's single highlighted
    # answer, so a mismatch is refused downstream. Real boards can carry two
    # grounded-but-complementary forms (e.g. "τ = r × F" and "|τ| = rF sin θ"),
    # and Gemini non-deterministically picks different ones per slot — so give
    # it exactly one bounded second chance to reconcile to a single grounded
    # expression. Anything still mismatched (or any other failure) refuses
    # honestly as before. At most one extra Gemini call, only on mismatch.
    contradiction = _composition_contradiction(spec)
    if contradiction is not None:
        relation_expr, result_expr = contradiction
        logger.warning(
            "VisualSpec composition contradicts scene relation (%r != %r); attempting one repair",
            result_expr,
            relation_expr,
        )
        repair_prompt = (
            COMPOSITION_RECONCILE_PROMPT.replace("{{RELATION}}", relation_expr)
            .replace("{{RESULT}}", result_expr)
            .replace("{{PREVIOUS_JSON}}", json.dumps(data, ensure_ascii=False))
        )
        if study_notes is not None:
            repair_prompt += (
                "\n\nGROUNDED CONTEXT (reconcile only against these; "
                "do not contradict the screenshot):\n" + _minimal_notes_context(study_notes)
            )
        try:
            # Tight sub-budget, no retry: reconciliation is small; expiry
            # refuses honestly instead of stretching the request.
            from app.config import settings

            repair_raw = await _call_gemini(
                repair_prompt,
                image_bytes,
                max_retries=0,
                json_mode=True,
                timeout_seconds=settings.GEMINI_REPAIR_TIMEOUT_SECONDS,
            )
            repair_data = _normalize_composition_ids(
                _normalize_task_enums(_clean_latex_in_dict(_extract_json(repair_raw)))
            )
            from app.utils.latex_clean import normalize_spec_math

            repaired = normalize_spec_math(VisualSpec(**repair_data))
            _sanitize_task_claims(repaired)
            repaired = _apply_table_quality(repaired)
        except (json.JSONDecodeError, ValidationError, UpstreamError) as e:
            logger.warning("VisualSpec contradiction repair failed: %s", e)
            raise UpstreamError(
                service="SnapNote AI",
                detail="Could not plan an educational visual for this material.",
            )
        if _composition_contradiction(repaired) is not None:
            logger.warning("VisualSpec contradiction persists after repair; refusing")
            raise UpstreamError(
                service="SnapNote AI",
                detail="Could not plan an educational visual for this material.",
            )
        return repaired
    return spec


COMPOSITION_RECONCILE_PROMPT = r"""You previously emitted the visual spec below, but its scene relation and composition result disagree, so the visual would show two different highlighted answers.

- Scene relation expression: "{{RELATION}}"
- Composition result expression: "{{RESULT}}"

YOUR PREVIOUS RESPONSE:
{{PREVIOUS_JSON}}

Re-emit the COMPLETE spec JSON with this fixed: choose the ONE expression that is actually grounded in the screenshot and study notes and put that identical expression in BOTH the scene relation and the composition result (whitespace differences are fine). The chosen expression MUST be a complete mathematical statement (contain "=" or a comparison operator) — a block name or node label such as "Plant G(s)" is never the relation; if the two candidates are "Plant G(s)" and an actual equation, the equation is the answer. If neither expression is truly grounded, omit the composition block entirely instead — never invent a third expression to paper over the disagreement. Keep every other field identical to your previous response above. Respect all field constraints from the original prompt (callout label max 24 chars, value max 80; reasoning expression max 120; result max 120; takeaway max 200; plain-ASCII ids). Return ONLY the JSON object, no other text."""


def _composition_contradiction(spec: VisualSpec) -> tuple[str, str] | None:
    """Return (relation, result) when both are present but disagree (pure).

    Whitespace-insensitive comparison; math is case-significant so no case
    folding. None means no contradiction (either side absent, or agreement).
    """
    from app.utils.visual_lesson import _scene_relation_expression

    det = spec.deterministic
    comp = det.composition
    if comp is None or det.scene is None:
        return None
    relation_expr = _scene_relation_expression(det.scene)
    result_expr = (comp.result.expression if comp.result is not None else "") or ""
    if relation_expr.strip() and result_expr.strip():
        if " ".join(relation_expr.split()) != " ".join(result_expr.split()):
            return relation_expr, result_expr
    return None


THIN_SCENE_REPAIR_PROMPT = r"""Your previous visual plan was not acceptable: {REASON}. A central box with a few callouts does not teach the lesson, and a wrong mathematical answer is worse than no visual at all.

Look at the screenshot again together with the grounded context below. If the lesson visibly supports a concrete scene — comparable variants or a feature matrix (use "comparison"), vectors/forces/angles ("force_diagram"), curves or plotted points ("plot"), process steps ("process_flow"), a taxonomy/cycle/chronology (use "concept_map" with layout tree/cycle/timeline), or a worked INEQUALITY/DOMAIN whose answer is an interval union ("number_line") — re-emit the COMPLETE spec JSON with that scene, using labels, values, and equations verbatim from the image or notes.

If the reason mentions a mathematics verification mismatch (phrases like "mathematically wrong", "cannot verify", "correct answer is", "correct domain is", "domain track is required", "case branch", "omits the positivity"), redo the solution carefully and re-emit a corrected "number_line" scene: apply the domain restrictions first (log argument > 0, log base > 0 and ≠ 1, denominators ≠ 0), ALWAYS include the "domain" track whenever any restriction exists, split by case (base > 1 keeps the inequality direction; 0 < base < 1 reverses it — write the condition with its full positivity bound, never a bare "BASE < 1"), intersect EVERY case with the domain, and only then take the union. Use the exact correct sets quoted in the reason when they are given. Never repeat the wrong set.

If the reason mentions "number_line has no number_line payload", the scene kind is right but the payload is empty: emit the COMPLETE "number_line" object (problem + domain when any restriction exists + cases + solution) as the WORKED MATHEMATICS rules describe. Never leave it null.

If the reason mentions "exact claim missing from scene", do NOT edit or drop short equation/label claims and do NOT change the scene kind. Repair by putting each claim where it is VISIBLE and physically attached:
- If the current scene is "comparison" and uses motion panels: the claims belong to the panels, not to equations or a general callout. Add the missing panel (maximum 3 total) or the missing vector/material point to the correct panel. Anchors are REQUIRED: "points": [{"label": "P'", "at": "top"}, {"label": "P", "at": "bottom"}], and vectors use "at": "top"|"center"|"bottom". A zero velocity is a real vector with "length": 0.0 and label "v = 0" (a dot, never an arrow). For rolling motion, the complete repair is three panels — Pure translation (all three v_CM vectors), Pure rotation (top v = Rω, CM length 0 v = 0, bottom v = Rω, one arc), Combined rolling (top v = v_CM + Rω = 2v_CM, CM v = v_CM, bottom length 0 v = 0, one arc). Never satisfy a missing motion claim with a composition callout while leaving the panel/vector absent.
- If the current scene is a comparison feature table: put each missing label/value into the correct table column's "values" (same row order) or into a row label. Every column must keep exactly one value per row. Never add a column that repeats the row labels or duplicates another column — remove such an axis instead. A verdict needs "verdict_basis" citing at least two real cells; otherwise leave the verdict empty.
- If the current scene is "force_diagram", "plot", or "process_flow": keep the existing scene kind and geometry; add the missing claims to "deterministic.equations" (expression = the claim) and/or a composition callout (short label, claim as value).
- If the current scene is "concept_map": put the claim into a node label, node detail, edge label, or the relation - whichever is semantically true. Never add a node that the screenshot does not support. A cycle accepts at most 8 nodes and a timeline at most 6: select the most important events up to that cap and make task.required_elements/exact_claims name only those visible events.
- If the current scene is already "generic": add the claims as its callouts (3-5 verbatim callouts).
Switching a force_diagram/plot/process_flow to "generic" is not a repair. Never add a motion claim only to a caption, note, or equation while the panel geometry still does not show it.

If the reason mentions "comparison has no comparison payload", the scene kind is right but the payload is missing. Re-emit the COMPLETE spec with a populated "comparison": a "table" for a feature/condition contrast (2-6 rows, 2-3 columns, one value per row per column) or "panels" for motion variants. For a source that contrasts MORE than 3 alternatives (e.g. four probability distributions), transpose the matrix: put the alternatives in the 4-6 ROWS and the 2-3 shared metrics in the COLUMNS, with every cell filled from the board. If neither can be grounded, switch to "concept_map" (taxonomy/cycle/timeline) or "generic" — never leave the payload null.
If the reason mentions a comparison panel "without vectors or arcs" (or "comparison needs at least 2 labeled panels"), do NOT force motion geometry onto a prose lesson. If the material contrasts 2-3 alternatives across shared features, re-emit "comparison" with a "table" (rows = shared dimensions, columns = alternatives, one exact value per cell). If it is a taxonomy/whole-part/cycle/chronology, re-emit "concept_map" with the matching layout. Only if neither structure honestly fits, use "generic" with a central_label and 3-5 exact callouts.

If the lesson is genuinely simple or nothing drawable is honestly supported, return the spec JSON unchanged. Never invent formulas, numbers, vectors, curves, or relationships to fill the page. Return ONLY the JSON object, no other text.

HARD FIELD CONSTRAINTS (violations fail validation — respect every one):
- callout ids: plain ASCII lowercase/digits/_/-, max 32 chars (never Unicode subscripts).
- callout label max 24 chars, value max 80 chars. reasoning expression max 120 chars,
  explanation max 200 chars. result expression max 120 chars. takeaway max 200 chars.
- scene_kind must be exactly one of: force_diagram, process_flow, plot, generic, comparison, concept_map, number_line.
- task.task_type must be exactly one of: compare_variants, multi_step_solution,
  graph_relationship, vector_arrangement, connected_process,
  coordinate_construction, definition_overview, interval_solution,
  hierarchy_structure, cyclic_process, chronology.
- task.acceptable_families must include your scene_kind; every required_elements
  entry must be satisfiable by the scene you emit; every exact_claims string
  must appear verbatim in scene labels or composition.
- number_line interval endpoints are "-inf"/"inf" or exact numbers ("-3", "1/2", "0.75");
  decimal endpoints are committed exactly — write every digit you mean, never round;
  lo_closed/hi_closed mark included endpoints; "transformed" is REQUIRED for every case;
  case conditions are exactly "BASE > 1" or "0 < BASE < 1" forms (the positivity bound
  must never be dropped); omit "cases" only for a constant base; include the "domain"
  track whenever any restriction exists (any log or denominator) and list EVERY excluded
  value in "points" (or as an open endpoint of the domain intervals); an inequality with
  no solutions emits empty intervals and points.

YOUR PREVIOUS RESPONSE:
{PREVIOUS_JSON}"""


COMPARISON_PAYLOAD_PROMPT = r"""The lesson's scene is "comparison", but its payload is empty, so no visual can be drawn. Extract ONLY the comparison payload from the screenshot — do NOT re-emit the whole spec.

Choose exactly ONE shape:

FEATURE MATRIX — when the lesson contrasts alternatives across shared dimensions:
{"table": {"rows": [{"label": "Binomial"}, {"label": "Poisson"}], "columns": [{"label": "Mean", "values": ["np", "l"]}, {"label": "Variance", "values": ["np(1-p)", "l"]}], "corner": "DISTRIBUTION", "verdict": "one grounded synthesis sentence, or empty", "verdict_basis": [{"column": "Mean", "row": "Binomial"}, {"column": "Mean", "row": "Poisson"}]}}

MOTION PANELS — ONLY when the screenshot itself shows motion/geometry variants (translation/rotation/rolling):
{"panels": [{"label": "Pure translation", "object": {"kind": "disk", "label": "CM"}, "points": [{"label": "P'", "at": "top"}], "vectors": [{"label": "v_CM", "angle_deg": 0, "length": 1.0, "at": "top"}], "arcs": [], "note": "one short line"}]}

Matrix rules (violations make the payload invalid):
- 2-6 row labels and 2-3 columns; every column has EXACTLY one non-empty value per row, in row order.
- Every column must add information. Never repeat the row labels inside a column (the observed failure: a "Condition" column whose cells were the row labels themselves), and never duplicate another column. If only one real dimension remains, choose another representation.
- A matrix may be oriented either way. If the source contrasts MORE than 3 alternatives, put the alternatives in the ROWS and the 2-3 shared metrics in the COLUMNS (e.g. Binomial/Poisson/Normal/Exponential as rows × Mean/Variance/MGF as columns). Never drop an alternative the lesson is about.
- "corner" names the row axis truthfully: "FEATURE" when rows are shared features, "DISTRIBUTION"/"SPECIES"/"STAGE" when rows are the alternatives.
- "verdict" states the practical synthesis of the matrix (a decision rule or takeaway), never a glossary definition of a single cell; leave it empty when no synthesis is grounded. When present, "verdict_basis" MUST cite at least two real printed cells (column label + row label) that support it, and the verdict must not introduce a number absent from the table.
- Copy every cell verbatim from the screenshot/notes; keep exact symbols/numbers. Use "-" only when the source cell truly does not apply.
- Labels max 40 chars, cells max 80 chars, verdict max 120 chars.

Return ONLY the JSON object, no prose and no other fields."""


def _needs_comparison_payload_repair(reason: str) -> bool:
    """Whether the repair reason is the empty comparison payload (pure)."""
    lowered = reason.lower()
    return (
        "comparison has no comparison payload" in lowered
        or "comparison needs at least 2 labeled panels" in lowered
    )


async def _repair_comparison_payload(
    spec: VisualSpec,
    image_bytes: bytes,
    study_notes: StudyNotes | None,
) -> VisualSpec | None:
    """One focused call that fills an empty comparison payload (pure merge).

    Re-emitting the complete nested spec to fill one missing object proved
    unreliable for dense multi-column boards (the model returned the same null
    payload). A small purpose-built extraction call succeeds because the whole
    response is the table/panels. This still runs inside the single repair
    budget; any failure returns None and the caller keeps the honest refusal.
    """
    prompt = COMPARISON_PAYLOAD_PROMPT
    if study_notes is not None:
        prompt += (
            "\n\nGROUNDED CONTEXT (use only these plus the screenshot; "
            "do not contradict the screenshot):\n" + _minimal_notes_context(study_notes)
        )
    try:
        repair_raw = await _call_gemini(
            prompt,
            image_bytes,
            max_retries=0,
            json_mode=True,
            timeout_seconds=settings.GEMINI_REPAIR_TIMEOUT_SECONDS,
        )
        data = _clean_latex_in_dict(_extract_json(repair_raw))
        nested = data.get("comparison")
        payload = nested if isinstance(nested, dict) else data
        if not isinstance(payload, dict):
            return None
        comparison = VisualComparison(**payload)
    except (json.JSONDecodeError, ValidationError, UpstreamError, ValueError) as e:
        logger.warning("Comparison payload repair failed: %s", e)
        return None
    scene = spec.deterministic.scene
    if scene is None:
        return None
    scene.scene_kind = "comparison"
    scene.comparison = comparison
    spec = _apply_table_quality(spec)
    scene = spec.deterministic.scene
    comparison = scene.comparison if scene is not None else None
    if comparison is not None and comparison.table is not None:
        rows = [r for r in comparison.table.rows if (r.label or "").strip()]
        columns = [c for c in comparison.table.columns if (c.label or "").strip()]
        if len(rows) < 2 or len(columns) < 2:
            return None
        if any(len(c.values) != len(rows) for c in columns):
            return None
    elif comparison is None or len(comparison.panels) < 2:
        return None
    return spec


async def repair_thin_spec(
    spec: VisualSpec,
    image_bytes: bytes,
    study_notes: StudyNotes | None,
    reason: str,
) -> VisualSpec | None:
    """One bounded attempt to upgrade a geometry-poor spec (pure).

    Returns the repaired spec, or None when the repair call itself fails
    (caller keeps the original). At most one extra Gemini call, tight
    sub-budget, no retry — same discipline as the contradiction repair.
    """
    if _needs_comparison_payload_repair(reason):
        # A dense matrix is best repaired by a response that IS the matrix,
        # not by asking the model to reproduce a large nested spec again.
        return await _repair_comparison_payload(spec, image_bytes, study_notes)
    prompt = THIN_SCENE_REPAIR_PROMPT.replace("{REASON}", reason).replace(
        "{PREVIOUS_JSON}", spec.model_dump_json(exclude_none=False)
    )
    if study_notes is not None:
        prompt += (
            "\n\nGROUNDED CONTEXT (use only these plus the screenshot; "
            "do not contradict the screenshot):\n" + _minimal_notes_context(study_notes)
        )
    try:
        from app.config import settings

        repair_raw = await _call_gemini(
            prompt,
            image_bytes,
            max_retries=0,
            json_mode=True,
            timeout_seconds=settings.GEMINI_REPAIR_TIMEOUT_SECONDS,
        )
        repair_data = _normalize_composition_ids(
            _normalize_task_enums(_clean_latex_in_dict(_extract_json(repair_raw)))
        )
        from app.utils.latex_clean import normalize_spec_math

        repaired = normalize_spec_math(VisualSpec(**repair_data))
    except (json.JSONDecodeError, ValidationError, UpstreamError) as e:
        logger.warning("VisualSpec thin-scene repair failed: %s", e)
        return None
    return _apply_table_quality(repaired)
