import asyncio
import io
import json
import logging
import re

import google.generativeai as genai
from PIL import Image

from app.config import settings
from app.exceptions import UpstreamError
from app.models.schemas import DiagramRep, DiagramSpec, StudyNotes, VisualSpec
from app.utils.diagram_validation import SUPPORTED_DIAGRAM_TYPES, validate_diagram_spec
from app.utils.diagram_renderer import render_polar_region
from app.utils.latex_clean import latex_to_unicode
from app.utils.svg_safe import sanitize_svg
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
- analogy: use an everyday comparison ONLY if it is genuinely helpful and accurate. Otherwise empty string. Never force one.

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
- analogy: only if genuinely helpful and accurate, otherwise empty string."""  # noqa: E501

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
    wait genuinely helps). Everything else fails fast.

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
                delay = 2 * (attempt + 1)
                logger.warning("Gemini rate limited (attempt %d/%d). Retrying in %ds...", attempt + 1, max_retries + 1, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("Gemini call failed: %s", str(e))
                raise UpstreamError(service="Gemini Vision", detail=str(e))
    logger.error("Gemini call failed after %d retries: %s", max_retries + 1, last_error)
    raise UpstreamError(service="Gemini Vision", detail=str(last_error))


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
    return result


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
            if reason.strip() and reason.strip() not in notes.uncertainties:
                notes.uncertainties.append(reason.strip())

    notes.diagram_spec = None


async def extract_revision_guide(image_bytes: bytes, context_hint: str = "") -> StudyNotes:
    prompt = REVISION_SYSTEM_PROMPT
    if context_hint:
        prompt += "\n\nEXTRACTED CONTEXT (use it to focus, but only teach what the image shows):\n" + context_hint
    result = await _extract_structured(
        prompt, REVISION_REPAIR_PROMPT, image_bytes, StudyNotes
    )
    return result


VISUAL_SPEC_PROMPT = r"""You are SnapNote AI. A student uploaded a messy lecture screenshot, and we already extracted study notes from it. We will now build ONE clean, educational visual that helps the student understand this concept. Your job: decide WHAT that visual should be and HOW it must be rendered, and describe it as a STRUCTURED SPEC — not as pixels, not as an SVG, not as a vague prompt.

THE RENDERING RULE (the most important decision you make):
Choose "render_mode": "deterministic" or "generative" based on INFORMATION PRECISION, not the subject name:
  - "deterministic" — when correctness of text, symbols, equations, relationships, or exact structure is CENTRAL to understanding. Our deterministic renderer then draws a clean SVG with your exact equations and steps. Use this for: formulas, derivations, definitions, labeled flows, diagrams where labels matter, physics/math/engineering content, chemistry mechanisms, anything where garbled text would be harmful.
  - "generative" — ONLY when the value is a rich conceptual illustration where exact text is NOT the main payload: conceptual scenes, biology/cell processes, analogy-like images, mechanisms you can convey with pictures. Never use generative mode for content whose understanding depends on reading exact symbols or labels.

If in doubt, choose "deterministic". A physics topic like "what is angular momentum?" may still benefit from a generative conceptual illustration — decide by precision, not by topic.

THE SCENE PRIMITIVES (deterministic mode):
Inside "deterministic.scene" you describe WHAT must be shown using universal educational primitives. Python decides HOW to draw everything — you NEVER give coordinates, pixel positions, or SVG. You decide only the SEMANTICS: which objects, which vectors, their directions in degrees, and which relationships to draw.
  - "scene_kind": "force_diagram" (vectors/forces/angles around a pivot: torque, lever arms, inclined planes, forces on a body) OR "process_flow" (labeled boxes with arrows: control loops, PID, workflows) OR "plot" (2D graphs: functions y=f(x), projectile trajectories, polar curves — rendered as clean axes + curves).
  - "force_diagram" fields:
      "object": {"kind": "pivot"|"disk"|"point"|"block", "label": "O" (a short name), "caption": "what it is"}
      "vectors": [{"label": "r", "angle_deg": 55, "length": 1.0, "tail": "r" (label of the vector this one starts from; omit/empty to start at the object), "color": ""|"accent"|"red"|"green", "caption": "what this vector represents"}]
        - "angle_deg": direction in degrees, 0 = pointing right, 90 = straight up, 180 = left, 270 = down. Use the direction that matches the real diagram.
        - "length": relative length (0.4 to 1.6). 1.0 is the default.
      "angles": [{"label": "θ", "between": ["r", "F"] (labels of exactly two vectors), "caption": "what the angle is"}]
      "arcs": [{"label": "τ", "around": "O" (object label), "direction": "ccw"|"cw", "caption": "e.g. rotation direction"}]
      "relation": {"expression": "τ = r × F" (exact Unicode equation, NO LaTeX, NO backslash), "caption": "one line meaning"}
  - "process_flow" fields:
      "nodes": [{"label": "Controller"}, {"label": "Plant"}, ...] (2-6 nodes)
      "connectors": [{"source": 0, "target": 1, "label": "u(t)" (optional), "feedback": false}]
        - "feedback": true on the connector that loops the output back to the input (drawn as a curved dashed return arrow).
      "relation": {"expression": "...", "caption": "..."} (optional)
  - "plot" fields:
      "plot": {"x_label": "x", "y_label": "y", "x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5, "show_grid": true,
               "curves": [{"label": "y = x²", "expr": "x**2", "x_min": 0, "x_max": 2, "style": "solid", "color": "accent"},
                          {"label": "trajectory", "points": [[0,0],[1,1.2],[2,1.8]], "style": "solid", "color": ""}]}
        - each curve: EITHER "expr" (safe math in x: x, sin(x), cos(x), sqrt(x), exp(x), log(x), pi) with its own x_min/x_max, OR explicit "points" [[x,y],...]. Prefer expr for accuracy. Colors ""|accent|red|green.
  - "caption": ONE sentence under "WHAT THE VISUAL SHOWS" teaching what the diagram means conceptually (never an inventory of parts). Example: "Torque magnitude depends on the lever arm r and the angle θ between r and F."

  When the screenshot shows a graph/plot/trajectory (parabola, sine wave, polar curve, projectile), prefer "plot" with a safe expr. When it shows vectors/forces, prefer "force_diagram". When it shows boxes/arrows, prefer "process_flow".

GROUNDING RULES:
1. The screenshot is the source of truth. The study notes below are helpful context, but never invent content that conflicts with what the image actually shows.
2. Never invent formulas, quantities, or relationships that are not in the image or the notes. If something is cut off or ambiguous, leave it out of the visual rather than guessing.
3. Keep the visual clean and minimal: white/light background, dark high-contrast text, flat vector style, no photo-realism, no watermark, no decoration.
4. Write labels as short, plain, readable text.
5. In "scene", only emit primitives you are confident about. A torque/force concept should use a force_diagram (pivot + r vector + F vector + θ angle + τ rotation). A control-loop concept should use a process_flow. A graph/trajectory should use a plot. If none of those fit, use "generic" with central_label (2-4 words) and 2-4 callouts (each a short phrase) — this guarantees every concept gets a labeled diagram. Never leave scene empty if you can make a generic.

OUTPUT: ONLY a JSON object with exactly this structure:
{
  "concept": "one phrase naming the concept the visual explains",
  "render_mode": "deterministic | generative",
  "text_required": true,
  "deterministic": {
    "title": "short title (2-6 words)",
    "scene": {
      "scene_kind": "force_diagram | process_flow | plot | generic",
      "caption": "one teaching sentence about what the diagram means",
      "force": { "object": {...}, "vectors": [...], "angles": [...], "arcs": [...], "relation": {...} },
      "flow": { "nodes": [...], "connectors": [...], "relation": {...} },
      "plot": { "x_label": "x", "y_label": "y", "x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5, "show_grid": true, "curves": [{"label": "y=x²", "expr": "x**2", "x_min": 0, "x_max": 2}] },
      "generic": { "central_label": "Main idea", "callouts": ["callout 1", "callout 2"] }
    },
    "equations": [{"expression": "exact formula in Unicode, no LaTeX, no backslash", "meaning": "one line: what each symbol means and what the relationship represents"}],
    "steps": ["ordered steps, each a short phrase"],
    "points": ["key points, each a short phrase"]
  },
  "visual_form": "the chosen visual form, e.g. 'force vector diagram' or 'labeled block diagram' or 'step-by-step flowchart'",
  "key_elements": ["every box/axis/label/marker the visual must contain, verbatim text in quotes"],
  "key_relationships": ["each relationship the visual must show, e.g. 'torque = cross product of r and F'"],
  "must_show": ["the essential things that must be visually present so the student understands"],
  "avoid": ["anything to leave out: unsupported quantities, decorative objects, unrelated details"]
}

FIELD RULES:
- "text_required": true when readable text/symbols are essential to the visual (always true in deterministic mode). false only for a purely conceptual illustration that conveys meaning through pictures alone.
- "deterministic": ALWAYS populate it. Prefer providing a "scene" (a real diagram) when the concept maps to force_diagram, process_flow or plot; the renderer draws it. IMPORTANT — Explain Visually is a VISUAL ARTIFACT, not a second study sheet: when you emit a scene, keep it geometry-first (objects, vectors, angles, arcs, relationships, labels) and do NOT fill equations/steps/points with the same material that already lives in the study notes. Leave equations/steps/points EMPTY when a scene is present; the only equation allowed inside the visual is the scene's own "relation" (e.g. "τ = r × F") plus the one-line caption. Fill equations/steps/points ONLY when there is no scene (the renderer then falls back to a card layout). In generative mode, still include at least title and any one exact relationship you do not want a generative model to garble (the renderer ignores it, but it keeps the exact content available). If nothing exact applies, keep deterministic.title set and leave the lists empty.
- Keep each list concise (2-6 items). Ground every item in the image and notes. If the material genuinely cannot benefit from a visual (e.g. pure prose with no structure worth drawing), set render_mode to "generative", concept to the topic, visual_form to "simple illustration", key_elements to one broad item like "the central idea shown as a simple icon", and avoid anything ungrounded — never invent a diagram the material doesn't support."""


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
        data = _clean_latex_in_dict(_extract_json(raw))
        return VisualSpec(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("VisualSpec parse failed: %s", e)
        raise UpstreamError(
            service="SnapNote AI",
            detail="Could not plan an educational visual for this material.",
        )
