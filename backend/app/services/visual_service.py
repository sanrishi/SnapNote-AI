import asyncio
import io
import logging

import httpx
from PIL import Image

from app.config import settings
from app.models.schemas import VisualRenderMode, VisualSpec
from app.services import ocr_service
from app.utils.visual_renderer import render_deterministic_visual

logger = logging.getLogger(__name__)

_POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

_STRONG_WHITE_HINT = (
    "CRITICAL: the ENTIRE background must be pure white (#FFFFFF). "
    "Only the diagram's lines, boxes, arrows and text may be dark. "
    "No shadows, no gradients, no colored background, no watermark."
)

# OCR legibility gate thresholds (generative mode only, when text_required).
_MIN_READABLE_TOKENS = 2
_MIN_TOKEN_CONFIDENCE = 0.35


def _build_render_prompt(spec: VisualSpec, retry: bool) -> str:
    """Convert the grounded VisualSpec into a single image-model prompt."""
    lines = [
        "Draw ONE clean educational visual that helps a student understand this concept.",
        "Style: flat vector diagram, white/light background, dark high-contrast readable text, minimal, no photo-realism, no watermark, no decoration.",
    ]
    if retry:
        lines.append(_STRONG_WHITE_HINT)
    if spec.concept:
        lines.append(f"CONCEPT: {spec.concept}")
    if spec.visual_form:
        lines.append(f"VISUAL FORM: {spec.visual_form}")
    if spec.key_elements:
        lines.append("KEY ELEMENTS (each must appear, with its label text): " + "; ".join(spec.key_elements))
    if spec.key_relationships:
        lines.append("KEY RELATIONSHIPS TO SHOW: " + "; ".join(spec.key_relationships))
    if spec.must_show:
        lines.append("MUST SHOW: " + "; ".join(spec.must_show))
    if spec.avoid:
        lines.append("AVOID: " + "; ".join(spec.avoid))
    return " ".join(lines)


def _quality_pass(png: bytes) -> bool:
    """Lightweight quality gate — never blind flattening to white.

    Checks: valid decodable image, sane size, background sufficiently light,
    content not mostly dark/empty. Colored arrows and shading are preserved;
    we only reject renders that would look broken or unreadable to a student.
    This is the FIRST gate; generative text-bearing renders are additionally
    checked by the OCR legibility gate (see _legibility_pass).
    """
    if not png or len(png) < 500:
        return False
    try:
        img = Image.open(io.BytesIO(png))
        img.load()
        w, h = img.size
        if w < 256 or h < 256:
            return False
        if img.mode == "RGBA":
            img = img.convert("RGB")
        small = img.resize((96, 96))
        pixels = list(small.getdata())
    except Exception as e:
        logger.warning("Quality gate: image decode failed: %s", e)
        return False
    n = len(pixels)
    light = sum(1 for r, g, b in pixels if min(r, g, b) > 200) / n
    dark = sum(1 for r, g, b in pixels if max(r, g, b) < 60) / n
    ok = light >= 0.20 and dark <= 0.80
    logger.info("Quality gate: light=%.2f dark=%.2f -> %s", light, dark, "pass" if ok else "fail")
    return ok


def _legibility_pass(png: bytes) -> bool:
    """OCR legibility gate — only used in generative mode when text_required.

    Never run blindly: the VisualSpec itself declares whether readable text is
    essential. When it is, we reject a render with essentially no readable
    tokens (the exact failure mode observed with Pollinations: clean-looking
    diagrams with garbled labels). When OCR is unavailable we cannot verify, so
    we accept rather than break the feature — the gate is a safety net, not a
    hard dependency.
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("OCR legibility gate unavailable (%s); accepting render", e)
        return True
    if not ocr_service.ocr_available():
        logger.warning("OCR reader not available; skipping legibility gate")
        return True
    try:
        arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    except Exception as e:
        logger.warning("Legibility gate: image decode failed: %s", e)
        return False
    try:
        tokens = [(text.strip(), conf) for _, text, conf in ocr_service.read_raw(arr) if text.strip()]
    except Exception as e:
        logger.warning("Legibility gate: OCR failed: %s", e)
        return True
    if len(tokens) < _MIN_READABLE_TOKENS:
        logger.info("Legibility gate: only %d readable token(s) -> fail", len(tokens))
        return False
    avg_conf = sum(conf for _, conf in tokens) / len(tokens)
    ok = avg_conf >= _MIN_TOKEN_CONFIDENCE
    logger.info("Legibility gate: %d tokens avg_conf=%.2f -> %s", len(tokens), avg_conf, "pass" if ok else "fail")
    return ok


async def _render_once(prompt: str) -> bytes | None:
    """One Pollinations image call via GET (POST on the anonymous tier returns a
    fixed cached image regardless of prompt, so we must use GET). Returns PNG/JPEG
    bytes or None."""
    import urllib.parse

    url = _POLLINATIONS_BASE + urllib.parse.quote(prompt)
    params = {
        "width": 768,
        "height": 768,
        "nologo": "true",
        "model": settings.POLLINATIONS_MODEL,
    }
    headers: dict[str, str] = {}
    if settings.POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {settings.POLLINATIONS_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=settings.POLLINATIONS_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            logger.warning("Pollinations returned %s: %s", resp.status_code, resp.text[:200])
            return None
        return resp.content
    except Exception as e:
        logger.warning("Pollinations call failed: %s", str(e)[:200])
        return None


async def _render_generative(spec: VisualSpec) -> bytes | None:
    """Generative branch: Pollinations render, quality gate, then a conditional
    OCR legibility gate (only when spec.text_required), then one hidden retry.

    First attempt uses the spec as-is. If a gate rejects it, one internal retry
    re-renders with an emphatic pure-white instruction. If that also fails,
    return None so the route reports an honest unavailable state. Never exposed
    as a student-facing regenerate control.
    """
    attempts = [
        _build_render_prompt(spec, retry=False),
        _build_render_prompt(spec, retry=True),
    ]
    for attempt, prompt in enumerate(attempts, start=1):
        png = await _render_once(prompt)
        if png is not None and _quality_pass(png):
            if spec.text_required and not await asyncio.to_thread(_legibility_pass, png):
                logger.info("Visual rejected by OCR legibility gate on attempt %d; one hidden retry", attempt)
                png = None
            else:
                logger.info("Visual generated on attempt %d", attempt)
                return png
        if attempt == 1:
            logger.info("Visual failed quality gate on attempt 1; one hidden retry")
            await asyncio.sleep(2.0)
    logger.warning("Visual generation failed after 2 attempts")
    return None


async def generate_visual(spec: VisualSpec) -> tuple[str, str | bytes] | None:
    """Hybrid dispatcher: render the educational visual for a VisualSpec.

    - deterministic: exact text/symbols are the payload -> code renders a clean
      sanitized SVG (visual_renderer). Returns ("svg", svg_string).
    - generative: exact typography is NOT the payload -> Pollinations draws a
      conceptual illustration, gated by brightness + conditional OCR legibility.
      Returns ("png", image_bytes).

    Returns None when the chosen branch produced nothing usable, so the route
    reports an honest unavailable state. The same spec always renders the same
    deterministic SVG (no randomness).
    """
    if spec.render_mode == VisualRenderMode.DETERMINISTIC:
        svg = await asyncio.to_thread(render_deterministic_visual, spec.deterministic)
        if not svg:
            logger.warning("Deterministic visual: empty SVG (nothing meaningful to draw)")
            return None
        return ("svg", svg)

    png = await _render_generative(spec)
    if png is None:
        return None
    return ("png", png)