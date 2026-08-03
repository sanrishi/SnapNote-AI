import asyncio
import io
import logging
import time

import google.generativeai as genai
from PIL import Image

from app.config import settings
from app.exceptions import UpstreamError
from app.models.schemas import DiagramResult

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-flash-lite-latest")

TEXT_SYSTEM_PROMPT = """You extract study notes from a screenshot. RULES — follow them in order:

RULE 1 — NEVER describe the image. START DIRECTLY with the transcribed content. No "The image shows", "This screenshot displays", "An illustration of", or any framing sentence. Just the content.

RULE 2 — Extract ALL text exactly as written (no paraphrasing).
RULE 3 — For simple content (a single formula, a short definition, one example): write as 1-3 plain natural sentences — no headers, no bullet list, no LaTeX unless the math genuinely can't be written in plain text. Reserve structure (headers, bullets) for content that's inherently a list: multiple distinct items, a comparison, or a multi-step derivation.
RULE 4 — For worked solutions: write question then each step sequentially, preserving derivation order.
RULE 5 — Format tables using Markdown table syntax.
RULE 6 — NEVER output LaTeX commands (no \omega, \hat, \frac, \sin, \, or any backslash) and NEVER wrap math in $...$. Write ALL math as plain text using real Unicode symbols — Greek letters (ω θ φ α), superscripts/subscripts where they exist (², ₁, etc.), ±, ∞, →, ≤, ×, ÷. Example: `ω_net = ω r̂₁ + ω_z k̂` and `ω sinθ = ω_z`. A fraction may be written as `a/b`, an integral as `∫`, a summation as `Σ`. Only if a construct genuinely cannot be represented in plain Unicode (e.g. a tall stacked fraction or a matrix) may you use minimal LaTeX — otherwise plain text always.
RULE 7 — Output ONLY the formatted markdown, no explanations, no greetings.
RULE 8 — Transcribe only what is visibly written. Do not solve, complete, continue, or extend any problem beyond what is shown. If a derivation is cut off, state that explicitly rather than filling in missing steps."""

DIAGRAM_SYSTEM_PROMPT = """You extract study notes from a screenshot. RULES — follow them in order:

RULE 1 — NEVER describe the image. START DIRECTLY with the transcribed content. No "The image shows", "This screenshot displays", "An illustration of", or any framing sentence. Just the content. No diagram type label, no description header — just the notes.

RULE 2 — Default to direct, sequential transcription of all visible text in the order it appears. Write it the way a student would jot notes.

**Case A — Worked solution / derivation / sequential reasoning** (math, physics, proof, step-by-step):
Format:
## Question
[the problem statement]

## Solution
1. [step 1]
2. [step 2]
...

For simple content (a single formula, short definition, one example): write as 1-3 plain sentences — no headers, no bullets. NEVER output LaTeX commands (no \omega, \hat, \frac, \sin, \, or any backslash) and NEVER wrap math in $...$. Write ALL math as plain text with real Unicode symbols — Greek letters (ω θ φ α), superscripts/subscripts (², ₁), ±, ∞, →, ≤, ×, ÷. Example: `ω_net = ω r̂₁ + ω_z k̂`, `ω sinθ = ω_z`, `Q = ±Ne`. A fraction may be written as `a/b`, an integral as `∫`, a summation as `Σ`. Only if a construct genuinely cannot be represented in plain Unicode (e.g. a tall stacked fraction or a matrix) may you use minimal LaTeX — otherwise plain text always.

**Case B — Visual diagram** (flowchart, circuit, graph, schematic, mind map):
Transcribe the visible labels, arrows, and relationships as plain text. If the image has a genuine drawn arrow or flow, describe it as a short plain-sentence (e.g. "Electrons transfer from A to B"). Do NOT use "→" between labels unless the image itself visually draws that arrow. Do NOT invent relationships between headings, titles, or text blocks that are just adjacent text.

RULE 3 — No SVG, Mermaid, ASCII box-drawing, or code fences.
RULE 4 — No greetings or sign-offs.
RULE 5 — Transcribe only what is visibly written. Do not solve, complete, continue, or extend any problem beyond what is shown. If a derivation is cut off, state that explicitly rather than filling in missing steps."""

async def _call_gemini(prompt: str, image_bytes: bytes, max_retries: int = 3) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await model.generate_content_async([prompt, img])
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


async def extract_diagram(image_bytes: bytes) -> DiagramResult:
    markdown = await _call_gemini(DIAGRAM_SYSTEM_PROMPT, image_bytes)
    return DiagramResult(markdown=markdown)
