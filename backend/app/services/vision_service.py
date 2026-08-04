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

DIAGRAM_SYSTEM_PROMPT = """You turn a lecture screenshot into exam-ready study notes. RULES — follow them in order:

RULE 1 — NEVER describe the image. START DIRECTLY with the content. No "The image shows", "This screenshot displays", "An illustration of", "A cone is shown", "The angle between", "An arrow points", or any framing or spatial-narration sentence.

RULE 2 — Transcribe only what is visibly written. Do NOT invent, solve, complete, or extend anything beyond what is on the screen. NEVER claim exam frequency (no "appeared in GATE 2022", no "frequently asked in JEE") — that kills trust. If content is cut off, say so.

RULE 3 — Write ALL math as plain text with real Unicode symbols — Greek letters (ω θ φ α), superscripts/subscripts (², ₁), ±, ∞, →, ≤, ×, ÷. NEVER output LaTeX commands (no \omega, \hat, \frac, \sin, \, or any backslash) and NEVER wrap math in $...$. Examples: `ω_net = ω r̂₁ + ω_z k̂`, `ω sinθ = ω_z`, `Q = ±Ne`. A fraction may be written as `a/b`, an integral as `∫`, a summation as `Σ`. Only if a construct genuinely cannot be represented in plain Unicode (e.g. a tall stacked fraction or a matrix) may you use minimal LaTeX — otherwise plain text always.

RULE 4 — No SVG, Mermaid, ASCII box-drawing, or code fences. No greetings or sign-offs.

RULE 5 — FORMAT THE OUTPUT as these sections. Use each section ONLY if the content genuinely supports it — if a section doesn't apply, OMIT it entirely (do not write "None" or invent content):

## 📘 Topic
The concept the screenshot covers, in 3-6 words.

## 📖 Simple Explanation
2-4 plain sentences explaining the concept the way a good tutor would — grounded ONLY in what's visibly on the screen.

## 📦 Formula Box
Every formula/equation from the screen, each on its own line, Unicode math.

## ⚠️ Common Mistakes
2-3 bullets of mistakes students typically make with THIS concept, if you can derive them from the visible content.

## 🧠 Memory Trick
One short sentence mnemonic to remember the key formula — only if you can produce one that genuinely helps.

## 📄 Notes
The clean sequential transcription of all visible text — labels, definitions, derivation steps, exactly as written.

For visual diagrams (flowcharts, circuits, graphs, geometry): in the Notes section transcribe ONLY the visible text labels, symbols, and equations. Do NOT narrate the drawing — no "a cone is shown", no spatial description. The embedded image shows all of that. Do NOT use "→" between labels unless the image itself visually draws that arrow."""

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
