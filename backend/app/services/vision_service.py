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

TEXT_SYSTEM_PROMPT = """You are a precise note extraction assistant. Given a screenshot of study content:
1. Extract ALL text exactly as written (no paraphrasing)
2. For worked solutions: write question then each step sequentially, preserving derivation order
3. Format tables using Markdown table syntax
4. Format lists as Markdown bullets
5. Use LaTeX for mathematical notation only — not for plain words, names, or labels
6. Output ONLY the formatted markdown, no explanations, no greetings
7. Never describe the image — just transcribe
8. Transcribe only what is visibly written. Do not solve, complete, continue, or extend any problem beyond what is shown. If a derivation is cut off, state that explicitly rather than filling in missing steps."""

DIAGRAM_SYSTEM_PROMPT = """You transcribe screenshots into study notes. Default to direct, sequential transcription of all visible text in the order it appears — write it the way a student would jot notes. Determine the content type, then output ONLY the format specified below — nothing else.

**Case A — Worked solution / derivation / sequential reasoning** (math, physics, proof, step-by-step):
Format:
## Question
[the problem statement]

## Solution
1. [step 1]
2. [step 2]
...

Use LaTeX for mathematical notation only — not for plain words, names, or labels. No headings like "Diagram Type", "Description", or "Labels & Text". No meta-commentary. Just the solution.

**Case B — Visual diagram** (flowchart, circuit, graph, schematic, mind map):
Format:
**Type:** [one-line diagram type]

[2-3 sentence plain-text description of what the diagram shows]

For the specific portion of the image that has a genuine drawn arrow or flow, express it as a short plain-sentence caption (e.g. "Electrons transfer from A to B"). Do NOT use "→" between labels unless the image itself visually draws that arrow. Do NOT invent relationships between headings, titles, or text blocks that are just adjacent text.

**Rules for both cases:**
- No "The image displays" or "This screenshot shows"
- No SVG, Mermaid, or code
- No greetings or sign-offs
- Start directly with the content
- Transcribe only what is visibly written. Do not solve, complete, continue, or extend any problem beyond what is shown. If a derivation is cut off, state that explicitly rather than filling in missing steps."""

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
