import base64
import logging
from openai import AsyncOpenAI

from app.config import settings
from app.models.schemas import DiagramResult
from app.exceptions import UpstreamError

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

TEXT_SYSTEM_PROMPT = """You are a precise note extraction assistant. Given a screenshot from a lecture video:
1. Extract ALL text exactly as written (no paraphrasing)
2. Format tables using Markdown table syntax
3. Format lists as Markdown bullets
4. Output ONLY the formatted markdown, no explanations, no greetings

Rules:
- Preserve mathematical notation using LaTeX where appropriate
- Keep bullet points and numbering intact
- If the content has columns/rows structure, output as a Markdown table
- Do NOT add any text outside the markdown content"""

DIAGRAM_SYSTEM_PROMPT = """You are a diagram extraction assistant. Given a screenshot containing a diagram/graph:
1. Extract all text labels and legends as a markdown list
2. Identify the diagram type (flowchart, graph, chart, schematic, etc.)
3. Provide a brief description of the diagram structure
4. Do NOT generate SVG or Mermaid code
5. Do NOT add greetings or explanations

Output format:
## Diagram Type: [type]
### Description
[2-3 sentence description of what the diagram shows]

### Labels & Text
- [label 1]
- [label 2]
..."""


async def extract_text_with_llm(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TEXT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=2048,
            temperature=0.1,
        )
    except Exception as e:
        logger.error("Vision LLM call failed: %s", str(e))
        raise UpstreamError(service="OpenAI Vision", detail=str(e))

    return resp.choices[0].message.content.strip()


async def extract_diagram(image_bytes: bytes) -> DiagramResult:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DIAGRAM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=1024,
            temperature=0.1,
        )
    except Exception as e:
        logger.error("Vision LLM diagram call failed: %s", str(e))
        raise UpstreamError(service="OpenAI Vision (diagram)", detail=str(e))

    content = resp.choices[0].message.content.strip()
    return DiagramResult(markdown=content)
