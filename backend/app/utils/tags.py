import json
import logging

from pydantic import ValidationError

from app.models.schemas import ExtractionContext

logger = logging.getLogger(__name__)


def parse_context(raw: str) -> ExtractionContext:
    try:
        data = json.loads(raw)
        return ExtractionContext(**data)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        # Client-supplied context is a hint, never required. A JSON object with
        # wrong field types used to raise ValidationError past this handler and
        # turn a normal extraction into a 500.
        logger.info("Ignoring invalid extraction context: %s", exc)
        return ExtractionContext()


def generate_tags(ctx: ExtractionContext) -> list[str]:
    tags: list[str] = []

    if ctx.week:
        tags.append(ctx.week)
    if "youtube" in ctx.url.lower():
        tags.append("youtube")
    if "iitm" in ctx.url.lower() or "bs" in ctx.title.lower():
        tags.append("iitm-bs")
    if ctx.title:
        words = ctx.title.lower().split()[:5]
        tags.extend(w.strip(",:;.!?") for w in words if len(w) > 3)

    return list(set(tags))[:10]
