import json
from app.models.schemas import ExtractionContext


def parse_context(raw: str) -> ExtractionContext:
    try:
        data = json.loads(raw)
        return ExtractionContext(**data)
    except (json.JSONDecodeError, TypeError):
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
