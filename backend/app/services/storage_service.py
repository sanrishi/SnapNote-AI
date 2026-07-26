import hashlib
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def upload_image(image_bytes: bytes, context: dict | None = None) -> str | None:
    """Save image locally and return a relative URL path."""
    from app.config import settings

    content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    timestamp = int(time.time())
    title_slug = _slugify(context.get("title", "diagram")) if context else "diagram"
    filename = f"{timestamp}_{content_hash}_{title_slug}.png"

    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(image_bytes)

    logger.info("Image saved locally: %s", filepath)
    return f"/uploads/{filename}"


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:40]
