import hashlib
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_image(image_bytes: bytes, context: dict | None = None) -> str | None:
    """Upload image to ImgBB, return public URL. Returns None on failure."""
    try:
        return _upload_to_imgbb(image_bytes, context)
    except Exception as e:
        logger.warning("ImgBB upload failed, returning None: %s", str(e))
        return None


def _upload_to_imgbb(image_bytes: bytes, context: dict | None) -> str:
    import httpx

    from app.config import settings

    api_key = settings.IMGBB_API_KEY
    if not api_key:
        raise ValueError("IMGBB_API_KEY not configured")

    title = _slugify(context.get("title", "diagram")) if context else "diagram"
    data = {"key": api_key, "name": f"{title}.png"}
    files = {"image": image_bytes}

    resp = httpx.post(
        "https://api.imgbb.com/1/upload",
        data=data,
        files=files,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("success"):
        raise ValueError(f"ImgBB error: {payload.get('error', payload)}")

    url = payload["data"]["url"]
    logger.info("Image uploaded to ImgBB: %s", url)
    return url


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:40]
