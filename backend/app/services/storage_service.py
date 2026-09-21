import base64
import hashlib
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# data_uri backend is STAGING/TESTING ONLY. It is not a production image
# storage solution: data URIs bloat API responses and database rows and are
# not CDN-cacheable. Production must use "imgbb" (or a future R2/S3 adapter).


def upload_image(image_bytes: bytes, context: dict | None = None) -> str | None:
    """Upload image via the configured storage backend, return public URL.

    Returns None on failure. The "data_uri" backend returns a data: URI and
    never touches the network; it is staging/testing only.
    """
    try:
        from app.config import settings

        backend = settings.IMAGE_STORAGE_BACKEND.strip().lower()
        if backend == "data_uri":
            return _upload_to_data_uri(image_bytes)
        if backend == "imgbb":
            return _upload_to_imgbb(image_bytes, context)
        raise ValueError(f"Unknown IMAGE_STORAGE_BACKEND: {backend!r}")
    except Exception as e:
        logger.warning("Image storage upload failed, returning None: %s", str(e))
        return None


def _detect_mime(image_bytes: bytes) -> str:
    """Detect the true MIME type from magic bytes (never trust the label)."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise ValueError("data_uri backend supports PNG, JPEG, and GIF images only")


def _upload_to_data_uri(image_bytes: bytes) -> str:
    mime = _detect_mime(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _upload_to_imgbb(image_bytes: bytes, context: dict | None) -> str:
    import httpx

    from app.config import settings

    api_key = settings.IMGBB_API_KEY.strip()
    if not api_key:
        raise ValueError("IMGBB_API_KEY not configured")

    title = _slugify(context.get("title", "diagram")) if context else "diagram"
    data = {"key": api_key, "name": f"{title}.jpg"}
    files = {"image": image_bytes}

    resp = httpx.post(
        "https://api.imgbb.com/1/upload",
        data=data,
        files=files,
        timeout=30,
    )
    if resp.status_code >= 400:
        detail = resp.text.strip().replace(api_key, "[REDACTED]")
        detail = " ".join(detail.split())[:500]
        raise ValueError(f"ImgBB HTTP {resp.status_code}: {detail or 'no response body'}")
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("success"):
        raise ValueError(f"ImgBB error: {payload.get('error', payload)}")

    url = payload["data"]["url"]
    logger.info("Image uploaded to ImgBB: %s", url)
    return url


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:40]
