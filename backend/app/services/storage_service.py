import base64
import hashlib
import hmac
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)

# data_uri backend is STAGING/TESTING ONLY. It is not a production image
# storage solution: data URIs bloat API responses and database rows and are
# not CDN-cacheable. Production uses "r2". "imgbb" is legacy/disabled, kept
# one migration cycle for rollback only.


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
        if backend == "r2":
            return _upload_to_r2(image_bytes, context)
        if backend == "imgbb":
            return _upload_to_imgbb(image_bytes, context)
        raise ValueError(f"Unknown IMAGE_STORAGE_BACKEND: {backend!r}")
    except Exception as e:
        logger.warning("Image storage upload failed, returning None: %s", str(e))
        return None


def validate_storage_config() -> None:
    """Fail fast at startup when the configured backend cannot work."""
    from app.config import settings

    backend = settings.IMAGE_STORAGE_BACKEND.strip().lower()
    if backend == "r2":
        missing = [
            name
            for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME")
            if not getattr(settings, name, "").strip()
        ]
        if missing:
            raise RuntimeError(f"IMAGE_STORAGE_BACKEND=r2 but missing: {', '.join(missing)}")
        public_url = settings.R2_PUBLIC_URL.strip().rstrip("/")
        if not public_url.startswith("https://") or "xxxxx" in public_url:
            raise RuntimeError("IMAGE_STORAGE_BACKEND=r2 but R2_PUBLIC_URL is not a real public base URL")
    elif backend not in ("data_uri", "imgbb"):
        raise RuntimeError(f"Unknown IMAGE_STORAGE_BACKEND: {backend!r}")


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


def _r2_object_key(image_bytes: bytes, context: dict | None) -> str:
    """Content-hash key: identical bytes always map to the same object, so a
    re-upload can never overwrite unrelated content. The slug prefix keeps
    keys debuggable (visuals/ vs diagrams/)."""
    title = _slugify(context.get("title", "diagram")) if context else "diagram"
    prefix = "visuals" if "visual" in title else "diagrams"
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif"}[_detect_mime(image_bytes)]
    return f"{prefix}/{digest}-{title}.{ext}"


def _upload_to_r2(image_bytes: bytes, context: dict | None) -> str:
    """S3-compatible object upload (Cloudflare R2 by default, or the provider
    in S3_ENDPOINT_URL). Single PUT, content-hash key, public URL out."""
    import httpx

    from app.config import settings

    account_id = settings.R2_ACCOUNT_ID.strip()
    access_key = settings.R2_ACCESS_KEY_ID.strip()
    secret_key = settings.R2_SECRET_ACCESS_KEY.strip()
    bucket = settings.R2_BUCKET_NAME.strip()
    public_base = settings.R2_PUBLIC_URL.strip().rstrip("/")
    endpoint_override = settings.S3_ENDPOINT_URL.strip().rstrip("/")
    region = settings.S3_REGION.strip() or "auto"
    if not (access_key and secret_key and bucket):
        raise ValueError("Object storage is not configured")
    if not public_base.startswith("https://") or "xxxxx" in public_base:
        raise ValueError("R2_PUBLIC_URL is not a real public base URL")

    mime = _detect_mime(image_bytes)
    key = _r2_object_key(image_bytes, context)
    if endpoint_override:
        # Path-style (Supabase and most S3 providers):
        # {endpoint}/{bucket}/{key}
        from urllib.parse import urlsplit

        parts = urlsplit(endpoint_override)
        host = parts.netloc
        base_path = parts.path.rstrip("/")
        path = f"{base_path}/{bucket}/{quote(key, safe='/')}"
        url = f"{parts.scheme}://{host}{path}"
    else:
        # Virtual-hosted style (Cloudflare R2):
        # {account}.r2.cloudflarestorage.com/{bucket}/{key}
        if not account_id:
            raise ValueError("Object storage is not configured")
        host = f"{account_id}.r2.cloudflarestorage.com"
        path = f"/{bucket}/{quote(key, safe='/')}"
        url = f"https://{host}{path}"

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(image_bytes).hexdigest()
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        f"content-type:{mime}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    canonical_request = (
        f"PUT\n{path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )
    signing_key = ("AWS4" + secret_key).encode("utf-8")
    for part in (date_stamp, region, "s3", "aws4_request"):
        signing_key = hmac.new(signing_key, part.encode("utf-8"), hashlib.sha256).digest()
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    resp = httpx.put(
        url,
        content=image_bytes,
        headers={
            "Content-Type": mime,
            "Content-Length": str(len(image_bytes)),
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "Authorization": (
                f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            ),
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        detail = " ".join(resp.text.strip().split())[:300]
        raise ValueError(f"Object storage HTTP {resp.status_code}: {detail or 'no response body'}")
    resp.raise_for_status()

    public_url = f"{public_base}/{key}"
    logger.info("Image uploaded to R2: %s", public_url)
    return public_url


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
