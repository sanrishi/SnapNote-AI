import hashlib
import time
import logging

logger = logging.getLogger(__name__)


def upload_image(image_bytes: bytes, context: dict | None = None) -> str | None:
    """Upload image to R2, return public URL. Returns None if unconfigured."""
    try:
        return _do_upload(image_bytes, context)
    except Exception as e:
        logger.warning("R2 upload failed, returning None: %s", str(e))
        return None


def _do_upload(image_bytes: bytes, context: dict | None) -> str:
    import boto3
    from botocore.config import Config

    from app.config import settings

    account_id = settings.R2_ACCOUNT_ID
    if not account_id:
        raise ValueError("R2_ACCOUNT_ID not configured")

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    timestamp = int(time.time())
    title_slug = _slugify(context.get("title", "diagram")) if context else "diagram"
    key = f"diagrams/{timestamp}_{content_hash}_{title_slug}.png"

    client.put_object(
        Bucket=settings.R2_BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType="image/png",
    )

    return f"{settings.R2_PUBLIC_URL}/{key}"


def _placeholder_url(image_bytes: bytes) -> str:
    content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    return f"https://storage.snapnote.ai/placeholder/{content_hash}.png"


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:40]
