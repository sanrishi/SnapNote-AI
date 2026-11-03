import logging
import os
import re
import secrets
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DEV_JWT_SECRET = "dev-secret-change-in-prod-please-set-JWT_SECRET-env"

# Placeholders that look long enough but are still public knowledge (the
# committed .env.example value is one of them). A production deployment must
# never sign sessions with any of these.
_WEAK_SECRET_MARKERS = (
    "change-this",
    "change-me",
    "changeme",
    "please-set",
    "your-secret",
    "dev-secret",
    "secret-key",
)


def is_weak_jwt_secret(secret: str, min_length: int = 32) -> bool:
    """True when a signing secret is a known/default placeholder or too short."""
    value = (secret or "").strip()
    if len(value) < min_length:
        return True
    lowered = value.lower()
    return any(marker in lowered for marker in _WEAK_SECRET_MARKERS)


_RENDER_DATA_DIR = Path("/data")
_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")


def _decode_mountinfo_path(value: str) -> str:
    """Decode the octal escapes the kernel uses in /proc/self/mountinfo."""
    return _OCTAL_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def is_mount_point(path: Path) -> bool:
    """True when ``path`` is a mount point, including same-device bind mounts.

    ``os.path.ismount`` compares the device with the parent, so a bind mount of
    the same filesystem reads as an ordinary directory. The kernel's mount
    table is authoritative, so consult it as well. Non-Linux hosts simply have
    no mountinfo file and fall back to ``os.path.ismount``.
    """
    try:
        if os.path.ismount(path):
            return True
    except OSError:
        pass
    target = os.path.normpath(str(path))
    try:
        with _MOUNTINFO_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 5 and _decode_mountinfo_path(fields[4]) == target:
                    return True
    except OSError:
        return False
    return False


def is_writable_mount(path: Path) -> bool:
    """True when ``path`` is a real mount point the process can write to."""
    try:
        return path.is_dir() and is_mount_point(path) and os.access(path, os.W_OK)
    except OSError:
        return False


def _on_render() -> bool:
    """True inside a Render deployment (running locally never sets these)."""
    return bool(os.environ.get("RENDER") or os.environ.get("RENDER_GIT_COMMIT"))


def _render_disk_available() -> bool:
    """True when running on Render with a writable mounted /data disk.

    The Dockerfile creates /data in the image, so a plain directory check is
    not enough: only a real mount point survives a redeploy. Render exposes
    RENDER_GIT_COMMIT (already used by /health); no local/test process has it.
    """
    if not _on_render():
        return False
    return is_writable_mount(_RENDER_DATA_DIR)


def _load_or_create_persisted_secret(db_path: str) -> tuple[str | None, str]:
    """Keep a generated signing secret across restarts on a mounted disk.

    Production mounts a durable disk (CREDITS_DB_PATH=/data/credits.db), so a
    secret generated because the operator forgot JWT_SECRET can live beside
    the database instead of invalidating every session on each cold start.
    Relative paths (local development, tests) deliberately never write files
    into the working tree.

    Returns ``(secret, reason)``. ``reason`` is a short, non-sensitive enum
    used by /health to explain why persistence did not happen.
    """
    db = Path(db_path)
    if not db.is_absolute():
        return None, "relative_path"
    path = db.parent / ".jwt_secret"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("Could not read persisted JWT secret %s: %s", path, exc)
            return None, "read_failed"
        if is_weak_jwt_secret(existing):
            return None, "weak_existing_file"
        return existing, ""
    except OSError as exc:
        logger.error("Could not create persisted JWT secret %s: %s", path, exc)
        return None, "create_failed"

    value = secrets.token_urlsafe(48)
    encoded = value.encode("utf-8")
    try:
        written = os.write(fd, encoded)
        if written != len(encoded):
            raise OSError(f"short write ({written}/{len(encoded)} bytes)")
    except OSError as exc:
        logger.error("Could not write persisted JWT secret %s: %s", path, exc)
        try:
            path.unlink()
        except OSError:
            pass
        return None, "write_failed"
    finally:
        os.close(fd)
    return value, ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    APP_NAME: str = "SnapNote AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    FIREBASE_CREDENTIALS_PATH: str = "firebase-credentials.json"
    GEMINI_API_KEY: str = ""
    IMGBB_API_KEY: str = ""

    # Storage backend for upload_image(): "r2" (default, production),
    # "data_uri" (staging/testing only — returns a data: URI, no network), or
    # "imgbb" (legacy/disabled — kept one migration cycle for rollback only).
    IMAGE_STORAGE_BACKEND: str = "r2"

    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "snapnote-diagrams"
    R2_PUBLIC_URL: str = "https://pub-xxxxx.r2.dev"

    # S3-compatible override. Empty = Cloudflare R2 (virtual-hosted style,
    # region "auto"). Set to a provider endpoint (e.g. Supabase
    # https://<ref>.storage.supabase.co/storage/v1/s3) for path-style
    # requests with S3_REGION set to the provider region.
    S3_ENDPOINT_URL: str = ""
    S3_REGION: str = "auto"

    FREE_CREDITS_MONTHLY: int = 50
    TEXT_CREDIT_COST: int = 1
    REVISION_CREDIT_COST: int = 1
    DIAGRAM_CREDIT_COST: int = 5
    ANONYMOUS_FREE_USES: int = 50  # overridden to 1 in production via render.yaml (prevents reload abuse)

    # Pollinations image generation (Explain Visually). Empty key = anonymous
    # tier (1 req / 15s, slower); a free registered key lifts it to 1 req / 5s.
    POLLINATIONS_API_KEY: str = ""
    POLLINATIONS_MODEL: str = "sana"
    POLLINATIONS_TIMEOUT_SECONDS: float = 25.0

    DAILY_REQ_LIMIT: int = 50
    RATE_LIMIT_PER_MIN: int = 10
    MAX_IMAGE_SIZE_MB: int = 10
    CREDITS_DB_PATH: str = "credits.db"
    # How the credits DB path was chosen: "environment" (explicit config or
    # default) or "render_default" (relative default upgraded to the mounted
    # /data disk). Exposed on /health so an ephemeral DB is visible.
    CREDITS_DB_SOURCE: str = "environment"
    # Comma-separated browser origins allowed to call the API. "*" (default)
    # allows any origin WITHOUT credentials — the API authenticates with a
    # bearer header, never cookies. Set explicit origins if credentialed
    # browser requests are ever needed.
    CORS_ALLOW_ORIGINS: str = "*"

    # OCR working-resolution cap: preprocess() downscales the long edge to
    # this BEFORE denoise/threshold so per-request transients (which scale
    # with pixels) stay bounded on the 512 MB instance. Slide fixtures and
    # 1080p screenshots (long edge <= 2048) are never touched.
    OCR_MAX_LONG_EDGE: int = 2048
    # Decode-time pixel budget. MAX_IMAGE_SIZE_MB bounds the compressed upload,
    # not the decoded bitmap: a 5 MB PNG can claim 30000x30000 pixels and
    # allocate gigabytes inside Pillow/OpenCV before any resize happens.
    # 16 MP accepts standard 12 MP phone photos and refuses decompression bombs
    # with 422 instead of risking an OOM kill on the 512 MB instance. Both
    # extraction paths downscale to <= 2048 px, so larger sources add memory
    # cost without adding OCR/vision detail.
    MAX_SOURCE_MEGAPIXELS: int = 16

    # Tuned for 15-20s student SLA: smaller JPEG + tighter Gemini deadlines so
    # a slow upstream fails fast (502 in 22s) instead of 504 in 60s. See
    # extract.py / vision_service.py which both respect these.
    MAX_VISION_LONG_EDGE: int = 1280
    VISION_JPEG_QUALITY: int = 75
    GEMINI_CALL_TIMEOUT_SECONDS: float = 22.0
    # Repair-leg sub-budget: reconciliation is a tiny operation (one pair of
    # expressions), so it gets a tighter deadline than full understanding.
    # Expiry refuses honestly; the timeout is never raised to force a pass.
    GEMINI_REPAIR_TIMEOUT_SECONDS: float = 12.0
    # Capacity fallback: a different model usually sits in a different
    # serving pool, so one fallback attempt on 503/504 (capacity errors)
    # can succeed during a partial outage. Empty disables the fallback.
    # Never used for 429/quota (account-wide) or auth errors.
    GEMINI_FALLBACK_MODEL: str = "gemini-3-flash-preview"
    DIAGRAM_TIMEOUT_SECONDS: float = 28.0

    # "semantic" = Gemini outputs a structured DiagramSpec, Python renders it
    # deterministically. "legacy" = Gemini writes the SVG directly (old path).
    DIAGRAM_RENDERER_MODE: str = "semantic"

    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    # Mock payments are a local-development convenience ONLY. In production a
    # missing gateway key must disable checkout, never mint free credits
    # (MOCK_MODE previously granted credits for any order id + fake signature).
    PAYMENTS_MOCK: bool = False

    JWT_SECRET: str = DEV_JWT_SECRET
    JWT_SECRET_MIN_LENGTH: int = 32
    # Set by the validator below when no durable secret was configured.
    # Exposed on /health (never the value) so an operator can see the state.
    JWT_SECRET_GENERATED: bool = False
    JWT_SECRET_PERSISTED: bool = False
    JWT_SECRET_STORAGE: str = "environment"
    JWT_SECRET_STORAGE_REASON: str = ""
    JWT_EXPIRES_HOURS: int = 168

    GOOGLE_CLIENT_ID: str = ""  # for Google Identity Services ID-token verification
    # Comma-separated OAuth client IDs accepted for Chrome-extension access
    # tokens. Falls back to GOOGLE_CLIENT_ID. When both are empty there is no
    # audience to compare against, so any genuine verified Google token works.
    GOOGLE_ALLOWED_CLIENT_IDS: str = ""

    # Explain Visually v3 composition boundary (visual_lesson.render_visual_lesson).
    # false = legacy generate_visual path, byte-for-byte. true = v3 composition
    # path for deterministic specs (generative specs always use legacy path).
    EXPLAIN_VISUALLY_V3: bool = False

    CREDIT_PACKS: dict = {
        "starter": {"credits": 50, "price_paise": 4900},
        "popular": {"credits": 120, "price_paise": 9900},
        "pro": {"credits": 300, "price_paise": 19900},
        "unlimited": {"credits": 1000, "price_paise": 49900},
    }

    @model_validator(mode="after")
    def _guard_jwt_secret(self) -> "Settings":
        """Never sign production sessions with a public placeholder secret.

        The repo commits its development default, so a deployment that forgets
        to set JWT_SECRET was verifiably accepting forged tokens (confirmed on
        the live API: a token signed with the default decoded successfully).
        Outside DEBUG, a weak/missing secret is replaced so forgery is
        impossible. When the credits DB lives on a mounted disk the generated
        secret is persisted next to it, so sessions still survive restarts;
        otherwise it is per-process and /health reports it.
        """
        if not self.DEBUG and self.CREDITS_DB_PATH == "credits.db" and _render_disk_available():
            # A Render service that lost its blueprint env vars would write an
            # ephemeral /app/credits.db and silently lose users/credits on every
            # deploy. The mounted disk is the durable location; upgrade only
            # the untouched default so explicit relative paths stay explicit.
            self.CREDITS_DB_PATH = str(_RENDER_DATA_DIR / "credits.db")
            self.CREDITS_DB_SOURCE = "render_default"
            logger.warning(
                "CREDITS_DB_PATH was the relative default on a mounted-disk "
                "deployment - using %s",
                self.CREDITS_DB_PATH,
            )

        if not self.DEBUG and _on_render():
            db_parent = Path(self.CREDITS_DB_PATH).expanduser().parent
            if not is_mount_point(db_parent):
                logger.critical(
                    "Credits DB path %r is not on a mounted disk while running "
                    "on Render - users, credits, payment orders and sessions "
                    "will be lost on every deploy. Attach a persistent disk at "
                    "/data and set CREDITS_DB_PATH=/data/credits.db.",
                    self.CREDITS_DB_PATH,
                )

        if is_weak_jwt_secret(self.JWT_SECRET, self.JWT_SECRET_MIN_LENGTH):
            if self.DEBUG:
                # Local workflow keeps the documented default.
                self.JWT_SECRET_STORAGE = "development"
                return self
            persisted, reason = _load_or_create_persisted_secret(self.CREDITS_DB_PATH)
            self.JWT_SECRET = persisted or secrets.token_urlsafe(48)
            self.JWT_SECRET_GENERATED = True
            self.JWT_SECRET_PERSISTED = persisted is not None
            self.JWT_SECRET_STORAGE = "persisted" if persisted else "ephemeral"
            self.JWT_SECRET_STORAGE_REASON = reason
            logger.critical(
                "JWT_SECRET is missing or a known placeholder — using a "
                "generated signing secret (%s). Set a random JWT_SECRET "
                "(>= %d chars) in the environment.",
                "persisted beside the credits DB" if persisted else "ephemeral, lost on restart",
                self.JWT_SECRET_MIN_LENGTH,
            )
        return self


settings = Settings()
