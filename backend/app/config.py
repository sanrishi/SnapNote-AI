import logging
import os
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


def _load_or_create_persisted_secret(db_path: str) -> str | None:
    """Keep a generated signing secret across restarts on a mounted disk.

    Production mounts a durable disk (CREDITS_DB_PATH=/data/credits.db), so a
    secret generated because the operator forgot JWT_SECRET can live beside
    the database instead of invalidating every session on each cold start.
    Relative paths (local development, tests) deliberately never write files
    into the working tree.
    """
    db = Path(db_path)
    if not db.is_absolute():
        return None
    path = db.parent / ".jwt_secret"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("Could not read persisted JWT secret %s: %s", path, exc)
            return None
        return existing if not is_weak_jwt_secret(existing) else None
    except OSError as exc:
        logger.error("Could not create persisted JWT secret %s: %s", path, exc)
        return None

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
        return None
    finally:
        os.close(fd)
    return value


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
        if is_weak_jwt_secret(self.JWT_SECRET, self.JWT_SECRET_MIN_LENGTH) and not self.DEBUG:
            persisted = _load_or_create_persisted_secret(self.CREDITS_DB_PATH)
            self.JWT_SECRET = persisted or secrets.token_urlsafe(48)
            self.JWT_SECRET_GENERATED = True
            self.JWT_SECRET_PERSISTED = persisted is not None
            logger.critical(
                "JWT_SECRET is missing or a known placeholder — using a "
                "generated signing secret (%s). Set a random JWT_SECRET "
                "(>= %d chars) in the environment.",
                "persisted beside the credits DB" if persisted else "ephemeral, lost on restart",
                self.JWT_SECRET_MIN_LENGTH,
            )
        return self


settings = Settings()
