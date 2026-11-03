import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routes import auth, extract, jee, payments
from app.exceptions import SnapNoteError
from app.services.ocr_service import OCR_ENGINE_NAME, ocr_importable, rss_mb
from app.utils.memory_probe import cgroup_memory_mb, current_rss_mb
from app.utils.visual_lesson import typst_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_PROCESS_START = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.storage_service import validate_storage_config

    validate_storage_config()  # fail fast when the storage backend cannot work
    logger.info(
        "Storage backend ready: %s", settings.IMAGE_STORAGE_BACKEND.strip().lower()
    )
    yield


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)
_cors_origins = [
    part.strip() for part in settings.CORS_ALLOW_ORIGINS.split(",") if part.strip()
] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # The API authenticates with an Authorization header, never cookies, so
    # credentialed CORS is unnecessary. Wildcard + credentials echoed ANY
    # origin with credentials (verified live), so the two stay exclusive.
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(extract.router, prefix="/api/extract", tags=["extract"])
app.include_router(payments.router, prefix="/api/payments", tags=["payments"])
app.include_router(jee.router, prefix="/api/jee", tags=["jee"])


def _validation_message(exc: RequestValidationError) -> str:
    """Human-readable first error for the shared ``{error, code}`` contract."""
    errors = exc.errors()
    if not errors:
        return "Invalid request"
    first = errors[0]
    location = [
        str(part)
        for part in first.get("loc", ())
        if part not in ("body", "query", "path", "header", "cookie")
    ]
    field = ".".join(location) or "request"
    message = str(first.get("msg") or "invalid value").strip()
    return f"{field}: {message}"


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Covers framework-raised 404/405 as well as our own HTTPException raises,
    # so every error body is the documented {"error", "code"} shape instead of
    # Starlette's bare {"detail"}. The frontend reads ``error``.
    detail = exc.detail if isinstance(exc.detail, str) and exc.detail else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail, "code": exc.status_code},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    message = _validation_message(exc)
    logger.info("Request validation failed at %s: %s", request.url.path, message)
    content: dict = {"error": message, "code": 422}
    if settings.DEBUG:
        content["details"] = jsonable_encoder(exc.errors())
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(SnapNoteError)
async def snapnote_error_handler(request: Request, exc: SnapNoteError) -> JSONResponse:
    logger.warning(
        "SnapNoteError: %s | path=%s | code=%d",
        exc.message,
        request.url.path,
        exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "code": exc.status_code},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": 500},
    )


def _db_dir_on_mount() -> bool:
    """True when the credits DB lives under a filesystem mount point.

    An absolute path alone is not durability: /app/credits.db is absolute but
    ephemeral in a container. The mount check distinguishes the two.
    """
    parent = Path(settings.CREDITS_DB_PATH).expanduser().parent
    try:
        return parent.is_dir() and os.path.ismount(parent)
    except OSError:
        return False


@app.get("/health")
async def health() -> dict[str, object]:
    ts = typst_status()
    if not ts["available"]:
        logger.warning("health: Typst CLI not available — v3 composition will use bare-hero fallback")
    if not ocr_importable():
        logger.warning("health: OCR engine packages not importable — text extraction will escalate to Gemini")
    mem = cgroup_memory_mb()
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "git_commit": os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT"),
        "typst_available": ts["available"],
        "typst_version": ts["version"],
        "explain_visually_v3": settings.EXPLAIN_VISUALLY_V3,
        "auth_secret_configured": not settings.JWT_SECRET_GENERATED,
        "auth_secret_persisted": settings.JWT_SECRET_PERSISTED,
        "auth_secret_storage": settings.JWT_SECRET_STORAGE,
        "auth_secret_storage_reason": settings.JWT_SECRET_STORAGE_REASON,
        "credits_db_absolute": Path(settings.CREDITS_DB_PATH).is_absolute(),
        "credits_db_on_mount": _db_dir_on_mount(),
        "credits_db_source": settings.CREDITS_DB_SOURCE,
        "ocr_engine": OCR_ENGINE_NAME,
        "ocr_importable": ocr_importable(),
        "rss_mb": rss_mb(),
        "rss_current_mb": current_rss_mb(),
        "cgroup_current_mb": mem["current_mb"],
        "cgroup_peak_mb": mem["peak_mb"],
        "cgroup_limit_mb": mem["limit_mb"],
        "uptime_s": int(time.time() - _PROCESS_START),
    }
