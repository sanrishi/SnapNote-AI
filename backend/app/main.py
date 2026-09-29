import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes import auth, extract, payments
from app.exceptions import SnapNoteError
from app.utils.visual_lesson import typst_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.storage_service import validate_storage_config

    validate_storage_config()  # fail fast when the storage backend cannot work
    logger.info(
        "Storage backend ready: %s", settings.IMAGE_STORAGE_BACKEND.strip().lower()
    )
    yield


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(extract.router, prefix="/api/extract", tags=["extract"])
app.include_router(payments.router, prefix="/api/payments", tags=["payments"])


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "code": exc.status_code},
    )


@app.exception_handler(SnapNoteError)
async def snapnote_error_handler(request: Request, exc: SnapNoteError):
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
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": 500},
    )


@app.get("/health")
async def health():
    ts = typst_status()
    if not ts["available"]:
        logger.warning("health: Typst CLI not available — v3 composition will use bare-hero fallback")
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "typst_available": ts["available"],
        "typst_version": ts["version"],
        "explain_visually_v3": settings.EXPLAIN_VISUALLY_V3,
    }
