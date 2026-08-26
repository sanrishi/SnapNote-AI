import asyncio
import logging
import uuid
from fastapi import APIRouter, Form, Header, UploadFile, File, HTTPException

from app.config import settings
from app.models.schemas import (
    ExtractionResponse,
    ExtractionType,
    RevisionResponse,
    StudyNotes,
    VisualExplanationResponse,
)
from app.services.preprocessor import preprocess, enhance_for_vision
from app.services.ocr_service import (
    read_raw,
    raw_to_lines,
    is_table_layout,
    format_as_table,
    format_structured_text,
    low_quality_result,
)
from app.services.vision_service import (
    build_visual_spec,
    extract_revision_guide,
    extract_study_notes,
    extract_text_with_llm,
)
from app.services.visual_service import generate_visual
from app.services.storage_service import upload_image
from app.utils.render_notes import render_study_notes
from app.utils.tags import parse_context, generate_tags
from app.utils.validation import validate_image_size
from app.exceptions import AuthError, InvalidInputError, CreditLimitError, UpstreamError
from app.utils.auth import decode_token
from app.utils.credits_store import (
    get_credits,
    get_user_by_id,
    get_user_credits,
    get_visual_entitlement,
    record_diagram_grant,
    set_visual_result,
    use_credits,
    use_user_credits,
)
from app.utils.rate_limiter import check_rate_limits

logger = logging.getLogger(__name__)
router = APIRouter()


def _resolve_identity(deviceId: str, authorization: str | None) -> tuple[str, bool]:
    if authorization and authorization.startswith("Bearer "):
        try:
            data = decode_token(authorization[7:])
            uid = data.get("sub")
            if uid and get_user_by_id(uid) is not None:
                return uid, True
        except Exception:
            pass
    return deviceId, False


def _check_credits(device_id: str, cost: int) -> None:
    remaining, _ = get_credits(device_id)
    if remaining < cost:
        raise CreditLimitError()


def _check_credits_effective(effective_id: str, is_user: bool, cost: int) -> None:
    if is_user:
        remaining, _ = get_user_credits(effective_id)
        if remaining < cost:
            raise CreditLimitError()
    else:
        remaining, used = get_credits(effective_id)
        # Anonymous gets ANONYMOUS_FREE_USES previews before signup is required (prevents reload abuse)
        if used >= settings.ANONYMOUS_FREE_USES:
            raise AuthError(message="Please sign up or log in to continue. Your free preview is used.")
        if remaining < cost:
            raise CreditLimitError()


def _use_credits_effective(effective_id: str, is_user: bool, amount: int) -> int:
    if is_user:
        return use_user_credits(effective_id, amount)
    return use_credits(effective_id, amount)


@router.post("/text", response_model=ExtractionResponse)
async def extract_text_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
    deviceId: str = Form(...),
    authorization: str | None = Header(default=None),
) -> ExtractionResponse:
    effective_id, is_user = _resolve_identity(deviceId, authorization)
    check_rate_limits(effective_id)
    _check_credits_effective(effective_id, is_user, settings.TEXT_CREDIT_COST)
    image_bytes = await image.read()
    validate_image_size(image_bytes)

    try:
        processed = await asyncio.to_thread(preprocess, image_bytes)
    except ValueError as e:
        raise InvalidInputError(message=str(e))

    ctx = parse_context(context)
    tags = generate_tags(ctx)
    extraction_type = ExtractionType.TEXT

    raw_results = await asyncio.to_thread(read_raw, processed)
    if low_quality_result(raw_results):
        logger.info("OCR quality low — escalating to Gemini (id=%s)", effective_id[:8])
        extra_cost = settings.DIAGRAM_CREDIT_COST - settings.TEXT_CREDIT_COST
        # check extra cost with effective identity
        if is_user:
            rem, _ = get_user_credits(effective_id)
            if rem < extra_cost:
                raise CreditLimitError()
        else:
            rem, _ = get_credits(effective_id)
            if rem < extra_cost:
                raise CreditLimitError()
        try:
            enhanced = await asyncio.to_thread(enhance_for_vision, image_bytes)
            result = await extract_text_with_llm(enhanced)
            markdown = result
            final_cost = settings.DIAGRAM_CREDIT_COST
        except Exception as e:
            logger.error("Gemini escalation failed: type=%s msg=%s", type(e).__name__, str(e))
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning("Gemini rate-limited (id=%s). Falling back to OCR.", effective_id[:8])
                text_lines = raw_to_lines(raw_results)
                ocr_text = format_structured_text(text_lines)
                markdown = ocr_text + "\n\n---\n*Enhanced extraction unavailable right now (high demand). Showing OCR result.*"
                final_cost = settings.TEXT_CREDIT_COST
            else:
                raise
        extraction_type = ExtractionType.TEXT
    elif is_table_layout(raw_results):
        markdown = format_as_table(raw_results)
        extraction_type = ExtractionType.TABLE
        final_cost = settings.TEXT_CREDIT_COST
    else:
        text_lines = raw_to_lines(raw_results)
        markdown = format_structured_text(text_lines)
        final_cost = settings.TEXT_CREDIT_COST

    _use_credits_effective(effective_id, is_user, final_cost)

    return ExtractionResponse(
        type=extraction_type,
        markdown=markdown,
        tags=tags,
        creditsUsed=final_cost,
    )


@router.post("/diagram", response_model=ExtractionResponse)
async def extract_diagram_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
    deviceId: str = Form(...),
    authorization: str | None = Header(default=None),
) -> ExtractionResponse:
    effective_id, is_user = _resolve_identity(deviceId, authorization)
    check_rate_limits(effective_id)
    _check_credits_effective(effective_id, is_user, settings.DIAGRAM_CREDIT_COST)
    image_bytes = await image.read()
    validate_image_size(image_bytes)

    try:
        enhanced = await asyncio.to_thread(enhance_for_vision, image_bytes)
    except ValueError as e:
        raise InvalidInputError(message=str(e))

    upload_task = asyncio.create_task(asyncio.to_thread(upload_image, enhanced))
    try:
        study_notes = await asyncio.wait_for(
            extract_study_notes(enhanced),
            timeout=settings.DIAGRAM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("Diagram extraction timed out after %ss (id=%s)", settings.DIAGRAM_TIMEOUT_SECONDS, effective_id[:8])
        upload_task.cancel()
        raise UpstreamError(
            service="SnapNote AI",
            detail="AI extraction is taking longer than usual right now. Please try again in a few minutes.",
        )
    except Exception as e:
        upload_task.cancel()
        err_str = str(e)
        logger.error("Gemini study notes failed: type=%s msg=%s", type(e).__name__, err_str)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            logger.warning("Gemini rate-limited on diagram (id=%s).", effective_id[:8])
            raise UpstreamError(service="SnapNote AI", detail="AI extraction is at high demand right now. Try again in a few minutes.")
        raise
    uploaded_url = await upload_task
    ctx = parse_context(context)
    tags = generate_tags(ctx)

    markdown = render_study_notes(study_notes)

    _use_credits_effective(effective_id, is_user, settings.DIAGRAM_CREDIT_COST)
    diagram_id = uuid.uuid4().hex
    record_diagram_grant(effective_id, diagram_id, study_notes.model_dump_json(exclude={"diagram_spec", "diagram"}))

    return ExtractionResponse(
        type=ExtractionType.DIAGRAM,
        markdown=markdown,
        imageUrl=uploaded_url,
        tags=tags,
        creditsUsed=settings.DIAGRAM_CREDIT_COST,
        studyNotes=study_notes,
        diagramId=diagram_id,
    )


@router.post("/revision", response_model=RevisionResponse)
async def extract_revision_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
    deviceId: str = Form(...),
    authorization: str | None = Header(default=None),
) -> RevisionResponse:
    effective_id, is_user = _resolve_identity(deviceId, authorization)
    check_rate_limits(effective_id)
    _check_credits_effective(effective_id, is_user, settings.REVISION_CREDIT_COST)
    image_bytes = await image.read()
    validate_image_size(image_bytes)

    context_hint = context.strip()[:2000] if context.strip() not in ("", "{}", "null") else ""

    try:
        enhanced = await asyncio.to_thread(enhance_for_vision, image_bytes)
    except ValueError as e:
        raise InvalidInputError(message=str(e))

    try:
        study_notes = await asyncio.wait_for(
            extract_revision_guide(enhanced, context_hint),
            timeout=settings.DIAGRAM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("Revision guide timed out after %ss (id=%s)", settings.DIAGRAM_TIMEOUT_SECONDS, effective_id[:8])
        raise UpstreamError(
            service="SnapNote AI",
            detail="AI enhancement is taking longer than usual right now. Please try again in a few minutes.",
        )
    except Exception as e:
        err_str = str(e)
        logger.error("Revision guide failed: type=%s msg=%s", type(e).__name__, err_str)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            logger.warning("Gemini rate-limited on revision (id=%s).", effective_id[:8])
            raise UpstreamError(service="SnapNote AI", detail="AI enhancement is at high demand right now. Try again in a few minutes.")
        raise

    _use_credits_effective(effective_id, is_user, settings.REVISION_CREDIT_COST)

    return RevisionResponse(
        study_notes=study_notes,
        creditsUsed=settings.REVISION_CREDIT_COST,
    )


@router.post("/visual", response_model=VisualExplanationResponse)
async def extract_visual_route(
    image: UploadFile = File(...),
    deviceId: str = Form(...),
    diagramId: str = Form(...),
    authorization: str | None = Header(default=None),
) -> VisualExplanationResponse:
    """Explain Visually — free, bundled with the 5-credit diagram result.

    The entitlement is tied to a specific completed diagram result (diagramId),
    never a time window. If a visual already exists for this diagram it is
    returned as-is (immutable — the image model is never called again). The
    generation pipeline is hybrid: Gemini reads the screenshot + extracted study
    notes -> VisualSpec -> render_mode dispatch:
      - deterministic -> code renders a clean sanitized SVG (exact text/symbols)
      - generative -> Pollinations render -> brightness gate -> conditional OCR
        legibility gate (only when text_required) -> one hidden retry -> ImgBB.
    No credits are charged.
    """
    effective_id, is_user = _resolve_identity(deviceId, authorization)
    check_rate_limits(effective_id)
    entitlement = get_visual_entitlement(diagramId)
    if entitlement is None:
        raise InvalidInputError(message="No purchased diagram result found for this device.")
    owner_device, visual_url, render_mode, visual_svg, study_notes_json = entitlement
    if owner_device != effective_id:
        raise InvalidInputError(message="No purchased diagram result found for this device.")

    if visual_url or visual_svg:
        logger.info("Visual already generated for diagram %s (id=%s)", diagramId[:8], effective_id[:8])
        return VisualExplanationResponse(
            diagramId=diagramId,
            renderMode=render_mode or "deterministic",
            imageUrl=visual_url or None,
            imageSvg=visual_svg or None,
            status="already_generated",
        )

    image_bytes = await image.read()
    validate_image_size(image_bytes)

    try:
        enhanced = await asyncio.to_thread(enhance_for_vision, image_bytes)
    except ValueError as e:
        raise InvalidInputError(message=str(e))

    study_notes: StudyNotes | None = None
    if study_notes_json:
        try:
            study_notes = StudyNotes.model_validate_json(study_notes_json)
        except Exception as e:
            logger.warning("Stored study notes invalid for diagram %s: %s", diagramId[:8], e)
    spec = await build_visual_spec(enhanced, study_notes)

    result = await generate_visual(spec)
    if result is None:
        raise UpstreamError(
            service="SnapNote AI",
            detail="Visual explanation unavailable for this material. The concept may not translate into a clean visual — your study notes above still cover it.",
        )

    mode, payload = result
    if mode == "svg":
        visual_svg = payload
        if not set_visual_result(diagramId, effective_id, "deterministic", visual_svg=visual_svg):
            logger.warning("Visual already set for diagram %s; returning existing", diagramId[:8])
            existing = get_visual_entitlement(diagramId)
            if existing is not None:
                _, visual_url, render_mode, visual_svg, _ = existing
        return VisualExplanationResponse(
            diagramId=diagramId,
            renderMode="deterministic",
            imageSvg=visual_svg,
            status="generated",
        )

    visual_url = await asyncio.to_thread(upload_image, payload, {"title": "explain-visually"})
    if not visual_url:
        raise UpstreamError(service="SnapNote AI", detail="Could not store the generated visual. Please try again.")

    if not set_visual_result(diagramId, effective_id, "generative", visual_url=visual_url):
        logger.warning("Visual already set for diagram %s; returning existing", diagramId[:8])
        existing = get_visual_entitlement(diagramId)
        if existing is not None:
            _, visual_url, render_mode, visual_svg, _ = existing

    return VisualExplanationResponse(
        diagramId=diagramId,
        renderMode="generative",
        imageUrl=visual_url,
        status="generated",
    )
