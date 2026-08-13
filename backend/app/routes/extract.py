import asyncio
import logging
import uuid
from fastapi import APIRouter, Form, UploadFile, File

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
from app.exceptions import InvalidInputError, CreditLimitError, UpstreamError
from app.utils.credits_store import (
    get_credits,
    use_credits,
    get_visual_entitlement,
    record_diagram_grant,
    set_visual_url,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _check_credits(device_id: str, cost: int) -> None:
    remaining, _ = get_credits(device_id)
    if remaining < cost:
        raise CreditLimitError()


@router.post("/text", response_model=ExtractionResponse)
async def extract_text_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
    deviceId: str = Form(...),
) -> ExtractionResponse:
    _check_credits(deviceId, settings.TEXT_CREDIT_COST)
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
        logger.info("OCR quality low — escalating to Gemini (device=%s)", deviceId[:8])
        extra_cost = settings.DIAGRAM_CREDIT_COST - settings.TEXT_CREDIT_COST
        remaining, _ = get_credits(deviceId)
        if remaining < extra_cost:
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
                logger.warning("Gemini rate-limited (device=%s). Falling back to OCR.", deviceId[:8])
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

    use_credits(deviceId, final_cost)

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
) -> ExtractionResponse:
    _check_credits(deviceId, settings.DIAGRAM_CREDIT_COST)
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
        logger.error("Diagram extraction timed out after %ss (device=%s)", settings.DIAGRAM_TIMEOUT_SECONDS, deviceId[:8])
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
            logger.warning("Gemini rate-limited on diagram (device=%s).", deviceId[:8])
            raise UpstreamError(service="SnapNote AI", detail="AI extraction is at high demand right now. Try again in a few minutes.")
        raise
    uploaded_url = await upload_task
    ctx = parse_context(context)
    tags = generate_tags(ctx)

    markdown = render_study_notes(study_notes)

    use_credits(deviceId, settings.DIAGRAM_CREDIT_COST)
    diagram_id = uuid.uuid4().hex
    record_diagram_grant(deviceId, diagram_id, study_notes.model_dump_json(exclude={"diagram_spec", "diagram"}))

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
) -> RevisionResponse:
    _check_credits(deviceId, settings.REVISION_CREDIT_COST)
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
        logger.error("Revision guide timed out after %ss (device=%s)", settings.DIAGRAM_TIMEOUT_SECONDS, deviceId[:8])
        raise UpstreamError(
            service="SnapNote AI",
            detail="AI enhancement is taking longer than usual right now. Please try again in a few minutes.",
        )
    except Exception as e:
        err_str = str(e)
        logger.error("Revision guide failed: type=%s msg=%s", type(e).__name__, err_str)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            logger.warning("Gemini rate-limited on revision (device=%s).", deviceId[:8])
            raise UpstreamError(service="SnapNote AI", detail="AI enhancement is at high demand right now. Try again in a few minutes.")
        raise

    use_credits(deviceId, settings.REVISION_CREDIT_COST)

    return RevisionResponse(
        study_notes=study_notes,
        creditsUsed=settings.REVISION_CREDIT_COST,
    )


@router.post("/visual", response_model=VisualExplanationResponse)
async def extract_visual_route(
    image: UploadFile = File(...),
    deviceId: str = Form(...),
    diagramId: str = Form(...),
) -> VisualExplanationResponse:
    """Explain Visually — free, bundled with the 5-credit diagram result.

    The entitlement is tied to a specific completed diagram result (diagramId),
    never a time window. If a visual already exists for this diagram it is
    returned as-is (immutable — the image model is never called again). The
    generation pipeline is: Gemini reads the screenshot + extracted study notes
    -> VisualSpec -> Pollinations render -> quality gate -> one hidden retry ->
    ImgBB. No credits are charged.
    """
    entitlement = get_visual_entitlement(diagramId)
    if entitlement is None:
        raise InvalidInputError(message="No purchased diagram result found for this device.")
    owner_device, visual_url, study_notes_json = entitlement
    if owner_device != deviceId:
        raise InvalidInputError(message="No purchased diagram result found for this device.")

    if visual_url:
        logger.info("Visual already generated for diagram %s (device=%s)", diagramId[:8], deviceId[:8])
        return VisualExplanationResponse(diagramId=diagramId, imageUrl=visual_url, status="already_generated")

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

    png_bytes = await generate_visual(spec)
    if png_bytes is None:
        raise UpstreamError(
            service="SnapNote AI",
            detail="Visual explanation unavailable for this material. The concept may not translate into a clean visual — your study notes above still cover it.",
        )

    visual_url = await asyncio.to_thread(upload_image, png_bytes, {"title": "explain-visually"})
    if not visual_url:
        raise UpstreamError(service="SnapNote AI", detail="Could not store the generated visual. Please try again.")

    if not set_visual_url(diagramId, deviceId, visual_url):
        logger.warning("Visual URL already set for diagram %s; returning existing", diagramId[:8])
        existing = get_visual_entitlement(diagramId)
        if existing is not None:
            visual_url = existing[1]

    return VisualExplanationResponse(diagramId=diagramId, imageUrl=visual_url, status="generated")
