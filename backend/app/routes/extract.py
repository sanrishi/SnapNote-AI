import asyncio
import logging
from fastapi import APIRouter, Form, UploadFile, File

from app.config import settings
from app.models.schemas import ExtractionResponse, ExtractionType
from app.services.preprocessor import preprocess, enhance_for_vision
from app.services.ocr_service import (
    read_raw,
    raw_to_lines,
    is_table_layout,
    format_as_markdown,
    format_as_table,
    low_quality_result,
)
from app.services.vision_service import extract_diagram, extract_text_with_llm
from app.services.storage_service import upload_image
from app.utils.tags import parse_context, generate_tags
from app.utils.validation import validate_image_size
from app.exceptions import InvalidInputError, CreditLimitError, UpstreamError
from app.utils.credits_store import get_credits, use_credits

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
            result = await extract_text_with_llm(image_bytes)
            markdown = result
            final_cost = settings.DIAGRAM_CREDIT_COST
        except Exception as e:
            logger.error("Gemini escalation failed: type=%s msg=%s", type(e).__name__, str(e))
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning("Gemini rate-limited (device=%s). Falling back to OCR.", deviceId[:8])
                text_lines = raw_to_lines(raw_results)
                ocr_text = format_as_markdown(text_lines)
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
        markdown = format_as_markdown(text_lines)
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

    try:
        result = await extract_diagram(enhanced)
    except Exception as e:
        err_str = str(e)
        logger.error("Gemini diagram failed: type=%s msg=%s", type(e).__name__, err_str)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            logger.warning("Gemini rate-limited on diagram (device=%s).", deviceId[:8])
            raise UpstreamError(service="SnapNote AI", detail="AI extraction is at high demand right now. Try again in a few minutes.")
        raise
    uploaded_url = await asyncio.to_thread(upload_image, enhanced)
    ctx = parse_context(context)
    tags = generate_tags(ctx)

    if uploaded_url:
        full_markdown = (
            f"{result.markdown}\n\n"
            f"![Diagram]({uploaded_url})"
        )
    else:
        full_markdown = result.markdown

    use_credits(deviceId, settings.DIAGRAM_CREDIT_COST)

    return ExtractionResponse(
        type=ExtractionType.DIAGRAM,
        markdown=full_markdown,
        imageUrl=uploaded_url,
        tags=tags,
        creditsUsed=settings.DIAGRAM_CREDIT_COST,
    )
