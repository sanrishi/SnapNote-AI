import asyncio
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.config import settings
from app.models.schemas import ExtractionResponse, ExtractionType
from app.services.preprocessor import preprocess, enhance_for_vision
from app.services.ocr_service import (
    read_raw,
    raw_to_lines,
    is_table_layout,
    format_as_markdown,
    format_as_table,
)
from app.services.vision_service import extract_diagram
from app.services.storage_service import upload_image
from app.utils.tags import parse_context, generate_tags
from app.utils.validation import validate_image_size
from app.exceptions import InvalidInputError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/text", response_model=ExtractionResponse)
async def extract_text_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
) -> ExtractionResponse:
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
    if is_table_layout(raw_results):
        markdown = format_as_table(raw_results)
        extraction_type = ExtractionType.TABLE
    else:
        text_lines = raw_to_lines(raw_results)
        markdown = format_as_markdown(text_lines)

    return ExtractionResponse(
        type=extraction_type,
        markdown=markdown,
        tags=tags,
        creditsUsed=settings.TEXT_CREDIT_COST,
    )


@router.post("/diagram", response_model=ExtractionResponse)
async def extract_diagram_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
) -> ExtractionResponse:
    image_bytes = await image.read()
    validate_image_size(image_bytes)

    try:
        enhanced = await asyncio.to_thread(enhance_for_vision, image_bytes)
    except ValueError as e:
        raise InvalidInputError(message=str(e))

    result = await extract_diagram(enhanced)
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

    return ExtractionResponse(
        type=ExtractionType.DIAGRAM,
        markdown=full_markdown,
        imageUrl=uploaded_url,
        tags=tags,
        creditsUsed=settings.DIAGRAM_CREDIT_COST,
    )
