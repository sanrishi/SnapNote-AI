import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import NoReturn

from fastapi import APIRouter, Depends, Form, Header, UploadFile, File, HTTPException

from app.config import settings
from app.models.schemas import (
    ExtractionResponse,
    ExtractionType,
    JEEConceptContext,
    RevisionResponse,
    StudyNotes,
    VisualExplanationResponse,
    VisualRenderMode,
    VisualSpec,
)
from app.services.jee_service import build_match_signals, map_concept_to_jee
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
    repair_thin_spec,
)
from app.services.visual_service import generate_visual
from app.utils.visual_lesson import (
    derive_task_from_scene,
    derive_task_from_source,
    promote_equation_claims,
    render_v3_visual,
    should_use_v3,
    spec_needs_repair,
    v3_store_mode,
    validate_legacy_visual,
)
from app.services.storage_service import upload_image
from app.utils.grounding import ground_visual_context
from app.utils.image_window import (
    admit_image_work,
    release_image_work,
    run_in_image_window,
)
from app.utils.render_notes import render_study_notes, render_jee_section
from app.utils.tags import parse_context, generate_tags
from app.utils.validation import validate_image_size
from app.exceptions import (
    AuthError,
    CreditLimitError,
    InvalidInputError,
    ServerBusyError,
    UpstreamError,
)
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

_VISUAL_REFUSAL = (
    "Visual explanation unavailable for this material. The concept may not "
    "translate into a clean visual — your study notes above still cover it."
)
_VISUAL_HIGH_DEMAND = (
    "Visual explanations are at high demand right now. Please try again in "
    "a few minutes — your study notes above are already saved."
)


def _is_rate_limited(message: str) -> bool:
    """Gemini capacity/quota signature (429 or RESOURCE_EXHAUSTED)."""
    return "429" in message or "RESOURCE_EXHAUSTED" in message


async def image_admission() -> AsyncIterator[None]:
    """Bounded-capacity gate for heavy routes, BEFORE credits or image work.

    Only MAX_ADMITTED requests may run at once; beyond that the request fails
    fast with 503 (retry guidance) instead of queueing behind a single heavy
    worker past the client timeout. A rejection here never charges credits.
    """
    if not admit_image_work():
        logger.warning("Extract admission rejected: heavy slot busy (fast 503)")
        raise ServerBusyError()
    try:
        yield
    finally:
        release_image_work()


def _refuse_visual(diagram_id: str, stage: str, reason: str = "") -> NoReturn:
    """Single structured refusal log + 502 for the Explain Visually pipeline.

    stage is the pipeline phase that refused: specification_generation |
    contract_validation | verification | rendering | upstream.
    """
    logger.warning(
        "Explain Visually refused (diagram %s): stage=%s reason=%s",
        diagram_id[:8],
        stage,
        (reason or "n/a")[:300],
    )
    raise UpstreamError(service="SnapNote AI", detail=_VISUAL_REFUSAL)


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


def _jee_for_notes(study_notes: StudyNotes) -> tuple[JEEConceptContext | None, str]:
    """Free deterministic JEE context for extracted study notes.

    Returns (context | None, markdown_section). No credits, no LLM calls.
    Unresolved input yields (None, "") so generic screenshots stay JEE-free.
    """
    key_terms = build_match_signals(study_notes)
    try:
        ctx = map_concept_to_jee(study_notes.topic.title, key_terms)
    except Exception:
        logger.warning("JEE mapping failed; continuing without JEE context")
        return None, ""
    if ctx.match_status not in ("matched", "ambiguous"):
        return None, ""
    return ctx, render_jee_section(ctx)


@router.post("/text", response_model=ExtractionResponse)
async def extract_text_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
    deviceId: str = Form(...),
    authorization: str | None = Header(default=None),
    _admission: None = Depends(image_admission),
) -> ExtractionResponse:
    effective_id, is_user = _resolve_identity(deviceId, authorization)
    check_rate_limits(effective_id)
    _check_credits_effective(effective_id, is_user, settings.TEXT_CREDIT_COST)
    image_bytes = await image.read()
    validate_image_size(image_bytes)

    try:
        processed = await run_in_image_window(preprocess, image_bytes)
    except ValueError as e:
        raise InvalidInputError(message=str(e))

    ctx = parse_context(context)
    tags = generate_tags(ctx)
    extraction_type = ExtractionType.TEXT

    raw_results = await run_in_image_window(read_raw, processed)
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
            enhanced = await run_in_image_window(enhance_for_vision, image_bytes)
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
    _admission: None = Depends(image_admission),
) -> ExtractionResponse:
    effective_id, is_user = _resolve_identity(deviceId, authorization)
    check_rate_limits(effective_id)
    _check_credits_effective(effective_id, is_user, settings.DIAGRAM_CREDIT_COST)
    image_bytes = await image.read()
    validate_image_size(image_bytes)

    try:
        enhanced = await run_in_image_window(enhance_for_vision, image_bytes)
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

    jee_ctx, jee_section = _jee_for_notes(study_notes)
    if jee_section:
        markdown = f"{markdown}\n{jee_section}"

    return ExtractionResponse(
        type=ExtractionType.DIAGRAM,
        markdown=markdown,
        imageUrl=uploaded_url,
        tags=tags,
        creditsUsed=settings.DIAGRAM_CREDIT_COST,
        studyNotes=study_notes,
        diagramId=diagram_id,
        jee=jee_ctx,
    )


@router.post("/revision", response_model=RevisionResponse)
async def extract_revision_route(
    image: UploadFile = File(...),
    context: str = Form("{}"),
    deviceId: str = Form(...),
    authorization: str | None = Header(default=None),
    _admission: None = Depends(image_admission),
) -> RevisionResponse:
    effective_id, is_user = _resolve_identity(deviceId, authorization)
    check_rate_limits(effective_id)
    _check_credits_effective(effective_id, is_user, settings.REVISION_CREDIT_COST)
    image_bytes = await image.read()
    validate_image_size(image_bytes)

    context_hint = context.strip()[:2000] if context.strip() not in ("", "{}", "null") else ""

    try:
        enhanced = await run_in_image_window(enhance_for_vision, image_bytes)
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

    jee_ctx, _ = _jee_for_notes(study_notes)

    return RevisionResponse(
        study_notes=study_notes,
        creditsUsed=settings.REVISION_CREDIT_COST,
        jee=jee_ctx,
    )


@router.post("/visual", response_model=VisualExplanationResponse)
async def extract_visual_route(
    image: UploadFile = File(...),
    deviceId: str = Form(...),
    diagramId: str = Form(...),
    authorization: str | None = Header(default=None),
    _admission: None = Depends(image_admission),
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
        enhanced = await run_in_image_window(enhance_for_vision, image_bytes)
    except ValueError as e:
        raise InvalidInputError(message=str(e))

    study_notes: StudyNotes | None = None
    if study_notes_json:
        try:
            study_notes = StudyNotes.model_validate_json(study_notes_json)
        except Exception as e:
            logger.warning("Stored study notes invalid for diagram %s: %s", diagramId[:8], e)
    try:
        spec = await build_visual_spec(enhanced, study_notes)
    except UpstreamError as e:
        if _is_rate_limited(e.message):
            logger.warning(
                "Explain Visually declined (diagram %s): stage=specification_generation reason=rate_limited",
                diagramId[:8],
            )
            raise UpstreamError(service="SnapNote AI", detail=_VISUAL_HIGH_DEMAND)
        logger.warning(
            "Explain Visually refused (diagram %s): stage=specification_generation reason=%s",
            diagramId[:8], e.message,
        )
        raise

    # Task coverage + validation gate (fail closed) — deterministic specs
    # only; generative specs are policed by generate_visual's own contract
    # (concept + grounded key elements + quality/legibility gates).
    # - scene lesson (non-generic): must carry an enforced task contract.
    #   task absent -> derive from SOURCE (StudyNotes; scene-derived
    #   fallback only when notes are missing and documented weak).
    # - generic scene: task contract waived (a central box cannot satisfy
    #   source-content requirements); density gate decides.
    # - sceneless legacy card: allowed only as title + card content
    #   (validate_legacy_visual) — the legacy renderer's other supported
    #   shape.
    # Derivation/assessment exceptions fall through to repair-or-refuse,
    # never to silent rendering. Non-spec pipeline output is refused
    # outright (build guarantees VisualSpec or raises). StudyNotes were
    # already delivered by the diagram response, so refusal here never
    # discards student work.
    if not isinstance(spec, VisualSpec):
        _refuse_visual(diagramId, "specification_generation", "pipeline returned non-spec output")
    if spec.render_mode == VisualRenderMode.DETERMINISTIC:
        scene_obj = spec.deterministic.scene
        scene_kind = scene_obj.scene_kind if scene_obj is not None else ""
        try:
            if scene_kind and scene_kind != "generic" and spec.deterministic.task is None:
                if study_notes is not None:
                    derived = derive_task_from_source(study_notes, scene_kind)
                else:
                    derived = derive_task_from_scene(spec)
                if derived is not None:
                    spec.deterministic.task = derived
                    logger.info(
                        "Explain Visually derived task contract for diagram %s",
                        diagramId[:8],
                    )
            gate_errors = validate_legacy_visual(spec)
            promoted = promote_equation_claims(spec, gate_errors)
            if promoted:
                logger.info(
                    "Explain Visually promoted %d equations-only claim(s) into visible callouts (diagram %s)",
                    promoted, diagramId[:8],
                )
                gate_errors = validate_legacy_visual(spec)
        except Exception as e:
            logger.warning("Explain Visually task/validation stage failed (diagram %s): %s",
                           diagramId[:8], e)
            gate_errors = ["task/validation stage failed"]
        needs_contract = (
            scene_obj is not None
            and scene_kind != "generic"
            and spec.deterministic.task is None
        )
        try:
            upgrade_reason = spec_needs_repair(spec)
        except Exception as e:
            logger.warning("Explain Visually repair assessment failed (diagram %s): %s",
                           diagramId[:8], e)
            upgrade_reason = None
        must_fix = bool(gate_errors) or needs_contract
        if upgrade_reason is not None or must_fix:
            reason = (
                "; ".join(gate_errors[:3])
                or ("missing task contract" if needs_contract else "")
                or upgrade_reason
                or "thin spec"
            )
            logger.info(
                "Explain Visually spec not enforceable (%s); attempting bounded repair (diagram %s)",
                reason, diagramId[:8],
            )
            # Bounded repair: at most 2 attempts, and a second one ONLY when
            # the request already fails validation (must_fix) — an upgrade-
            # only spec renders either way, so one attempt is enough there.
            # A malformed/parse-failed repair or a still-invalid spec earns
            # the second attempt with the fresh revalidation errors as the
            # new reason; anything still failing refuses honestly after it.
            rechecked: list[str] = ["repair failed"]
            reconciled: VisualSpec | None = None
            accepted = False
            attempts = 2 if must_fix else 1
            rate_limited_repairs = 0
            repair_specs_seen = 0
            for attempt in range(1, attempts + 1):
                reconciled = None
                try:
                    reconciled = await repair_thin_spec(spec, enhanced, study_notes, reason)
                except Exception as e:
                    logger.warning("Explain Visually repair call failed (diagram %s, attempt %d): %s",
                                   diagramId[:8], attempt, e)
                    if _is_rate_limited(str(e)):
                        rate_limited_repairs += 1
                    reconciled = None
                rechecked = ["repair failed"]
                if reconciled is not None and isinstance(reconciled, VisualSpec):
                    repair_specs_seen += 1
                    try:
                        rscene = reconciled.deterministic.scene
                        rkind = rscene.scene_kind if rscene is not None else ""
                        if rkind and rkind != "generic" and reconciled.deterministic.task is None:
                            if study_notes is not None:
                                rderived = derive_task_from_source(study_notes, rkind)
                            else:
                                rderived = derive_task_from_scene(reconciled)
                            if rderived is not None:
                                reconciled.deterministic.task = rderived
                        rechecked = validate_legacy_visual(reconciled)
                        rpromoted = promote_equation_claims(reconciled, rechecked)
                        if rpromoted:
                            logger.info(
                                "Explain Visually promoted %d equations-only claim(s) from repaired spec (diagram %s)",
                                rpromoted, diagramId[:8],
                            )
                            rechecked = validate_legacy_visual(reconciled)
                        if (
                            rscene is not None
                            and rkind != "generic"
                            and reconciled.deterministic.task is None
                        ):
                            rechecked = [*rechecked, "repaired spec has no task contract"]
                    except Exception as e:
                        rechecked = [f"revalidation failed: {e}"]
                if reconciled is not None and not rechecked:
                    logger.info("Explain Visually repair accepted (diagram %s, attempt %d)",
                                diagramId[:8], attempt)
                    spec = reconciled
                    accepted = True
                    break
                if attempt < attempts:
                    if rechecked and rechecked != ["repair failed"]:
                        reason = "; ".join(rechecked[:3])
                    logger.info(
                        "Explain Visually repair attempt %d still invalid (%s); retrying (diagram %s)",
                        attempt, reason, diagramId[:8],
                    )
            if not accepted:
                if must_fix:
                    if rate_limited_repairs and not repair_specs_seen:
                        logger.warning(
                            "Explain Visually declined (diagram %s): stage=contract_validation reason=rate_limited",
                            diagramId[:8],
                        )
                        raise UpstreamError(service="SnapNote AI", detail=_VISUAL_HIGH_DEMAND)
                    _refuse_visual(
                        diagramId, "contract_validation", "; ".join(rechecked[:3])
                    )
                else:
                    logger.info(
                        "Explain Visually upgrade repair not applied; rendering validated original (diagram %s)",
                        diagramId[:8],
                    )

    # Trust layer: the study-notes context summary describes the SOURCE
    # screenshot, but it renders under OUR rebuilt visual — drop any sentence
    # claiming structural elements the emitted scene does not draw.
    visual_context: str | None = None
    if study_notes is not None and study_notes.visual_context.present:
        grounded = ground_visual_context(study_notes.visual_context.summary, spec)
        visual_context = grounded or None

    # v3 composition boundary (flag-gated). Same spec object — no second
    # understanding pipeline. Generative specs always use the legacy path.
    v3_composed = should_use_v3(spec)
    t_visual = time.perf_counter()
    if v3_composed:
        logger.info("Explain Visually v3 composition path (diagram %s)", diagramId[:8])
        result = await render_v3_visual(spec)
    else:
        if settings.EXPLAIN_VISUALLY_V3:
            logger.info(
                "Explain Visually v3 flag on but render_mode=%s; using legacy path (diagram %s)",
                spec.render_mode, diagramId[:8],
            )
        else:
            logger.info("Explain Visually legacy path (v3 flag off, diagram %s)", diagramId[:8])
        result = await generate_visual(spec)
    if result is None:
        _refuse_visual(diagramId, "rendering", "renderer returned no visual")

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
            visualContext=visual_context,
        )

    t_upload = time.perf_counter()
    try:
        from app.services.storage_service import _detect_mime
        logger.info(
            "Explain Visually uploading (diagram %s): mode=%s bytes=%d mime=%s",
            diagramId[:8], mode, len(payload), _detect_mime(payload),
        )
    except Exception as e:
        logger.warning(
            "Explain Visually payload failed format check (diagram %s): %s",
            diagramId[:8], e,
        )
        raise UpstreamError(service="SnapNote AI", detail="Could not store the generated visual. Please try again.")
    visual_url = await asyncio.to_thread(upload_image, payload, {"title": "explain-visually"})
    upload_ms = (time.perf_counter() - t_upload) * 1000
    if not visual_url:
        logger.warning(
            "Explain Visually storage upload returned no URL (diagram %s, mode=%s, upload=%.1fms)",
            diagramId[:8], mode, upload_ms,
        )
        raise UpstreamError(service="SnapNote AI", detail="Could not store the generated visual. Please try again.")

    # A v3-composed PNG is deterministic content (Typst from a validated spec),
    # not a generative illustration — label it honestly for observability.
    store_mode = v3_store_mode(v3_composed, mode)
    logger.info(
        "Explain Visually stored (diagram %s, v3=%s, mode=%s, stored_as=%s, upload=%.1fms, total=%.1fms)",
        diagramId[:8], v3_composed, mode, store_mode, upload_ms, (time.perf_counter() - t_visual) * 1000,
    )
    if not set_visual_result(diagramId, effective_id, store_mode, visual_url=visual_url):
        logger.warning("Visual already set for diagram %s; returning existing", diagramId[:8])
        existing = get_visual_entitlement(diagramId)
        if existing is not None:
            _, visual_url, render_mode, visual_svg, _ = existing

    return VisualExplanationResponse(
        diagramId=diagramId,
        renderMode=store_mode if v3_composed else "generative",
        imageUrl=visual_url,
        status="generated",
        visualContext=visual_context,
    )
