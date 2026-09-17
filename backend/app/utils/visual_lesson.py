"""Production composition contract: VisualSpec -> educational visual lesson.

Pipeline (additive — the old path `visual_service.generate_visual` is untouched
and remains the fallback until this path is validated):

    VisualSpec
      -> validate_lesson_spec      (semantic validity, exact values preserved)
      -> select_family             (vector | coordinate | flow | generic)
      -> render_hero_geometry      (geometry + geometry labels ONLY, no prose)
      -> extract_lesson_content    (structured prose: title/callouts/reasoning/result)
      -> build_lesson_typst        (Typst owns 100% of layout + prose)
      -> compile_typst             (official CLI; deterministic output)

Contract: ``render_visual_lesson(spec) -> RenderedLesson``.

Rules enforced here, not by convention:
- Gemini/LLM output is NEVER executed and NEVER trusted with pixels.
- The hero SVG carries zero prose (no title/caption/relation cards).
- All numeric values pass through byte-identical (no rounding, no rewriting).
- If Typst is unavailable, fall back to the bare hero SVG (honest, never fake).
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.models.schemas import DeterministicVisual, VisualRenderMode, VisualScene, VisualSpec
from app.utils.visual_renderer import render_hero_geometry

logger = logging.getLogger(__name__)

# scene_kind -> visual family. New families plug in here; no per-subject code.
SUPPORTED_FAMILIES: dict[str, str] = {
    "force_diagram": "vector",
    "plot": "coordinate",
    "process_flow": "flow",
    "generic": "generic",
}

TYMST_TIMEOUT_SECONDS = 25
MAX_CALLOUTS = 4


@dataclass
class LessonCallout:
    heading: str
    body: str = ""


@dataclass
class LessonContent:
    title: str
    subtitle: str = ""
    hero_svg: str = ""
    hero_kind: str = ""
    callouts: list[LessonCallout] = field(default_factory=list)
    reasoning: str = ""
    result_text: str = ""
    takeaway: str = ""


@dataclass
class RenderedLesson:
    out_format: str  # "pdf" | "png" | "svg"
    data: bytes
    hero_kind: str
    ms_total: float
    ms_hero: float = 0.0
    ms_compose: float = 0.0
    warnings: list[str] = field(default_factory=list)
    fallback_used: bool = False


class LessonValidationError(ValueError):
    """Semantic spec is invalid — refuse to render rather than guess."""


class TypstUnavailable(RuntimeError):
    """Typst CLI binary not found — caller must fall back honestly."""


class TypstCompileError(RuntimeError):
    """Typst source failed to compile — never return a partial visual."""


def _finite(value: object, what: str, errors: list[str]) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        errors.append(f"{what} is not a number: {value!r}")
        return None
    if not math.isfinite(number):
        errors.append(f"{what} is not finite: {value!r}")
        return None
    return number


def validate_lesson_spec(spec: VisualSpec) -> list[str]:
    """Return a list of semantic errors (empty == valid). Pure, no I/O."""
    errors: list[str] = []
    det: DeterministicVisual = spec.deterministic
    if not (det.title or "").strip():
        errors.append("deterministic.title is empty")
    scene: VisualScene | None = det.scene
    if scene is None:
        errors.append("deterministic.scene is missing")
        return errors
    if scene.scene_kind not in SUPPORTED_FAMILIES:
        errors.append(f"unsupported scene_kind: {scene.scene_kind!r}")
        return errors
    kind = scene.scene_kind
    if kind == "force_diagram":
        force = scene.force
        if force is None:
            errors.append("force_diagram has no force payload")
            return errors
        labeled = [v for v in force.vectors if (v.label or "").strip()]
        if not labeled:
            errors.append("force_diagram has no labeled vectors")
        known = {(force.object.label or "").strip(), *(v.label.strip() for v in labeled)}
        for v in labeled:
            _finite(v.angle_deg, f"vector {v.label!r} angle_deg", errors)
            _finite(v.length, f"vector {v.label!r} length", errors)
            tail = (v.tail or "").strip()
            if tail and tail not in known:
                errors.append(f"vector {v.label!r} tails on unknown {tail!r}")
        for a in force.angles:
            for lab in a.between:
                if lab.strip() not in {v.label.strip() for v in labeled}:
                    errors.append(f"angle {a.label!r} references unknown vector {lab!r}")
        for arc in force.arcs:
            if (arc.around or "").strip() and (arc.around or "").strip() not in known:
                errors.append(f"arc {arc.label!r} wraps unknown {arc.around!r}")
    elif kind == "plot":
        plot = scene.plot
        if plot is None or not plot.curves:
            errors.append("plot has no curves")
            return errors
        _finite(plot.x_min, "plot.x_min", errors)
        _finite(plot.x_max, "plot.x_max", errors)
        _finite(plot.y_min, "plot.y_min", errors)
        _finite(plot.y_max, "plot.y_max", errors)
        try:
            if float(plot.x_max) <= float(plot.x_min) or float(plot.y_max) <= float(plot.y_min):
                errors.append("plot ranges are degenerate")
        except (TypeError, ValueError):
            pass
        for c in plot.curves:
            if not (c.expr or "").strip() and not c.points:
                errors.append(f"curve {c.label!r} has neither expr nor points")
            for p in c.points:
                if len(p) != 2 or _finite(p[0], "point x", errors) is None or _finite(p[1], "point y", errors) is None:
                    errors.append(f"curve {c.label!r} has a malformed point")
                    break
    elif kind == "process_flow":
        flow = scene.flow
        if flow is None or len([n for n in flow.nodes if (n.label or "").strip()]) < 2:
            errors.append("process_flow needs at least 2 labeled nodes")
            return errors
        n = len(flow.nodes)
        for c in flow.connectors:
            if not (0 <= c.source < n and 0 <= c.target < n):
                errors.append(f"connector {c.source}->{c.target} out of range for {n} nodes")
    elif kind == "generic":
        if scene.generic is None or not (scene.generic.central_label or "").strip():
            errors.append("generic scene needs a central_label")
    comp = det.composition
    if comp is not None:
        errors.extend(_validate_composition(comp))
    return errors


def _validate_composition(comp) -> list[str]:
    """Validate the explicit v3 composition block (shape already enforced by
    pydantic: id pattern, lengths, counts, extra=forbid). Checks semantic
    rules only; values pass through byte-identical."""
    from app.models.schemas import LessonComposition

    assert isinstance(comp, LessonComposition)
    errors: list[str] = []
    seen_callouts: set[str] = set()
    for c in comp.callouts:
        if c.id in seen_callouts:
            errors.append(f"composition callout id duplicated: {c.id!r}")
        seen_callouts.add(c.id)
        if not c.label.strip():
            errors.append(f"composition callout {c.id!r} has empty label")
    seen_steps: set[str] = set()
    for s in comp.reasoning:
        if s.id in seen_steps:
            errors.append(f"composition reasoning id duplicated: {s.id!r}")
        seen_steps.add(s.id)
        if not s.expression.strip():
            errors.append(f"composition reasoning {s.id!r} has empty expression")
    if comp.result is not None and not (comp.result.expression or "").strip():
        errors.append("composition result has empty expression")
    if (
        not comp.callouts
        and not comp.reasoning
        and comp.result is None
        and not (comp.takeaway or "").strip()
    ):
        errors.append("composition is empty: needs callouts, reasoning, result or takeaway")
    return errors


def select_family(spec: VisualSpec) -> str:
    """Map scene_kind to a visual family. Raises LessonValidationError if unknown."""
    kind = (spec.deterministic.scene.scene_kind if spec.deterministic.scene else "")
    try:
        return SUPPORTED_FAMILIES[kind]
    except KeyError:
        raise LessonValidationError(f"unsupported scene_kind: {kind!r}") from None


def should_use_v3(spec: VisualSpec) -> bool:
    """Route decision for Explain Visually: v3 composition boundary or legacy.

    True only when the flag is on AND the spec is deterministic. Generative
    specs always use the legacy path (v3 composition is deterministic-only).
    Pure function of (flag, spec) — no I/O, so the route stays thin and the
    decision is unit-testable without FastAPI.
    """
    from app.config import settings

    return bool(settings.EXPLAIN_VISUALLY_V3) and spec.render_mode == VisualRenderMode.DETERMINISTIC


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def extract_lesson_content(spec: VisualSpec, hero_svg: str = "") -> LessonContent:
    """Pull structured lesson content from the spec WITHOUT rewriting values.

    Numbers, labels and equations pass through byte-identical. No LLM call.
    """
    det = spec.deterministic
    scene = det.scene
    assert scene is not None
    family = select_family(spec)
    comp = det.composition
    if comp is not None:
        # v3 explicit composition: prefer it over derivation. Values pass
        # through byte-identical; order kept; no markdown parsing, no LLM.
        v3_callouts = [
            LessonCallout(heading=c.label.strip(), body=_clean(c.value))
            for c in comp.callouts
        ]
        v3_reasoning = "; ".join(
            (f"{s.expression.strip()} — {_clean(s.explanation)}" if _clean(s.explanation) else s.expression.strip())
            for s in comp.reasoning
        )
        return LessonContent(
            title=_clean(comp.title) or _clean(det.title) or "Key concept",
            subtitle=_clean(comp.framing) or _clean(spec.concept),
            hero_svg=hero_svg,
            hero_kind=family,
            callouts=v3_callouts,
            reasoning=v3_reasoning,
            result_text=(comp.result.expression.strip() if comp.result is not None else ""),
            takeaway=_clean(comp.takeaway) or _clean(scene.caption),
        )
    callouts: list[LessonCallout] = []
    reasoning_bits: list[str] = []
    result_text = ""
    if family == "vector":
        force = scene.force
        assert force is not None
        for v in force.vectors:
            if (v.label or "").strip():
                callouts.append(LessonCallout(heading=v.label.strip(), body=_clean(v.caption)))
        for a in force.angles:
            reasoning_bits.append(_clean(a.caption) or f"Angle {a.label.strip()} between {', '.join(a.between)}")
        for arc in force.arcs:
            reasoning_bits.append(_clean(arc.caption) or f"{arc.label.strip()} {arc.direction}")
        if force.relation is not None and (force.relation.expression or "").strip():
            result_text = force.relation.expression.strip()
            if (force.relation.caption or "").strip():
                reasoning_bits.append(force.relation.caption.strip())
    elif family == "coordinate":
        plot = scene.plot
        assert plot is not None
        for c in plot.curves:
            if (c.label or "").strip():
                callouts.append(LessonCallout(heading=c.label.strip(), body=""))
        if plot.x_label or plot.y_label:
            reasoning_bits.append(f"Axes: {plot.x_label} × {plot.y_label}".strip())
    elif family == "flow":
        flow = scene.flow
        assert flow is not None
        for nd in flow.nodes:
            if (nd.label or "").strip():
                callouts.append(LessonCallout(heading=nd.label.strip(), body=""))
        for c in flow.connectors:
            if (c.label or "").strip():
                reasoning_bits.append(c.label.strip())
        if flow.relation is not None and (flow.relation.expression or "").strip():
            result_text = flow.relation.expression.strip()
    else:  # generic
        gen = scene.generic
        assert gen is not None
        for c in (gen.callouts or [])[:MAX_CALLOUTS]:
            if c.strip():
                callouts.append(LessonCallout(heading=c.strip(), body=""))
    callouts = callouts[:MAX_CALLOUTS]
    reasoning = "; ".join(b for b in reasoning_bits if b)
    return LessonContent(
        title=_clean(det.title) or "Key concept",
        subtitle=_clean(spec.concept),
        hero_svg=hero_svg,
        hero_kind=family,
        callouts=callouts,
        reasoning=reasoning,
        result_text=result_text,
        takeaway=_clean(scene.caption),
    )


_ESCAPE_CHARS = frozenset("\\#$%*_{}@`<>[]")

_ESCAPE_DOC = "Escape Typst markup chars; parens/commas/dots pass through raw."


def _typst_esc(text: str) -> str:
    return "".join(("\\" + ch) if ch in _ESCAPE_CHARS else ch for ch in text)


def build_lesson_typst(content: LessonContent, hero_filename: str = "hero.svg") -> str:
    """Render the composition as Typst source. Deterministic: same content in,
    same source out. No pixels, no subject branches — layout driven by counts.
    """
    n = len(content.callouts)
    cols = f"({'1fr, ' * max(n - 1, 0)}1fr)" if n else "(1fr)"
    chips = "\n".join(
        f'  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 6pt)'
        f'[#align(center)[#text(weight: "bold")[{_typst_esc(c.heading)}]'
        + (f' \\ {_typst_esc(c.body)}' if c.body else '')
        + ']],'
        for c in content.callouts
    )
    reasoning_block = ""
    if content.reasoning:
        reasoning_block = (
            '#v(8pt)\n#rect(width: 100%, fill: rgb("#fef9c3"), stroke: rgb("#facc15"), '
            f'radius: 6pt, inset: 8pt)[#text(weight: "bold")[Reasoning:] {_typst_esc(content.reasoning)}]\n'
        )
    result_block = ""
    if content.result_text:
        result_block = (
            '#v(6pt)\n#align(center)[#rect(fill: rgb("#facc15"), stroke: none, radius: 6pt, inset: 8pt)'
            f'[#text(weight: "bold", size: 13pt)[{_typst_esc(content.result_text)}]]]\n'
        )
    takeaway_block = ""
    if content.takeaway:
        takeaway_block = f'#align(left)[#text(size: 9pt, fill: rgb("#64748b"))[{_typst_esc(content.takeaway)}]]\n'
    subtitle_block = ""
    if content.subtitle:
        subtitle_block = f'#v(4pt)\n#align(center)[#text(size: 10pt, fill: rgb("#64748b"))[{_typst_esc(content.subtitle)}]]\n#v(6pt)\n'
    callouts_block = ""
    if content.callouts:
        callouts_block = f"#grid(columns: {cols}, gutter: 8pt,\n{chips}\n)\n"
    return (
        '#set page(width: 800pt, height: 900pt, margin: 18pt)\n'
        '#set text(size: 10pt)\n\n'
        f'#align(center)[#text(size: 18pt, weight: "bold")[{_typst_esc(content.title)}]]\n'
        f'{subtitle_block}'
        '#v(6pt)\n'
        f'#figure(\n  image("{hero_filename}", width: 68%),\n'
        f'  caption: [{_typst_esc(content.takeaway) if content.takeaway else _typst_esc(content.subtitle)}],\n'
        f') <lesson-hero>\n\n'
        f'{callouts_block}'
        f'{reasoning_block}'
        f'{result_block}'
        f'{takeaway_block}'
    )


def compile_typst(
    source: str,
    out_format: str = "pdf",
    timeout: int = TYMST_TIMEOUT_SECONDS,
    extra_files: dict[str, str] | None = None,
) -> bytes:
    """Compile Typst source with the official CLI. Raises TypstUnavailable /
    TypstCompileError — never returns a partial visual.

    `extra_files` maps filenames (e.g. ``{"hero.svg": "<svg/>"}``) to contents;
    they are written next to ``in.typ`` so ``image("hero.svg")`` resolves.
    """
    binary = shutil.which("typst")
    if binary is None:
        raise TypstUnavailable("typst CLI not found on PATH")
    with tempfile.TemporaryDirectory() as td:
        inp = str(Path(td) / "in.typ")
        out = str(Path(td) / f"out.{out_format}")
        Path(inp).write_text(source, encoding="utf-8")
        for filename, contents in (extra_files or {}).items():
            target = Path(td) / filename
            if target.parent != Path(td):
                raise TypstCompileError(f"refusing to write outside build dir: {filename!r}")
            target.write_text(contents, encoding="utf-8")
        try:
            proc = subprocess.run(
                [binary, "compile", inp, out, "--format", out_format],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise TypstCompileError(f"typst compile timed out after {timeout}s") from e
        if proc.returncode != 0:
            raise TypstCompileError(f"typst failed: {(proc.stderr or '')[:400]}")
        data = Path(out).read_bytes() if Path(out).exists() else b""
        if not data:
            raise TypstCompileError("typst produced no output")
        return data


def render_visual_lesson(spec: VisualSpec, out_format: str = "pdf") -> RenderedLesson:
    """Production composition contract.

    Validates semantics -> selects family -> renders bare hero geometry ->
    extracts structured content -> composes with Typst. Falls back to the bare
    hero SVG (honest, never fake) when Typst is unavailable or fails.
    """
    t0 = time.perf_counter()
    errors = validate_lesson_spec(spec)
    if errors:
        raise LessonValidationError("; ".join(errors))
    family = select_family(spec)
    t_hero = time.perf_counter()
    hero_svg = render_hero_geometry(spec.deterministic)
    ms_hero = (time.perf_counter() - t_hero) * 1000
    if not hero_svg.strip():
        raise LessonValidationError("hero geometry is empty")
    content = extract_lesson_content(spec, hero_svg=hero_svg)
    t_compose = time.perf_counter()
    source = build_lesson_typst(content)
    warnings: list[str] = []
    try:
        data = compile_typst(source, out_format, extra_files={"hero.svg": hero_svg})
        fallback = False
    except (TypstUnavailable, TypstCompileError) as e:
        warnings.append(str(e))
        logger.warning("visual_lesson falling back to bare hero: %s", e)
        data = hero_svg.encode("utf-8")
        out_format = "svg"
        fallback = True
    ms_compose = (time.perf_counter() - t_compose) * 1000
    return RenderedLesson(
        out_format=out_format,
        data=data,
        hero_kind=family,
        ms_total=(time.perf_counter() - t0) * 1000,
        ms_hero=ms_hero,
        ms_compose=ms_compose,
        warnings=warnings,
        fallback_used=fallback,
    )


async def render_v3_visual(spec: VisualSpec) -> tuple[str, str | bytes] | None:
    """v3 composition path for the Explain Visually route.

    Same return shape as the legacy ``generate_visual``: ``("svg", svg)`` for
    the honest bare-hero fallback, ``("png", bytes)`` for a composed lesson.
    Returns None (like the legacy dispatcher) when the spec fails semantic
    validation or nothing usable was produced, so the caller reports the same
    honest unavailable state. Logs the reason. Touches no credits.
    """
    errors = validate_lesson_spec(spec)
    if errors:
        logger.warning("v3 composition refused (semantic validation): %s", "; ".join(errors))
        return None
    try:
        lesson = await asyncio.to_thread(render_visual_lesson, spec, "png")
    except LessonValidationError as e:
        logger.warning("v3 composition refused at render: %s", e)
        return None
    if lesson.fallback_used:
        logger.info(
            "v3 composition fell back to bare hero svg (family=%s, warnings=%s)",
            lesson.hero_kind, lesson.warnings,
        )
        return ("svg", lesson.data.decode("utf-8"))
    logger.info(
        "v3 composition rendered (%s, %d bytes, total=%.1fms hero=%.1f compose=%.1f)",
        lesson.out_format, len(lesson.data), lesson.ms_total, lesson.ms_hero, lesson.ms_compose,
    )
    return ("png", lesson.data)
