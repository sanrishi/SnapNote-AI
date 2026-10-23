"""Deterministic SVG renderer for the Explain Visually educational visual.

Hybrid architecture rule: generative AI is used for visual creativity, never
for exact mathematical typography. When exact text/symbols/relationships are
the payload, this renderer turns a bounded, structured spec (DeterministicVisual)
into a clean, flat, sanitized SVG.

Gemini decides WHAT to show (universal educational primitives: objects,
vectors, angles, arcs, relations, process boxes, connectors). Code decides HOW
it is drawn: ALL layout, coordinates, arrowheads, and typography are computed
here. The same spec ALWAYS produces byte-identical SVG: no randomness, no time,
no font measurement, no LLM.

Design constraints honoured (svg_safe allowlist):
  - No <marker> (sanitizer renames marker-end to markerEnd and it silently
    fails). Arrowheads are drawn as <polygon>.
  - No <style> block (stripped by the sanitizer). All styling is inline.
  - Every string is XML-escaped before being embedded.
"""

from __future__ import annotations

import logging
import math
import re
from fractions import Fraction
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from app.models.schemas import (
    ComparisonPanel,
    ComparisonPoint,
    DeterministicVisual,
    FlowConnector,
    FlowNode,
    ForceDiagram,
    VisualAngle,
    VisualComparison,
    VisualCurve,
    VisualGeneric,
    VisualInterval,
    VisualNumberLineTrack,
    VisualObject,
    VisualPlot,
    VisualScene,
    VisualVector,
)

if TYPE_CHECKING:
    from app.utils.math_intervals import IntervalSet
    from app.utils.math_verify import MathCase, SolveResult
from app.utils.svg_safe import sanitize_svg

logger = logging.getLogger(__name__)

VIEW_W = 800
MARGIN = 48
CONTENT_W = VIEW_W - 2 * MARGIN

INK = "#1e293b"
MUTED = "#64748b"
SOFT = "#94a3b8"
CARD_FILL = "#f8fafc"
CARD_STROKE = "#e2e8f0"
ACCENT = "#6366f1"
GREEN = "#16a34a"
RED = "#dc2626"
BG_STAGE = "#f8fafc"
STAGE_STROKE = "#e2e8f0"

# Visual design tokens (readability contract — enforced by the min-font and
# density regression tests). Sizes are final-rendered px at 800px width.
# Geometry labels never go below 16, ticks never below 14, titles sit at 32.
TYPE_TITLE = 32
TYPE_PANEL = 22
TYPE_LABEL = 20  # vector/object/point labels
TYPE_EQUATION = 24
TYPE_TICK = 14
TYPE_AXIS = 16
TYPE_SECONDARY = 14  # legend/caption/notes/connector labels
TYPE_CURVE = 16  # long curve/vertex labels


def _halo(x: float, y: float, w: float, h: float, align: str = "center") -> str:
    """Opaque label background so text over geometry stays readable.

    `align` matches the text anchor: a start-anchored label begins at x, so
    its halo must begin there too (centering the halo would paint over the
    vector line or the zero-velocity dot it is supposed to annotate).
    """
    if align == "start":
        x0 = x - 7
    elif align == "end":
        x0 = x - w + 7
    else:
        x0 = x - w / 2
    return (
        f'<rect x="{_f(x0)}" y="{_f(y - h / 2)}" width="{_f(w)}" height="{_f(h)}" rx="7" '
        f'fill="white" opacity="0.88"/>'
    )

# Diagram stage geometry (force diagram) — enlarged ~22% for composition (was 470).
STAGE_X = 48
STAGE_Y = 96
STAGE_W = VIEW_W - 2 * STAGE_X
STAGE_H = 560
PIVOT_X = 240
PIVOT_Y = 520
BASE_LEN = 215
MIN_R = 12

_FONT = "sans-serif"

_VECTOR_COLORS = {
    "": INK,
    "accent": ACCENT,
    "indigo": ACCENT,
    "red": RED,
    "green": GREEN,
}


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _f(value: float) -> str:
    return f"{value:.1f}"


def _wrap(
    text: str, max_chars: int, hyphenate: bool = False, break_long_words: bool = True
) -> list[str]:
    """Deterministic word-wrap (no font metrics). Breaks long tokens when needed.

    hyphenate=True marks mid-word breaks with a visible "-" (narrow flow
    boxes): "Proportional" -> ["Proporti-", "onal"]. Spec values are never
    altered — the hyphen is presentation only, like newspaper columns.

    break_long_words=False keeps every word whole (observed flow defect:
    "Proportion"/"al" split across lines); the fit search then measures the
    unbroken word and shrinks the font instead of fragmenting the label.
    """
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for word in words:
        while break_long_words and len(word) > max_chars:
            if cur:
                lines.append(cur)
                cur = ""
            chunk_len = max_chars if not hyphenate else max_chars - 1
            chunk_len = max(1, chunk_len)
            piece, word = word[:chunk_len], word[chunk_len:]
            lines.append(piece + "-" if hyphenate and word else piece)
        candidate = f"{cur} {word}".strip() if cur else word
        if len(candidate) > max_chars:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines or [""]


def _title(y: int, text: str) -> tuple[str, int]:
    return f'<text x="{VIEW_W/2:.0f}" y="{y}" font-family="{_FONT}" font-size="{TYPE_TITLE}" font-weight="700" fill="{INK}" text-anchor="middle">{_esc(text)}</text>', y + 50


def _section_header(y: int, text: str) -> tuple[str, int]:
    return (
        f'<text x="{MARGIN}" y="{y}" font-family="{_FONT}" font-size="{TYPE_SECONDARY}" font-weight="700" '
        f'fill="{ACCENT}" text-anchor="start" letter-spacing="1">{_esc(text)}</text>',
        y + 26,
    )


def _equation_card(y: int, expression: str, meaning: str) -> tuple[str, int]:
    """One equation card: boxed expression + its meaning below."""
    expr_lines = _wrap(expression, 42)
    meaning_lines = _wrap(meaning, 76) if meaning else []
    expr_h = 30 * len(expr_lines)
    meaning_h = 18 * len(meaning_lines) if meaning_lines else 0
    card_h = 30 + expr_h + meaning_h
    height = max(64, card_h)

    parts = [
        f'<rect x="{MARGIN}" y="{y}" width="{CONTENT_W}" height="{height}" rx="10" fill="{CARD_FILL}" stroke="{CARD_STROKE}" stroke-width="1.5"/>',
    ]
    ty = y + 30
    for line in expr_lines:
        parts.append(
            f'<text x="{MARGIN + 20}" y="{ty}" font-family="{_FONT}" font-size="{TYPE_EQUATION}" font-weight="700" '
            f'fill="{INK}" text-anchor="start">{_esc(line)}</text>'
        )
        ty += 30
    for line in meaning_lines:
        parts.append(
            f'<text x="{MARGIN + 20}" y="{ty}" font-family="{_FONT}" font-size="16" '
            f'fill="{MUTED}" text-anchor="start">{_esc(line)}</text>'
        )
        ty += 20
    return "".join(parts), y + height + 22


def _point_line(y: int, text: str) -> tuple[str, int]:
    lines = _wrap(text, 76)
    parts = []
    ty = y
    for i, line in enumerate(lines):
        bullet_x = MARGIN + 8
        text_x = MARGIN + 30
        if i == 0:
            parts.append(f'<circle cx="{bullet_x}" cy="{ty - 5}" r="4" fill="{GREEN}"/>')
        parts.append(
            f'<text x="{text_x}" y="{ty}" font-family="{_FONT}" font-size="16" '
            f'fill="{INK}" text-anchor="start">{_esc(line)}</text>'
        )
        ty += 22
    return "".join(parts), ty + 6


# ─────────────────────────────────────────────────────────────────────────────
# Scene engine: code owns ALL geometry. Gemini supplies only semantics.
# ─────────────────────────────────────────────────────────────────────────────

def _polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    """Point on a circle. SVG y grows downward, so 0° = right, 90° = up."""
    rad = math.radians(deg)
    return (cx + r * math.cos(rad), cy - r * math.sin(rad))


def _arrowhead(x: float, y: float, deg: float, size: float = 12.0) -> str:
    """Triangle arrowhead pointing along `deg` at (x, y)."""
    rad = math.radians(deg)
    dx, dy = math.cos(rad), -math.sin(rad)
    px, py = -dy, dx
    bx = x - dx * size
    by = y - dy * size
    p1 = (x, y)
    p2 = (bx + px * size * 0.42, by + py * size * 0.42)
    p3 = (bx - px * size * 0.42, by - py * size * 0.42)
    pts = " ".join(_f(v) for xy in (p1, p2, p3) for v in xy)
    return f'<polygon points="{pts}" fill="{INK}"/>'


def _arc_points(
    cx: float, cy: float, r: float, deg_a: float, deg_b: float, steps: int = 24
) -> list[tuple[float, float]]:
    """Points along the short arc from deg_a to deg_b (radius r)."""
    diff = (deg_b - deg_a + 180.0) % 360.0 - 180.0
    pts = []
    for i in range(steps + 1):
        t = i / steps
        pts.append(_polar(cx, cy, r, deg_a + diff * t))
    return pts


def _polyline(pts: list[tuple[float, float]], stroke: str, width: float = 2.0, dash: str | None = None) -> str:
    points = " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 2.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def _vector_deg(v: VisualVector) -> float:
    return float(v.angle_deg % 360)


def _render_vector(
    v: VisualVector,
    tail: tuple[float, float],
    color: str,
    length_scale: float = 1.0,
    font_size: float = TYPE_LABEL,
    label_perp: float = 14.0,
    label_mode: str = "mid",
    max_length: float = 1.6,
) -> str:
    """Draw one vector: line + arrowhead + label at its midpoint.

    A zero-length vector is a *value*, not missing geometry: it renders a
    filled point marker with the label (e.g. ``v = 0`` at the contact point)
    and never an arrowhead. Drawing a minimum-length arrow here would print
    motion where the physics says there is none.

    ``max_length`` caps the arrow so a head-mounted long claim remains inside
    the stage; comparison panels use a slightly smaller cap for long equations.
    """
    if v.length <= 0.0:
        parts = [f'<circle cx="{_f(tail[0])}" cy="{_f(tail[1])}" r="5" fill="{color}"/>']
        if v.label:
            lx, ly = tail[0] + 14, tail[1] + 5
            parts.append(
                _halo(
                    lx,
                    ly - font_size * 0.55,
                    len(v.label) * font_size * 0.62 + 14,
                    font_size + 8,
                    align="start",
                )
            )
            parts.append(
                f'<text x="{_f(lx)}" y="{_f(ly)}" font-family="{_FONT}" font-size="{_f(font_size)}" '
                f'font-weight="700" fill="{color}" text-anchor="start">{_esc(v.label)}</text>'
            )
        return "".join(parts)

    deg = _vector_deg(v)
    length = BASE_LEN * max(0.4, min(max_length, v.length)) * length_scale
    head = _polar(tail[0], tail[1], length, deg)
    mid = _polar(tail[0], tail[1], length * 0.52, deg)
    # Label sits beside the midpoint, or beyond the arrowhead for long claims
    # (a centered "v = v_CM + Rω = 2v_CM" would cover the P' point label).
    rad = math.radians(deg)
    lx, ly = mid[0] + label_perp * math.sin(rad), mid[1] + label_perp * math.cos(rad)
    anchor = "middle"
    if label_mode == "head" and abs(math.cos(rad)) >= 0.3:
        if math.cos(rad) > 0:
            lx, anchor = head[0] + 10, "start"
        else:
            lx, anchor = head[0] - 10, "end"
        ly = head[1] + font_size * 0.35
    parts = [
        _line(tail[0], tail[1], head[0], head[1], color, width=3.2),
        _arrowhead(head[0], head[1], deg),
    ]
    if v.label:
        parts.append(
            _halo(
                lx,
                ly - font_size * 0.28,
                len(v.label) * font_size * 0.62 + 14,
                font_size + 8,
                align=anchor,
            )
        )
        parts.append(
            f'<text x="{_f(lx)}" y="{_f(ly)}" font-family="{_FONT}" font-size="{_f(font_size)}" '
            f'font-weight="700" fill="{color}" text-anchor="{anchor}">{_esc(v.label)}</text>'
        )
    return "".join(parts)


def _render_object(
    obj: VisualObject,
    x: float,
    y: float,
    label_side: str = "below",
    radius: float = 52.0,
) -> str:
    kind = obj.kind
    parts: list[str] = []
    if kind == "disk":
        parts.append(
            f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(radius)}" fill="none" '
            f'stroke="{INK}" stroke-width="2.5"/>'
        )
        parts.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="6" fill="{INK}"/>')
    elif kind == "pivot":
        parts.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="9" fill="{INK}"/>')
        parts.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="13" fill="none" stroke="{SOFT}" stroke-width="1.5"/>')
    elif kind == "block":
        parts.append(
            f'<rect x="{_f(x - 26)}" y="{_f(y - 18)}" width="52" height="36" rx="6" '
            f'fill="{CARD_FILL}" stroke="{INK}" stroke-width="2"/>'
        )
    else:  # point
        parts.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="6" fill="{INK}"/>')
    if obj.label:
        if label_side == "left":
            tx, ty, anchor = x - 16, y + 5, "end"
        else:
            tx, ty, anchor = x, y + 38, "middle"
        parts.append(
            f'<text x="{_f(tx)}" y="{_f(ty)}" font-family="{_FONT}" font-size="16" '
            f'font-weight="700" fill="{INK}" text-anchor="{anchor}">{_esc(obj.label)}</text>'
        )
    return "".join(parts)


def _angle_arc(
    angle: VisualAngle,
    vertex: tuple[float, float],
    dirs: list[float],
) -> str:
    """Arc between two ray directions at `vertex`, labeled, with arrowhead.

    Only the small label sits near the arc. Long captions must NOT be dropped
    inline (a wide text block placed at the shared vertex reliably overlaps the
    vector geometry); they are collected by the caller into the legend block
    below the stage instead.
    """
    if len(dirs) < 2:
        return ""
    a, b = dirs[0], dirs[1]
    diff = (b - a + 180.0) % 360.0 - 180.0
    if abs(diff) < 4.0:
        return ""
    r_arc = 34.0
    pts = _arc_points(vertex[0], vertex[1], r_arc, a, b)
    mid_deg = a + diff * 0.5
    label_pos = _polar(vertex[0], vertex[1], r_arc + 18, mid_deg)
    tip = pts[-1]
    tip_deg = a + diff
    parts = [
        _polyline(pts, ACCENT, width=2.2),
        _arrowhead(tip[0], tip[1], tip_deg, 10),
    ]
    if angle.label:
        parts.append(_halo(label_pos[0], label_pos[1] - 6, len(angle.label) * 11.0 + 14, 24))
        parts.append(
            f'<text x="{_f(label_pos[0])}" y="{_f(label_pos[1])}" font-family="{_FONT}" '
            f'font-size="18" font-weight="700" fill="{ACCENT}" text-anchor="middle">{_esc(angle.label)}</text>'
        )
    return "".join(parts)


def _rotation_arc(
    arc: VisualArc,
    around: tuple[float, float],
    label_deg: float | None = None,
    show_label: bool = True,
) -> str:
    """Curved rotation arrow around a point (e.g. torque ↻)."""
    r = 30.0
    start = 220.0 if arc.direction == "ccw" else -40.0
    sweep = 240.0 if arc.direction == "ccw" else -240.0
    pts = _arc_points(around[0], around[1], r, start, start + sweep, steps=28)
    tip = pts[-1]
    tip_deg = start + sweep
    # Explicit label angle (collision avoidance) or the arc midpoint.
    mid_deg = start + sweep * 0.5 if label_deg is None else label_deg
    parts = [
        _polyline(pts, GREEN, width=2.6),
        _arrowhead(tip[0], tip[1], tip_deg, 11),
    ]
    if arc.label and show_label:
        # Label sits OUTSIDE the arc ring (radius r+24 along the arc midpoint),
        # never on top of the arc line itself.
        lpos = _polar(around[0], around[1], r + 24, mid_deg)
        parts.append(_halo(lpos[0], lpos[1] - 6, len(arc.label) * 11.0 + 14, 24))
        parts.append(
            f'<text x="{_f(lpos[0])}" y="{_f(lpos[1])}" font-family="{_FONT}" font-size="18" '
            f'font-weight="700" fill="{GREEN}" text-anchor="middle">{_esc(arc.label)}</text>'
        )
    return "".join(parts)


def _legend_line(y: int, label: str, text: str) -> str:
    """One muted legend line (label chip + explanation) below the stage."""
    parts = []
    if label:
        parts.append(
            f'<text x="{MARGIN}" y="{y}" font-family="{_FONT}" font-size="{TYPE_SECONDARY}" font-weight="700" '
            f'fill="{ACCENT}" text-anchor="start">{_esc(label)}</text>'
        )
        tx = MARGIN + 30
    else:
        tx = MARGIN
    parts.append(
        f'<text x="{tx}" y="{y}" font-family="{_FONT}" font-size="{TYPE_SECONDARY}" fill="{MUTED}" '
        f'text-anchor="start">{_esc(text)}</text>'
    )
    return "".join(parts)


def _render_force_diagram(scene: VisualScene) -> tuple[str, int, list[str], VisualRelation | None]:
    """Render the force/vector diagram inside the stage.

    Returns (svg, next_y, legend_lines, relation). Long angle/arc captions are
    NOT drawn inline (they would overlap the vector geometry); they are
    returned as legend lines the caller renders below the stage. The relation
    card is returned separately so the caller can place it after the legend.
    """
    force: ForceDiagram | None = scene.force
    parts: list[str] = []
    legend: list[str] = []
    if force is None:
        return "", STAGE_Y + STAGE_H + 24, legend, None

    pivot = (PIVOT_X, PIVOT_Y)

    # Vector geometry.
    vectors = [v for v in force.vectors if v.label.strip()]
    heads: dict[str, tuple[float, float]] = {}
    drawn: list[str] = []
    for v in vectors:
        color = _VECTOR_COLORS.get(v.color, INK)
        if v.tail.strip() and v.tail.strip() in heads:
            tail = heads[v.tail.strip()]
        else:
            tail = pivot
        head = _polar(tail[0], tail[1], BASE_LEN * max(0.4, min(1.6, v.length)), _vector_deg(v))
        heads[v.label.strip()] = head
        drawn.append(_render_vector(v, tail, color))
    parts.extend(drawn)

    # Objects drawn under vectors (so arrows overlay).
    obj_html = _render_object(force.object, pivot[0], pivot[1])
    parts.insert(0, obj_html)

    # Ground/contact line: rolling and friction scenes need the surface the
    # object rolls/rests on, otherwise the body reads as floating. Code-owned
    # geometry: baseline + hatch ticks + contact dot at the tangent point.
    if force.ground:
        drop = {"disk": 56.0, "block": 24.0}.get(force.object.kind, 20.0)
        gy = PIVOT_Y + drop
        gx0, gx1 = STAGE_X + 24, STAGE_X + STAGE_W - 24
        ground_parts = [f'<line x1="{_f(gx0)}" y1="{_f(gy)}" x2="{_f(gx1)}" y2="{_f(gy)}" stroke="{INK}" stroke-width="2.5"/>']
        x = gx0 + 4
        while x + 12 < gx1 - 4:
            ground_parts.append(_line(x, gy, x + 12, gy + 9, SOFT, width=1.5))
            x += 22
        if force.object.kind == "disk":
            ground_parts.append(f'<circle cx="{_f(PIVOT_X)}" cy="{_f(gy)}" r="4.5" fill="{ACCENT}"/>')
        ground_parts.append(
            f'<text x="{_f(gx0)}" y="{_f(gy + 26)}" font-family="{_FONT}" font-size="15" '
            f'fill="{MUTED}" text-anchor="start">surface</text>'
        )
        parts.append("".join(ground_parts))

    # Angle arcs: vertex is the shared head of the referenced vectors, else pivot.
    for angle in force.angles:
        dirs: list[float] = []
        vertex = pivot
        for lab in angle.between:
            for v in vectors:
                if v.label.strip() == lab.strip():
                    if v.tail.strip() and v.tail.strip() in heads:
                        vertex = heads[v.tail.strip()]
                    dirs.append(_vector_deg(v))
                    break
        parts.append(_angle_arc(angle, vertex, dirs))
        if angle.caption.strip():
            legend.append(f"{angle.label.strip()} — {angle.caption.strip()}")

    # Rotation arcs around the pivot.
    for arc in force.arcs:
        parts.append(_rotation_arc(arc, pivot))
        if arc.caption.strip():
            legend.append(f"{arc.label.strip()} — {arc.caption.strip()}")

    return "".join(parts), STAGE_Y + STAGE_H + 18, legend, force.relation


def _render_flow(scene: VisualScene) -> tuple[str, int, list[str], VisualRelation | None]:
    """Render a process-flow scene: labeled boxes + arrows (+ optional feedback loop)."""
    flow = scene.flow
    parts: list[str] = []
    if flow is None:
        return "", STAGE_Y + STAGE_H + 24, [], None
    nodes: list[FlowNode] = [n for n in flow.nodes if n.label.strip()]
    if not nodes:
        return "", STAGE_Y + STAGE_H + 24, [], None

    n = len(nodes)
    # Fit guarantee (R1): boxes size from their own text and the row ALWAYS
    # fits. Mid-word hyphenation is banned (observed: "Proport-ional",
    # "Summati-on" — unreadable); instead the search picks the first
    # (font size, wrap width) pair that fits, shrinking the FONT as the last
    # resort. Pure character-count arithmetic — no font metrics, stays
    # byte-deterministic. Mean glyph advance 0.64 for 600-weight sans was
    # measured against the real PID board (0.58 let 8-char lines spill).
    gap = 30 if n <= 4 else (20 if n <= 6 else (12 if n <= 7 else 10))
    available = (CONTENT_W - 40 - (n - 1) * gap) / n
    font_size = 12
    wrapped: list[list[str]] = [_wrap(node.label, 10) for node in nodes]
    # Width preference OUTSIDE font preference: pick the widest wrap that fits
    # at some readable font, then the largest font for that width. Reversing
    # the loops let 14px win at a 6-char wrap, fragmenting every label.
    inner_w = min(available, 170.0)
    for max_chars in (18, 16, 14, 12, 10, 8):
        for size in (14, 13, 12, 11, 10, 9, 8):
            trial = [
                _wrap(node.label, max_chars, break_long_words=False) for node in nodes
            ]
            longest = max(len(line) for lines in trial for line in lines)
            if longest * size * 0.64 + 6 <= inner_w:
                wrapped, font_size = trial, size
                break
        else:
            continue
        break
    else:
        # A monster token cannot fit unbroken at any readable size. Split
        # ONLY the labels that overflow: one over-long word must never mangle
        # every other label (observed on the real PID row, where a 13-char
        # word missed the 8px fit by 0.06px and all 8 labels broke mid-word).
        font_size = 8
        chunk = max(6, int((inner_w - 6) / (font_size * 0.64)))
        wrapped = []
        for node in nodes:
            lines = _wrap(node.label, chunk, break_long_words=False)
            widest = max((len(line) for line in lines), default=0)
            if widest > chunk:
                lines = _wrap(node.label, chunk)
            wrapped.append(lines)
    box_w = min(170.0, available)
    max_lines = max(len(lines) for lines in wrapped)
    box_h = max(54, 20 + 22 * max_lines)
    y_center = STAGE_Y + STAGE_H // 2 - 40
    total_w = n * box_w + (n - 1) * gap
    x0 = (VIEW_W - total_w) / 2

    centers: list[tuple[float, float]] = []
    for i, node in enumerate(nodes):
        cx = x0 + i * (box_w + gap) + box_w / 2
        centers.append((cx, y_center))
        lines = wrapped[i]
        bx = cx - box_w / 2
        by = y_center - box_h / 2
        parts.append(
            f'<rect x="{_f(bx)}" y="{_f(by)}" width="{_f(box_w)}" height="{box_h}" rx="10" '
            f'fill="{CARD_FILL}" stroke="{ACCENT}" stroke-width="1.5"/>'
        )
        ty = y_center - (len(lines) - 1) * 11
        for line in lines:
            parts.append(
                f'<text x="{_f(cx)}" y="{_f(ty)}" font-family="{_FONT}" font-size="{font_size}" '
                f'font-weight="600" fill="{INK}" text-anchor="middle">{_esc(line)}</text>'
            )
            ty += 22

    y_top = y_center - box_h / 2
    y_bot = y_center + box_h / 2
    # Connectors (arrows between boxes). Exact-duplicate edges are drawn once
    # (R4): overlapping specs otherwise stack identical arrowheads.
    seen_edges: set[tuple[int, int, bool]] = set()
    elbow_index = 0
    # Spread multiple elbow landings on the same target so their arrowheads
    # never stack on one point (R4).
    landings: dict[int, int] = {}
    bottom = y_bot + 74
    node_labels = [" ".join(node.label.split()).casefold() for node in nodes]

    def _echo_label(label: str, conn: FlowConnector) -> str:
        """Drop connector labels that merely restate a node's own label.

        Observed: node "Error e(t)" + connector label "e(t)" + chip "Error"
        printed the same symbol around one arrow repeatedly. A label already
        contained in its source/target node text adds no information.
        """
        text = " ".join(label.split()).casefold()
        if not text:
            return ""
        for idx in (conn.source, conn.target):
            if text in node_labels[idx]:
                return ""
        return label

    for conn in flow.connectors:
        if not (0 <= conn.source < n and 0 <= conn.target < n):
            continue
        key = (conn.source, conn.target, bool(conn.feedback))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        conn_label = _echo_label(conn.label, conn)
        s = centers[conn.source]
        t = centers[conn.target]
        if conn.feedback:
            # V return path below the row, clamped to the stage (R3).
            lo_x = STAGE_X + 8
            hi_x = STAGE_X + STAGE_W - 8
            sx = max(lo_x, min(hi_x, s[0]))
            tx = max(lo_x, min(hi_x, t[0]))
            vy = y_bot + 52
            pts = [(sx, y_bot + 8), ((sx + tx) / 2, vy), (tx, y_bot + 8)]
            path_pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)
            parts.append(
                f'<polyline points="{path_pts}" fill="none" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>'
            )
            if conn_label:
                # Label BELOW the V point — never inside the node band (R2).
                parts.append(
                    f'<text x="{_f((sx + tx) / 2)}" y="{_f(vy + 18)}" '
                    f'font-family="{_FONT}" font-size="14" fill="{RED}" text-anchor="middle">{_esc(conn_label)}</text>'
                )
            bottom = max(bottom, vy + 34)
        elif abs(conn.target - conn.source) == 1:
            from_x, to_x = s[0], t[0]
            arrow_deg = 0.0 if to_x >= from_x else 180.0
            y = y_center
            parts.append(_line(from_x + box_w / 2, y, to_x - box_w / 2, y, ACCENT, width=2.2))
            tip_x = to_x - box_w / 2 if arrow_deg == 0.0 else to_x + box_w / 2
            parts.append(_arrowhead(tip_x, y, arrow_deg, 11))
            if conn_label:
                # Label ABOVE the boxes — the gap midpoint sits inside the
                # node band, so centering there renders through boxes (R2).
                parts.append(
                    f'<text x="{_f((from_x + to_x) / 2)}" y="{_f(y_top - 10)}" font-family="{_FONT}" font-size="14" '
                    f'fill="{MUTED}" text-anchor="middle">{_esc(conn_label)}</text>'
                )
        else:
            # Non-adjacent edge (fan-out/fan-in/backward): elbow routed ABOVE
            # the row so the line never crosses intermediate boxes. Heights
            # stagger deterministically; landings on one target spread out.
            route_y = y_top - 26 - 12 * elbow_index
            elbow_index += 1
            j = landings.get(conn.target, 0)
            landings[conn.target] = j + 1
            # Offset by running index — deterministic;exact group sizes are
            # unknowable in one pass, and groups stay tiny in practice.
            land_x = t[0] + (j * 14 if conn.target > conn.source else -j * 14)
            ex = s[0]
            seg = (
                f'<polyline points="{_f(ex)},{_f(y_top)} {_f(ex)},{_f(route_y)} '
                f'{_f(land_x)},{_f(route_y)} {_f(land_x)},{_f(y_top)}" '
                f'fill="none" stroke="{ACCENT}" stroke-width="2"/>'
            )
            parts.append(seg)
            parts.append(_arrowhead(land_x, y_top, 270, 11))
            if conn_label:
                parts.append(
                    f'<text x="{_f((ex + land_x) / 2)}" y="{_f(route_y - 6)}" font-family="{_FONT}" '
                    f'font-size="14" fill="{MUTED}" text-anchor="middle">{_esc(conn_label)}</text>'
                )

    # Relation card below (returned for the caller to place after the legend).
    return "".join(parts), bottom, [], flow.relation


# ── Plot renderer: pure-SVG axes + grid + curves (no matplotlib, stays deterministic) ──

_ALLOWED_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
                  "asin": math.asin, "acos": math.acos, "atan": math.atan,
                  "exp": math.exp, "log": math.log, "log10": math.log10, "fabs": math.fabs, "abs": abs}


def _safe_eval(expr: str, x_val: float) -> float | None:
    """Safe eval of a math expr in x (no exec). Returns None on error or non-finite."""
    import ast as _ast
    if not expr or not expr.strip():
        return None
    py = expr.strip().replace("^", "**").replace("**", "**")
    # normalize unicode
    py = py.replace("×", "*").replace("·", "*").replace("−", "-")
    py = py.replace("π", "pi")
    try:
        tree = _ast.parse(py, mode="eval")
    except SyntaxError:
        return None

    def _ev(node) -> float | None:
        if isinstance(node, _ast.Expression):
            return _ev(node.body)
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, _ast.Name):
            if node.id == "x":
                return float(x_val)
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
            return None
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.UAdd, _ast.USub)):
            v = _ev(node.operand)
            if v is None:
                return None
            return v if isinstance(node.op, _ast.UAdd) else -v
        if isinstance(node, _ast.BinOp):
            l, r = _ev(node.left), _ev(node.right)
            if l is None or r is None:
                return None
            if isinstance(node.op, _ast.Add):
                return l + r
            if isinstance(node.op, _ast.Sub):
                return l - r
            if isinstance(node.op, _ast.Mult):
                return l * r
            if isinstance(node.op, _ast.Div):
                return None if abs(r) < 1e-12 else l / r
            if isinstance(node.op, _ast.Pow):
                try:
                    return float(l ** r)
                except Exception:
                    return None
            return None
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            if node.func.id not in _ALLOWED_FUNCS or len(node.args) != 1 or node.keywords:
                return None
            v = _ev(node.args[0])
            if v is None:
                return None
            try:
                return float(_ALLOWED_FUNCS[node.func.id](v))
            except Exception:
                return None
        return None
    try:
        v = _ev(tree)
        if v is None or not math.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _polar_radius(curve: VisualCurve, x_min: float, x_max: float) -> float | None:
    """Constant polar radius for labels like "r = 1" / "r = √5", else None.

    Only fires when the LABEL declares a radial bound AND the evaluated
    expression is constant across the range — a genuine r = const bound.
    Anything else (functions of x, explicit points) keeps Cartesian plotting.
    Label forms: "r = 1" as well as descriptive bounds like
    "Inner Radius g(θ) = 1" (real math-board spec).
    """
    import re

    label = (curve.label or "").strip()
    if not (re.match(r"(?i)^r\s*=", label) or re.search(r"(?i)radius", label)):
        return None
    if curve.points:
        return None
    expr = (curve.expr or "").strip()
    if not expr:
        return None
    lo = float(curve.x_min) if curve.x_min != 0 or curve.x_max != 0 else x_min
    hi = float(curve.x_max) if curve.x_max != 0 or curve.x_min != 0 else x_max
    if hi <= lo:
        return None
    vals = [_safe_eval(expr, lo + (hi - lo) * i / 9) for i in range(10)]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if len(vals) < 10 or max(vals) - min(vals) > 1e-9:
        return None
    radius = vals[0]
    if radius <= 0:
        return None
    return radius


def _curve_points(curve: VisualCurve, x_min: float, x_max: float) -> list[tuple[float, float]]:
    # explicit points take precedence (already in data coords)
    if curve.points:
        pts: list[tuple[float, float]] = []
        for p in curve.points:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except Exception:
                    continue
        if pts:
            return pts
    # expr path: sample 80 points across its own x_min/x_max (falls back to plot range)
    expr = (curve.expr or "").strip()
    if not expr:
        return []
    lo = float(curve.x_min) if curve.x_min != 0 or curve.x_max != 0 else x_min
    hi = float(curve.x_max) if curve.x_max != 0 or curve.x_min != 0 else x_max
    # if curve has its own range, use it; otherwise use plot range
    if curve.x_min == 0 and curve.x_max == 5.0 and (x_min != 0 or x_max != 5.0):
        lo, hi = x_min, x_max
    elif curve.x_min == 0 and curve.x_max == 0:
        lo, hi = x_min, x_max
    if hi <= lo:
        return []
    n = 80
    pts = []
    for i in range(n):
        xv = lo + (hi - lo) * i / (n - 1)
        yv = _safe_eval(expr, xv)
        if yv is not None and math.isfinite(yv):
            pts.append((xv, yv))
    return pts


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 0.0) -> bool:
    """Do two (x, y, w, h) boxes overlap (with optional padding)? Deterministic."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or ay + ah + pad <= by or by + bh + pad <= ay)


def _fit_polyline_plot(plot: VisualPlot) -> tuple[float, float, float, float] | None:
    """Tighten a data-only plot's window around its points.

    Expression curves need their declared window (asymptotes, branch shape).
    Polyline/marker plots (Argand squares, geometry constructions) do not, and
    a fixed wide window leaves the actual figure stranded in one quadrant.
    Origin stays visible; a small outward margin keeps labels off the border.
    """
    if any((curve.expr or "").strip() for curve in plot.curves):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for curve in plot.curves:
        for point in curve.points or []:
            if len(point) < 2:
                continue
            px, py = float(point[0]), float(point[1])
            if math.isfinite(px) and math.isfinite(py):
                xs.append(px)
                ys.append(py)
    if not xs or not ys:
        return None

    def fit(values: list[float]) -> tuple[float, float]:
        lo = min(0.0, min(values))
        hi = max(0.0, max(values))
        span = hi - lo
        if span <= 0:
            return -1.0, 1.0
        pad = max(0.2, span * 0.04)
        lo = lo - pad if lo < 0 else 0.0
        hi = hi + pad if hi > 0 else 0.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    x_lo, x_hi = fit(xs)
    y_lo, y_hi = fit(ys)
    return x_lo, x_hi, y_lo, y_hi


def _axis_ticks(lo: float, hi: float) -> list[float]:
    """Round tick values inside a window, aiming for 4-6 labels."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / 5.0
    base = 10.0 ** math.floor(math.log10(raw))
    ticks: list[float] = []
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mult * base
        start = math.ceil(lo / step - 1e-9) * step
        ticks = []
        value = start
        while value <= hi + 1e-9:
            rounded = round(value, 10)
            ticks.append(0.0 if abs(rounded) < 1e-9 else rounded)
            value += step
        if 3 <= len(ticks) <= 6:
            return ticks
    return ticks


def _tick_text(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _render_plot(scene: VisualScene) -> tuple[str, int, list[str], object]:
    plot = scene.plot
    if plot is None or not plot.curves:
        return "", STAGE_Y + STAGE_H + 24, [], None
    fitted = _fit_polyline_plot(plot)
    if fitted is not None:
        x_min, x_max, y_min, y_max = fitted
    else:
        x_min, x_max = float(plot.x_min), float(plot.x_max)
        y_min, y_max = float(plot.y_min), float(plot.y_max)
    if x_max <= x_min:
        x_min, x_max = 0.0, 5.0
    if y_max <= y_min:
        y_min, y_max = 0.0, 5.0
    # stage geometry
    x0, y0, w, h = STAGE_X, STAGE_Y, STAGE_W, STAGE_H
    pad_l, pad_b = 24, 16
    plot_x0, plot_y0, plot_w, plot_h = x0 + pad_l, y0 + 8, w - pad_l - 10, h - 8 - pad_b

    def map_x(xv: float) -> float:
        return plot_x0 + (xv - x_min) / (x_max - x_min) * plot_w if x_max != x_min else plot_x0

    def map_y(yv: float) -> float:
        return plot_y0 + plot_h - (yv - y_min) / (y_max - y_min) * plot_h if y_max != y_min else plot_y0

    parts: list[str] = []
    # grid
    if plot.show_grid:
        for i in range(1, 4):
            gx = plot_x0 + plot_w * i / 4
            gy = plot_y0 + plot_h * i / 4
            parts.append(f'<line x1="{_f(gx)}" y1="{_f(plot_y0)}" x2="{_f(gx)}" y2="{_f(plot_y0+plot_h)}" stroke="{STAGE_STROKE}" stroke-width="1" stroke-dasharray="3 3"/>')
            parts.append(f'<line x1="{_f(plot_x0)}" y1="{_f(gy)}" x2="{_f(plot_x0+plot_w)}" y2="{_f(gy)}" stroke="{STAGE_STROKE}" stroke-width="1" stroke-dasharray="3 3"/>')
    # axes (draw at 0 if inside range, else at border)
    x_axis_y = map_y(0) if y_min <= 0 <= y_max else (plot_y0 + plot_h if y_min >= 0 else plot_y0)
    y_axis_x = map_x(0) if x_min <= 0 <= x_max else plot_x0
    # clamp axes inside plot
    x_axis_y = max(plot_y0, min(plot_y0 + plot_h, x_axis_y))
    y_axis_x = max(plot_x0, min(plot_x0 + plot_w, y_axis_x))
    parts.append(f'<line x1="{_f(plot_x0)}" y1="{_f(x_axis_y)}" x2="{_f(plot_x0+plot_w)}" y2="{_f(x_axis_y)}" stroke="{INK}" stroke-width="2"/>')
    parts.append(f'<line x1="{_f(y_axis_x)}" y1="{_f(plot_y0)}" x2="{_f(y_axis_x)}" y2="{_f(plot_y0+plot_h)}" stroke="{INK}" stroke-width="1.5"/>')
    parts.append(_arrowhead(plot_x0+plot_w, x_axis_y, 0))
    parts.append(_arrowhead(y_axis_x, plot_y0, 90))
    # axis labels + ticks
    parts.append(f'<text x="{_f(plot_x0+plot_w+8)}" y="{_f(x_axis_y+4)}" font-family="{_FONT}" font-size="16" fill="{MUTED}" text-anchor="start">{_esc(plot.x_label or "x")}</text>')
    parts.append(f'<text x="{_f(y_axis_x+6)}" y="{_f(plot_y0-6)}" font-family="{_FONT}" font-size="16" fill="{MUTED}" text-anchor="start">{_esc(plot.y_label or "y")}</text>')
    # Origin tick dedupe (R5): when 0 sits inside both ranges the two 0
    # labels land on top of each other at the crossing - draw one instead.
    origin_inside = x_min < 0 < x_max and y_min < 0 < y_max
    x_ticks = _axis_ticks(x_min, x_max)
    y_ticks = _axis_ticks(y_min, y_max)
    for xv in x_ticks:
        if origin_inside and abs(xv) < 1e-9:
            continue
        parts.append(f'<text x="{_f(map_x(xv))}" y="{_f(x_axis_y+18)}" font-family="{_FONT}" font-size="14" fill="{MUTED}" text-anchor="middle">{_tick_text(xv)}</text>')
    for yv in y_ticks:
        if origin_inside and abs(yv) < 1e-9:
            continue
        parts.append(f'<text x="{_f(y_axis_x-10)}" y="{_f(map_y(yv)+3)}" font-family="{_FONT}" font-size="14" fill="{MUTED}" text-anchor="end">{_tick_text(yv)}</text>')
    if origin_inside:
        parts.append(f'<text x="{_f(y_axis_x-10)}" y="{_f(x_axis_y+18)}" font-family="{_FONT}" font-size="14" fill="{MUTED}" text-anchor="end">0</text>')

    # curves (labels placed in a second pass with collision avoidance)
    label_anchors: list[tuple[tuple[float, float], str, str]] = []
    for curve in plot.curves:
        radius = _polar_radius(curve, x_min, x_max)
        if radius is not None:
            # Polar radial bound (label like "r = √5", constant radius):
            # a circle in data space, NOT a Cartesian horizontal line (which
            # misreads as y = const). Ellipse radii follow each axis scale.
            color = _VECTOR_COLORS.get(curve.color, ACCENT if curve.color == "" else INK)
            rx = radius / (x_max - x_min) * plot_w
            ry = radius / (y_max - y_min) * plot_h
            parts.append(
                f'<ellipse cx="{_f(map_x(0))}" cy="{_f(map_y(0))}" rx="{_f(rx)}" ry="{_f(ry)}" '
                f'fill="none" stroke="{color}" stroke-width="2.2"/>'
            )
            if curve.label:
                label_anchors.append(((map_x(0) + rx, map_y(0)), curve.label, color))
            continue
        pts = _curve_points(curve, x_min, x_max)
        if not pts:
            continue
        color = _VECTOR_COLORS.get(curve.color, ACCENT if curve.color == "" else INK)
        if len(pts) == 1 and (curve.label or "").strip():
            # Labeled single point: plotted vertex marker (e.g. Argand-plane
            # vertices). Unlabeled singletons stay skipped — no stray dots.
            mx, my = map_x(pts[0][0]), map_y(pts[0][1])
            parts.append(
                f'<circle cx="{_f(mx)}" cy="{_f(my)}" r="5" fill="{color}"/>'
            )
            label_anchors.append(((mx, my), curve.label.strip(), color))
            continue
        # clip to y range for clean visual (keep slightly outside then clip)
        segs: list[list[tuple[float, float]]] = []
        cur: list[tuple[float, float]] = []
        for xv, yv in pts:
            if yv < y_min - (y_max - y_min) * 0.1 or yv > y_max + (y_max - y_min) * 0.1:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            cur.append((map_x(xv), map_y(yv)))
        if cur:
            segs.append(cur)
        dash = ' stroke-dasharray="6 4"' if curve.style == "dashed" else ""
        for seg in segs:
            if len(seg) < 2:
                continue
            points = " ".join(f"{_f(x)},{_f(y)}" for x, y in seg)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
        if curve.label and segs and segs[-1]:
            anchor = segs[-1][-1]
            if len(segs[-1]) > 3:
                first, last = segs[-1][0], segs[-1][-1]
                if abs(first[0] - last[0]) < 0.5 and abs(first[1] - last[1]) < 0.5:
                    # Closed loop (e.g. a square): the end anchor coincides
                    # with the start corner where a vertex label already sits.
                    # Anchor at the first edge midpoint instead (no vertex).
                    p0, p1 = segs[-1][0], segs[-1][1]
                    anchor = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            label_anchors.append((anchor, curve.label, color))
    # Spec-supplied straight primitives: lens lines, object/image arrows,
    # dimension segments. Code owns the arrowhead geometry (never spec SVG).
    for segment in plot.segments:
        color = _VECTOR_COLORS.get(segment.color, ACCENT if segment.color == "" else INK)
        sx, sy = map_x(segment.x1), map_y(segment.y1)
        ex, ey = map_x(segment.x2), map_y(segment.y2)
        dash = "6 4" if segment.style == "dashed" else None
        parts.append(_line(sx, sy, ex, ey, color, width=2.4, dash=dash))
        if segment.arrow:
            deg = math.degrees(math.atan2(segment.y2 - segment.y1, segment.x2 - segment.x1))
            parts.append(_arrowhead(ex, ey, deg, 11))
        if (segment.label or "").strip():
            label_anchors.append((((sx + ex) / 2, (sy + ey) / 2), segment.label.strip(), color))
    for marker in plot.markers:
        color = _VECTOR_COLORS.get(marker.color, ACCENT if marker.color == "" else INK)
        mx, my = map_x(marker.x), map_y(marker.y)
        parts.append(f'<circle cx="{_f(mx)}" cy="{_f(my)}" r="5" fill="{color}"/>')
        if (marker.label or "").strip():
            label_anchors.append(((mx, my), marker.label.strip(), color))
    # Generic collision-aware curve labels: labels sharing an anchor point
    # (e.g. "square ABCD" and "A(1,1)" ending at the same corner) get
    # deterministic fallback offsets so they never occupy the same box.
    # First candidate == legacy (+6,-6) placement, so non-colliding labels
    # render byte-identically to before.
    placed_boxes: list[tuple[float, float, float, float]] = []
    for (ax, ay), label, color in label_anchors:
        est_w = len(label) * 9.6 + 8
        est_h = 18.0
        candidates = [
            (ax + 6, ay - 6 - est_h / 2),
            (ax + 6, ay - 24 - est_h / 2),
            (ax + 6, ay + 14 - est_h / 2),
            (ax - est_w - 6, ay - 6 - est_h / 2),
            (ax - est_w - 6, ay - 24 - est_h / 2),
            (ax + 6, ay + 32 - est_h / 2),
        ]
        # Candidates must ALSO stay inside the plot area — the old code let a
        # long label spill past the card edge (real math board: "Outer
        # Radius..." clipped). est_w is conservative, so clamp loosely.
        def _inside(cx: float) -> bool:
            return plot_x0 - 4 <= cx and cx + est_w <= plot_x0 + plot_w + 4

        chosen = candidates[0]
        for cx, cy in candidates:
            box = (cx, cy, est_w, est_h)
            if _inside(cx) and not any(_boxes_overlap(box, other, pad=3.0) for other in placed_boxes):
                chosen = (cx, cy)
                break
        else:
            # No free in-bounds slot: prefer staying inside the plot over
            # avoiding overlap (a clipped label is lost; an overlapped one
            # is still readable via its halo).
            for cx, cy in candidates:
                if _inside(cx):
                    chosen = (cx, cy)
                    break
            else:
                for cx, cy in candidates:
                    box = (cx, cy, est_w, est_h)
                    if not any(_boxes_overlap(box, other, pad=3.0) for other in placed_boxes):
                        chosen = (cx, cy)
                        break
        # Hard guarantee: slide the chosen box inside the plot horizontally.
        # The anchor may sit at the plot edge (curve ends there); selection
        # alone cannot fix that, clamping can.
        cx, cy = chosen
        cx = min(cx, plot_x0 + plot_w + 4 - est_w)
        cx = max(cx, plot_x0 - 4)
        chosen = (cx, cy)
        placed_boxes.append((chosen[0], chosen[1], est_w, est_h))
        # text-anchor=start at box left; baseline chosen so the legacy
        # (+6,-6) candidate renders byte-identically to before.
        parts.append(_halo(chosen[0] + est_w / 2, chosen[1] + est_h / 2 - 3, est_w + 8, est_h + 8))
        parts.append(f'<text x="{_f(chosen[0])}" y="{_f(chosen[1] + est_h / 2)}" font-family="{_FONT}" font-size="16" font-weight="600" fill="{color}" text-anchor="start">{_esc(label)}</text>')
    return "".join(parts), STAGE_Y + STAGE_H + 18, [], None


# Visible satellite capacity of the generic hero. The gate may only treat
# callouts within this capacity as shown, and the deterministic claim
# promotion (visual_lesson.promote_equation_claims) must not exceed it.
GENERIC_SATELLITE_CAP = 8


def _render_generic(scene: VisualScene, fallback_title: str = "") -> tuple[str, int, list[str], object]:
    generic = scene.generic
    central = ""
    callouts: list[str] = []
    if generic is not None:
        central = (generic.central_label or "").strip()
        callouts = [c.strip() for c in (generic.callouts or []) if c.strip()][:GENERIC_SATELLITE_CAP]
    if not central:
        central = (scene.title or fallback_title or "Key Concept").strip()
    if not callouts:
        # synthesize from caption if needed — still guarantees a visual
        cap = (scene.caption or "").strip()
        if cap:
            callouts = _wrap(cap, 28)[:2]
    if not central and not callouts:
        return "", STAGE_Y + STAGE_H + 24, [], None
    # layout: centre box + up to GENERIC_SATELLITE_CAP satellites
    cx, cy = 400.0, STAGE_Y + 230
    cw, ch = 220, 64
    parts: list[str] = []
    # centre
    parts.append(f'<rect x="{_f(cx-cw/2)}" y="{_f(cy-ch/2)}" width="{cw}" height="{ch}" rx="14" fill="#ede9fe" stroke="{ACCENT}" stroke-width="2"/>')
    clines = _wrap(central, 20)
    ty = cy - (len(clines) - 1) * 9
    for line in clines:
        parts.append(f'<text x="{_f(cx)}" y="{_f(ty)}" font-family="{_FONT}" font-size="14" font-weight="700" fill="{INK}" text-anchor="middle">{_esc(line)}</text>')
        ty += 18
    # satellites (fifth drops below centre so 5-item formula sheets never
    # silently lose a formula to the old 4-satellite cap; sixth/seventh flank
    # it and eighth sits above centre — all inside the stage)
    positions = [
        (-150, -96), (150, -96), (-150, 96), (150, 96),
        (0, 196), (-180, 196), (180, 196), (0, -196),
    ]
    bw, bh = 168, 46
    for idx, txt in enumerate(callouts[: len(positions)]):
        px, py = cx + positions[idx][0], cy + positions[idx][1]
        # connector
        sx = cx + (cw/2 + 8) * (1 if px > cx else -1) if abs(px - cx) > 80 else cx
        sy = cy + (ch/2 + 6) * (1 if py > cy else -1) if abs(py - cy) > 40 else cy
        parts.append(f'<line x1="{_f(sx)}" y1="{_f(sy)}" x2="{_f(px)}" y2="{_f(py)}" stroke="{ACCENT}" stroke-width="1.4" stroke-dasharray="5 4" opacity="0.9"/>')
        # Wrap to the pill width (18 chars at size 14 fits 168px at the 0.64
        # advance; 22-char lines spilled ~15px past each edge) and grow the
        # pill when the text block needs the height.
        tlines = _wrap(txt, 18)
        ph = max(bh, 22 + 14 * (len(tlines) - 1))
        parts.append(f'<rect x="{_f(px-bw/2)}" y="{_f(py-ph/2)}" width="{bw}" height="{_f(ph)}" rx="10" fill="{CARD_FILL}" stroke="{CARD_STROKE}" stroke-width="1.5"/>')
        ty2 = py - (len(tlines) - 1) * 7
        for line in tlines:
            parts.append(f'<text x="{_f(px)}" y="{_f(ty2)}" font-family="{_FONT}" font-size="14" fill="{INK}" text-anchor="middle">{_esc(line)}</text>')
            ty2 += 14
    return "".join(parts), STAGE_Y + STAGE_H + 18, [], None


def _comparison_anchor(
    center: tuple[float, float],
    at: str,
    radius: float,
) -> tuple[float, float]:
    """Resolve a comparison-panel anchor (P'/CM/P) to a drawn point."""
    cx, cy = center
    if at == "top":
        return (cx, cy - radius)
    if at == "bottom":
        return (cx, cy + radius)
    if at == "left":
        return (cx - radius, cy)
    if at == "right":
        return (cx + radius, cy)
    return center


def _comparison_point(
    point: ComparisonPoint,
    center: tuple[float, float],
    radius: float,
) -> str:
    """Draw one labeled material point on a comparison body."""
    if not point.label.strip():
        return ""
    x, y = _comparison_anchor(center, point.at, radius)
    if point.at == "top":
        tx, ty, anchor = x, y - 12, "middle"
    elif point.at == "bottom":
        tx, ty, anchor = x, y + 22, "middle"
    elif point.at == "left":
        tx, ty, anchor = x - 12, y + 5, "end"
    elif point.at == "right":
        tx, ty, anchor = x + 12, y + 5, "start"
    else:
        tx, ty, anchor = x - 14, y + 5, "end"
    parts = [f'<circle cx="{_f(x)}" cy="{_f(y)}" r="5" fill="{INK}"/>']
    parts.append(
        f'<text x="{_f(tx)}" y="{_f(ty)}" font-family="{_FONT}" font-size="15" '
        f'font-weight="700" fill="{INK}" text-anchor="{anchor}">{_esc(point.label)}</text>'
    )
    return "".join(parts)


def _render_comparison(scene: VisualScene) -> tuple[str, int, list[str], object]:
    """Render a 2-3 panel comparison with code-owned geometry.

    Two panels sit side by side; three panels use a 2+1 grid that mirrors the
    canonical physics-board layout (translation | rotation, then combined).
    Each panel supports labeled material points and vectors anchored at
    ``at`` (top = P', center = CM, bottom = P). A zero-length vector is a real
    physical value (v = 0 at the rolling contact / pure-rotation CM) and is
    drawn as a point + label, never as a minimum-length arrow.
    """
    comparison: VisualComparison | None = scene.comparison
    legend: list[str] = []
    if comparison is None:
        return "", STAGE_Y + STAGE_H + 24, legend, None
    panels = [p for p in comparison.panels if p.label.strip()][:3]
    if len(panels) < 2:
        return "", STAGE_Y + STAGE_H + 24, legend, None

    parts: list[str] = []
    n = len(panels)
    top_radius = 62.0 if n == 3 else 72.0
    if n == 3:
        col_w = STAGE_W / 2
        positions = [
            (STAGE_X + col_w * 0.5, STAGE_Y + 152),
            (STAGE_X + col_w * 1.5, STAGE_Y + 152),
            (STAGE_X + STAGE_W * 0.5, STAGE_Y + 452),
        ]
        label_ys = [STAGE_Y + 30, STAGE_Y + 30, STAGE_Y + 336]
        note_ys = [STAGE_Y + 264, STAGE_Y + 264, STAGE_Y + 554]
        note_widths = [30, 30, 48]
        note_caps = [2, 2, 1]
        parts.append(
            f'<line x1="{_f(STAGE_X + 12)}" y1="{_f(STAGE_Y + 306)}" '
            f'x2="{_f(STAGE_X + STAGE_W - 12)}" y2="{_f(STAGE_Y + 306)}" '
            f'stroke="{STAGE_STROKE}" stroke-width="1.5" stroke-dasharray="5 5"/>'
        )
        divider_ys = [(STAGE_Y + 12, STAGE_Y + 306)]
    else:
        col_w = STAGE_W / n
        positions = [(STAGE_X + col_w * (i + 0.5), STAGE_Y + 234) for i in range(n)]
        label_ys = [STAGE_Y + 34 for _ in range(n)]
        note_ys = [STAGE_Y + 404 for _ in range(n)]
        note_widths = [34 for _ in range(n)]
        note_caps = [4 for _ in range(n)]
        divider_ys = [(STAGE_Y + 12, STAGE_Y + STAGE_H - 12)]

    for i, panel in enumerate(panels):
        cx, obj_y = positions[i]
        radius = top_radius
        center = (cx, obj_y)
        if n == 3 and i == 1:
            x_div = STAGE_X + col_w
            y0, y1 = divider_ys[0]
            parts.append(
                f'<line x1="{_f(x_div)}" y1="{_f(y0)}" x2="{_f(x_div)}" '
                f'y2="{_f(y1)}" stroke="{STAGE_STROKE}" stroke-width="1.5" '
                f'stroke-dasharray="5 5"/>'
            )
        elif n == 2 and i == 1:
            x_div = STAGE_X + col_w
            y0, y1 = divider_ys[0]
            parts.append(
                f'<line x1="{_f(x_div)}" y1="{_f(y0)}" x2="{_f(x_div)}" '
                f'y2="{_f(y1)}" stroke="{STAGE_STROKE}" stroke-width="1.5" '
                f'stroke-dasharray="5 5"/>'
            )
        for j, line in enumerate(_wrap(panel.label.strip(), 22 if n == 3 and i < 2 else 30)):
            if j >= 2:
                break
            parts.append(
                f'<text x="{_f(cx)}" y="{_f(label_ys[i] + j * 26)}" font-family="{_FONT}" '
                f'font-size="20" font-weight="700" fill="{INK}" text-anchor="middle">{_esc(line)}</text>'
            )
        parts.append(_render_object(panel.object, cx, obj_y, label_side="left", radius=radius))
        for point in panel.points:
            parts.append(_comparison_point(point, center, radius))
        for v in panel.vectors:
            if not v.label.strip():
                continue
            color = _VECTOR_COLORS.get(v.color, INK)
            tail = _comparison_anchor(center, v.at, radius)
            long_claim = len(v.label) > 12
            parts.append(
                _render_vector(
                    v,
                    tail,
                    color,
                    length_scale=0.55,
                    font_size=13 if long_claim else 15,
                    label_perp=-14.0,
                    label_mode="head" if long_claim else "mid",
                    # A head-mounted long claim (v = v_CM + Rω = 2v_CM) must
                    # finish inside the 704px stage; 1.15 leaves ~6px margin
                    # while still reading longer than the 1.0 centre vector.
                    max_length=1.15 if long_claim else 1.6,
                )
            )
        for arc in panel.arcs:
            parts.append(_rotation_arc(arc, center, show_label=False))
            if arc.caption.strip():
                legend.append(f"{arc.label.strip()} — {arc.caption.strip()}")
        note_lines = (
            _wrap(panel.note.strip(), note_widths[i])[: note_caps[i]]
            if panel.note.strip()
            else []
        )
        for k, line in enumerate(note_lines):
            parts.append(
                f'<text x="{_f(cx)}" y="{_f(note_ys[i] + k * 22)}" font-family="{_FONT}" '
                f'font-size="14" fill="{MUTED}" text-anchor="middle">{_esc(line)}</text>'
            )
    return "".join(parts), STAGE_Y + STAGE_H + 24, legend, comparison.relation


# ── worked mathematics: number-line family ─────────────────────────────────
#
# Compact, content-fitted layout (no fixed stage, no minimum canvas height):
# problem card -> domain track -> case rows -> verified final answer track.
# All geometry is computed here from exact Fractions; typography never
# overlaps and every label sits inside the canvas.



def _nl_numbers(nl) -> list[float]:
    from app.utils.math_intervals import parse_bound

    values: list[float] = []
    for track in (nl.domain, nl.solution):
        if track is None:
            continue
        for iv in track.intervals:
            lo = parse_bound(iv.lo, minus_side=True)
            if not isinstance(lo, str) and lo is not None:
                values.append(float(lo))
            hi = parse_bound(iv.hi, minus_side=False)
            if not isinstance(hi, str) and hi is not None:
                values.append(float(hi))
        for p in track.points:
            parsed = parse_bound(p, minus_side=True)
            if not isinstance(parsed, str) and parsed is not None:
                values.append(float(parsed))
    return values




def _nl_label(value: Fraction) -> str:
    """Exact endpoint label: integers plain, rationals as n/d (never float)."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


# ── worked mathematics: study-sheet composition ────────────────────────────
#
# The visual is a teaching sheet, not a dump of cards: a dark question header,
# numbered domain/case/answer sections, color-coded number lines, and a short
# "why this works" strip. All mathematical strings come from the verifier
# (SolveResult) or the verified claim; nothing is re-derived in the renderer.

WM_NAVY = "#0f172a"
WM_NAVY_LABEL = "#a5b4fc"
WM_INDIGO = "#4f46e5"
WM_INDIGO_TINT = "#eef2ff"
WM_AMBER = "#b45309"
WM_AMBER_TINT = "#fffbeb"
WM_TEAL = "#0f766e"
WM_TEAL_TINT = "#f0fdfa"
WM_GREEN_TINT = "#f0fdf4"
WM_LINE = "#e7e5e4"
WM_SOFT = "#78716c"
WM_CARD = "#ffffff"
WM_M = 32
WM_CW = VIEW_W - 2 * WM_M
WM_SKY = "#eef7ff"
WM_GRID = "#dbeafe"
WM_RAW = "#f1f5f9"


def _wm_num(text: str) -> str:
    return text.replace("-inf", "\u2212\u221e").replace("inf", "\u221e").replace("-", "\u2212")


def _wm_set_text(values: "IntervalSet") -> str:
    """Canonical interval-set text with typographic minus/infinity."""
    from app.utils.math_intervals import format_interval_set

    return _wm_num(format_interval_set(values))


_WM_SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎", "x": "ₓ",
}


def _wm_math(text: str) -> str:
    """Display-only math typography; never changes the verified expression."""
    text = text.replace("^2", "²").replace("^3", "³")
    text = text.replace("*", "·")
    return text.replace(" - ", " − ")


def _wm_problem(text: str) -> str:
    """Typeset a log base as a real subscript when every glyph is supported."""
    text = _wm_math(text)

    def _sub(match: re.Match[str]) -> str:
        base = match.group(1).strip()
        compact = base.replace(" ", "")
        if compact and all(ch in _WM_SUBSCRIPTS for ch in compact):
            return "log" + "".join(_WM_SUBSCRIPTS[ch] for ch in compact)
        return f"log_({base})"

    return re.sub(r"log_\{([^{}]+)\}", _sub, text)


def _wm_phrase(values: "IntervalSet") -> str:
    """Human sentence for a small union (x < -1 or x > 3)."""
    from app.utils.math_intervals import format_bound

    parts: list[str] = []
    for iv in values.normalized().intervals:
        lo = format_bound(iv.lo, minus_side=True)
        hi = format_bound(iv.hi, minus_side=False)
        lo_inf = iv.lo is None
        hi_inf = iv.hi is None
        lower = "≤" if iv.lo_closed else "<"
        upper = "≤" if iv.hi_closed else "<"
        beyond = "≥" if iv.lo_closed else ">"
        if lo_inf and hi_inf:
            parts.append("all real x")
        elif lo_inf:
            parts.append(f"x {upper} {hi}")
        elif hi_inf:
            parts.append(f"x {beyond} {lo}")
        elif lo == hi and iv.lo_closed and iv.hi_closed:
            parts.append(f"x = {lo}")
        else:
            parts.append(f"{lo} {lower} x {upper} {hi}")
    if not parts:
        return "no real x"
    if len(parts) > 3:
        return _wm_set_text(values)
    return " or ".join(parts)


def _wm_track_from_set(values: "IntervalSet") -> VisualNumberLineTrack:
    from app.utils.math_intervals import format_bound

    return VisualNumberLineTrack(
        intervals=[
            VisualInterval(
                lo=format_bound(iv.lo, minus_side=True),
                hi=format_bound(iv.hi, minus_side=False),
                lo_closed=iv.lo_closed,
                hi_closed=iv.hi_closed,
            )
            for iv in values.normalized().intervals
        ]
    )


def _wm_text(
    x: float,
    y: float,
    text: str,
    size: float = 14,
    fill: str = INK,
    weight: str = "400",
    anchor: str = "start",
    spacing: float | None = None,
) -> str:
    extra = f' letter-spacing="{spacing}"' if spacing is not None else ""
    return (
        f'<text x="{_f(x)}" y="{_f(y)}" font-family="{_FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{_esc(text)}</text>'
    )


def _wm_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    rx: float,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 0,
    opacity: float = 1.0,
) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke != "none" else ""
    opacity_attr = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return (
        f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" '
        f'rx="{_f(rx)}" fill="{fill}"{stroke_attr}{opacity_attr}/>'
    )


def _wm_rel(rel: str) -> str:
    return {"<=": "≤", ">=": "≥"}.get(rel, rel)


def _wm_scales(solved: "SolveResult", nl) -> tuple[float, float]:
    values = _nl_numbers(nl)
    for case in solved.cases:
        for track in (case.raw_solution, case.solution):
            if track is None:
                continue
            for iv in track.normalized().intervals:
                for bound in (iv.lo, iv.hi):
                    if bound is not None:
                        values.append(float(bound))
    for point in solved.critical_points:
        values.append(float(point))
    if not values:
        return -1.0, 1.0
    lo, hi = min(values) - 1.0, max(values) + 1.0
    if hi - lo < 1.0:
        mid = (lo + hi) / 2
        lo, hi = mid - 1.0, mid + 1.0
    return lo, hi


def _wm_light_section(y: int, number: str, label: str, color: str = WM_INDIGO) -> tuple[str, int]:
    """Editorial section marker: colour bar, step number, label, hairline.

    Deliberately lighter than the old numbered chip: this sheet reads as one
    composed poster, not a stack of unrelated cards.
    """
    parts = [
        _wm_rect(WM_M, y, 3, 22, 1.5, color),
        _wm_text(WM_M + 12, y + 11, number, 8.5, color, "700", "start", 1.2),
        _wm_text(WM_M + 30, y + 11, label, 10, color, "700", "start", 1.2),
        f'<line x1="{WM_M}" y1="{y + 30}" x2="{VIEW_W - WM_M}" y2="{y + 30}" stroke="{WM_LINE}" stroke-width="1"/>',
    ]
    return "".join(parts), y + 42


def _wm_infographic_header(
    title: str,
    problem: str,
    key_idea: str,
) -> tuple[str, int]:
    """Light, reference-style masthead: big title, question, one-line idea."""
    title_lines = _wrap(title, 40)[:2]
    parts = [
        f'<rect x="0" y="0" width="{VIEW_W}" height="0" fill="{WM_SKY}"/>',
    ]
    # Background height is patched after the text height is known.
    top = 22
    parts.append(
        _wm_text(WM_M, top + 2, "WORKED MATHEMATICS  ·  VERIFIED SOLUTION", 8.5, WM_INDIGO, "700", "start", 1.5)
    )
    parts.append(
        _wm_text(VIEW_W - WM_M, top + 2, "EXACT · SOURCE-GROUNDED", 8.5, WM_SOFT, "700", "end", 1.2)
    )
    ty = top + 36
    for line in title_lines:
        parts.append(_wm_text(WM_M, ty, line, 29, INK, "700"))
        ty += 34
    ty += 6
    parts.append(_wm_text(WM_M, ty, "QUESTION", 8.5, WM_INDIGO, "700", "start", 1.3))
    ty += 24
    parts.append(_wm_text(WM_M, ty, problem, 21, INK, "700"))
    ty += 22
    if key_idea.strip():
        for line in _wrap(key_idea.strip(), 94):
            parts.append(_wm_text(WM_M, ty, line, 12.5, WM_SOFT))
            ty += 17
    bottom = ty + 12
    parts[0] = f'<rect x="0" y="0" width="{VIEW_W}" height="{bottom}" fill="{WM_SKY}"/>'
    parts.append(
        f'<line x1="{WM_M}" y1="{bottom - 1}" x2="{VIEW_W - WM_M}" y2="{bottom - 1}" '
        f'stroke="{WM_GRID}" stroke-width="1"/>'
    )
    return "".join(parts), bottom


def _wm_x(value: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (value - lo) / (hi - lo) * (x1 - x0)


def _wm_span(iv, lo: float, hi: float, x0: float, x1: float) -> tuple[float, float]:
    left = x0 if iv.lo is None else _wm_x(float(iv.lo), lo, hi, x0, x1)
    right = x1 if iv.hi is None else _wm_x(float(iv.hi), lo, hi, x0, x1)
    return left, right


def _wm_draw_row(
    row_y: int,
    label: str,
    sub: str,
    color: str,
    final,
    lo: float,
    hi: float,
    x0: float,
    x1: float,
    labels: list[tuple[float, str, str]],
    raw=None,
    excluded_points: tuple | list = (),
) -> str:
    """One aligned track in the solution map.

    The light band is the raw branch and the solid band is what survives the
    domain. Every track shares the same x scale, so the student can see the
    intersection instead of reading four unrelated number lines.
    """
    from app.utils.math_intervals import format_bound

    parts: list[str] = []
    baseline = row_y + 20
    parts.append(_wm_text(WM_M, row_y + 7, label, 9, color, "700", "start", 1.1))
    parts.append(_wm_text(WM_M, row_y + 21, sub, 8, WM_SOFT))
    parts.append(
        f'<line x1="{_f(x0)}" y1="{_f(baseline)}" x2="{_f(x1)}" y2="{_f(baseline)}" '
        f'stroke="{WM_LINE}" stroke-width="1.5"/>'
    )

    if raw is not None and not raw.equals(final):
        for iv in raw.normalized().intervals:
            left, right = _wm_span(iv, lo, hi, x0, x1)
            if right - left >= 1:
                parts.append(
                    _wm_rect(
                        left, baseline - 18, right - left, 5, 2.5, color,
                        opacity=0.28,
                    )
                )

    excluded_set = {float(point) for point in excluded_points}
    final_intervals = final.normalized().intervals
    if not final_intervals:
        parts.append(_wm_text(x0 + 8, baseline - 10, "no intervals survive", 9, color, "700"))
    for iv in final_intervals:
        left, right = _wm_span(iv, lo, hi, x0, x1)
        if right - left >= 1:
            parts.append(_wm_rect(left, baseline - 10, right - left, 10, 5, color))
        if iv.lo is not None and float(iv.lo) not in excluded_set:
            lx = _wm_x(float(iv.lo), lo, hi, x0, x1)
            fill = color if iv.lo_closed else "white"
            parts.append(
                f'<circle cx="{_f(lx)}" cy="{_f(baseline)}" r="6.5" fill="{fill}" '
                f'stroke="{color}" stroke-width="2.2"/>'
            )
            labels.append((lx, _nl_label(iv.lo), color))
        if iv.hi is not None and float(iv.hi) not in excluded_set:
            rx = _wm_x(float(iv.hi), lo, hi, x0, x1)
            fill = color if iv.hi_closed else "white"
            parts.append(
                f'<circle cx="{_f(rx)}" cy="{_f(baseline)}" r="6.5" fill="{fill}" '
                f'stroke="{color}" stroke-width="2.2"/>'
            )
            labels.append((rx, _nl_label(iv.hi), color))

    for point in excluded_points:
        px = _wm_x(float(point), lo, hi, x0, x1)
        parts.append(
            f'<circle cx="{_f(px)}" cy="{_f(baseline)}" r="6.5" fill="white" '
            f'stroke="{RED}" stroke-width="2.2"/>'
        )
    return "".join(parts)


def _wm_solution_map(
    y: int,
    solved: "SolveResult",
    nl,
    lo: float,
    hi: float,
) -> tuple[str, int]:
    """Full-width, four-track solution map + answer band."""
    from app.utils.math_intervals import Interval
    from app.utils.math_intervals import IntervalSet as _IntervalSet

    parts: list[str] = []
    header, y = _wm_light_section(y, "01", "FINAL ANSWER — MATHEMATICALLY VERIFIED", GREEN)
    parts.append(header)

    answer = _wm_set_text(solved.solution) if solved.solution is not None else "∅"
    empty = solved.solution is not None and not solved.solution.normalized().intervals
    answer_h = 72
    parts.append(_wm_rect(WM_M, y, WM_CW, answer_h, 14, WM_GREEN_TINT))
    parts.append(_wm_rect(WM_M, y, 5, answer_h, 2.5, GREEN))
    parts.append(_wm_text(WM_M + 18, y + 22, "SOLUTION SET", 8.5, GREEN, "700", "start", 1.2))
    if empty:
        parts.append(_wm_text(WM_M + 18, y + 50, "No real solutions — the inequality is never true", 17, INK, "700"))
    else:
        parts.append(_wm_text(WM_M + 18, y + 50, f"x ∈ {answer}", 21, "#14532d", "700"))
    badge_w = 86
    parts.append(_wm_rect(VIEW_W - WM_M - badge_w - 12, y + 22, badge_w, 26, 13, "#dcfce7"))
    parts.append(_wm_text(VIEW_W - WM_M - badge_w / 2 - 12, y + 39, "✓ VERIFIED", 9, GREEN, "700", "middle", 0.8))
    y += answer_h + 12

    whole_line = _IntervalSet.of(Interval(None, None, False, False))
    has_domain = solved.domain is not None and not solved.domain.equals(whole_line)
    case_pairs = _wm_case_pairs(nl, solved)
    row_count = int(has_domain) + len(case_pairs) + 1
    map_top = y
    map_h = 34 + row_count * 58 + 52
    parts.append(_wm_rect(WM_M, map_top, WM_CW, map_h, 14, "#ffffff", WM_GRID, 1.2))
    parts.append(_wm_text(WM_M + 16, map_top + 18, "SOLUTION MAP", 8.5, WM_INDIGO, "700", "start", 1.2))
    parts.append(_wm_text(VIEW_W - WM_M - 16, map_top + 18, "light = before domain  ·  solid = kept", 8.5, WM_SOFT, "400", "end"))

    label_w = 92
    x0 = WM_M + 16 + label_w
    x1 = VIEW_W - WM_M - 16
    critical: list[float] = [float(p) for p in solved.critical_points]
    for track in (solved.domain, solved.solution):
        if track is None:
            continue
        for iv in track.normalized().intervals:
            if iv.lo is not None:
                critical.append(float(iv.lo))
            if iv.hi is not None:
                critical.append(float(iv.hi))
    for p in critical:
        gx = _wm_x(p, lo, hi, x0, x1)
        parts.append(
            f'<line x1="{_f(gx)}" y1="{map_top + 28}" x2="{_f(gx)}" y2="{map_top + map_h - 22}" '
            f'stroke="{WM_GRID}" stroke-width="1" stroke-dasharray="3 4"/>'
        )

    labels: list[tuple[float, str, str]] = []
    row_y = map_top + 34
    if has_domain:
        assert solved.domain is not None
        parts.append(
            _wm_draw_row(
                row_y, "DOMAIN", "where it is defined", ACCENT,
                solved.domain, lo, hi, x0, x1, labels,
                excluded_points=solved.excluded_points,
            )
        )
        row_y += 58
    for idx, (model_case, solver_case) in enumerate(case_pairs):
        color = WM_AMBER if idx == 0 else WM_TEAL
        row_label = "CASE 1" if idx == 0 else "CASE 2"
        row_sub = "base > 1" if solver_case is not None and solver_case.regime_kind == ">1" else "0 < base < 1"
        final = solver_case.solution if solver_case is not None else (solved.domain or solved.solution)
        raw = solver_case.raw_solution if solver_case is not None else None
        parts.append(
            _wm_draw_row(
                row_y, row_label, row_sub, color, final, lo, hi, x0, x1, labels,
                raw=raw,
            )
        )
        row_y += 58
    parts.append(
        _wm_draw_row(
            row_y, "SOLUTION", "final union", GREEN,
            solved.solution, lo, hi, x0, x1, labels,
        )
    )

    # Shared endpoint ruler: labels are placed once for the whole map and
    # staggered onto extra rows when two exact values are close together.
    rows_x: list[float] = []
    placements: list[tuple[float, str, str, int]] = []
    seen: set[tuple[str, int]] = set()
    for x, text, color in sorted(labels, key=lambda item: (item[0], item[1])):
        key = (text, int(round(x)))
        if key in seen:
            continue
        seen.add(key)
        row = 0
        while row < len(rows_x) and abs(x - rows_x[row]) < 34:
            row += 1
        if row == len(rows_x):
            rows_x.append(x)
        else:
            rows_x[row] = x
        placements.append((x, text, color, row))
    ruler_y = map_top + map_h - 44
    for x, text, color, row in placements:
        label_x = min(max(x, x0), x1)
        parts.append(
            _wm_text(label_x, ruler_y + row * 20, text, 10.5, color, "700", "middle")
        )
    note_y = ruler_y + max(1, len(rows_x)) * 20
    for point in solved.excluded_points:
        px = _wm_x(float(point), lo, hi, x0, x1)
        parts.append(
            _wm_text(px, note_y, f"x ≠ {_nl_label(point)}", 9.5, RED, "700", "middle")
        )
    y = map_top + map_h + 10
    return "".join(parts), y


def _wm_domain_constraints(y: int, solved: "SolveResult") -> tuple[str, int]:
    domain_label = (
        "DOMAIN — WHERE THE LOG IS DEFINED"
        if solved.base_display
        else "DOMAIN — WHERE THE EXPRESSION IS DEFINED"
    )
    header, y = _wm_light_section(y, "02", domain_label, WM_INDIGO)
    parts = [header]
    intro = (
        "Every restriction must hold at once. The logarithm needs a positive argument, "
        "a positive base, and a base different from 1."
        if solved.base_display
        else "The expression is undefined wherever a denominator is zero."
    )
    for line in _wrap(intro, 110):
        parts.append(_wm_text(WM_M, y, line, 12, WM_SOFT))
        y += 17
    y += 8

    cards: list[tuple[str, str, str, str]] = []
    if solved.base_display:
        cards.append(
            (
                "ARGUMENT > 0",
                solved.arg_display,
                _wm_phrase(solved.arg_positive) if solved.arg_positive is not None else "positive",
                WM_INDIGO,
            )
        )
        if not solved.constant_base:
            cards.append(
                (
                    "BASE > 0",
                    solved.base_display,
                    _wm_phrase(solved.base_positive) if solved.base_positive is not None else "positive",
                    WM_INDIGO,
                )
            )
            exclude = ", ".join(_nl_label(p) for p in solved.excluded_points)
            cards.append(
                (
                    "BASE ≠ 1",
                    f"{solved.base_display} ≠ 1",
                    f"exclude x = {exclude}" if exclude else "base stays valid",
                    RED,
                )
            )
    else:
        exclude = ", ".join(_nl_label(p) for p in solved.excluded_points)
        cards.append(("DENOMINATOR ≠ 0", f"x ≠ {exclude}", "excluded from the domain", RED))

    if cards:
        cols = len(cards) if len(cards) <= 3 else 3
        gutter = 10
        cw = (WM_CW - gutter * (cols - 1)) / cols
        ch = 68
        for i, (label, expr, consequence, color) in enumerate(cards):
            cx = WM_M + i * (cw + gutter)
            parts.append(_wm_rect(cx, y, cw, ch, 10, "#ffffff", WM_LINE, 1))
            parts.append(_wm_rect(cx, y, 4, ch, 2, color))
            parts.append(_wm_text(cx + 14, y + 18, label, 8, color, "700", "start", 1.1))
            parts.append(_wm_text(cx + 14, y + 38, expr, 12.5, INK, "700"))
            parts.append(_wm_text(cx + 14, y + 55, _wrap(consequence, 30)[0], 9.5, WM_SOFT if color != RED else RED))
        y += ch + 10

    domain_text = _wm_set_text(solved.domain) if solved.domain is not None else "all real x"
    parts.append(_wm_text(WM_M, y + 4, "Together:", 10.5, WM_SOFT, "700"))
    parts.append(_wm_text(WM_M + 58, y + 4, domain_text, 12, WM_INDIGO, "700"))
    return "".join(parts), y + 18


def _wm_case_pairs(nl, solved: "SolveResult"):
    """Pair model-authored case text with the verifier's exact case result."""
    pairs = []
    used: set[int] = set()
    for idx, model_case in enumerate(getattr(nl, "cases", []) or []):
        cond = (model_case.condition or "").strip()
        want = "<1" if cond.startswith("0 <") or "0 <" in cond else ">1"
        solver_case = None
        for candidate in solved.cases:
            if id(candidate) not in used and candidate.regime_kind == want:
                solver_case = candidate
                break
        if solver_case is None and idx < len(solved.cases):
            solver_case = solved.cases[idx]
        if solver_case is not None:
            used.add(id(solver_case))
        pairs.append((model_case, solver_case))
    for candidate in solved.cases:
        if id(candidate) not in used:
            pairs.append((None, candidate))
    return pairs


def _wm_case_rows(y: int, solved: "SolveResult", nl) -> tuple[str, int]:
    header, y = _wm_light_section(y, "03", "CASE ANALYSIS", WM_AMBER)
    parts = [header]
    intro = "The base regime decides whether the inequality keeps its direction or reverses it."
    for line in _wrap(intro, 110):
        parts.append(_wm_text(WM_M, y, line, 12, WM_SOFT))
        y += 17
    y += 8

    from app.utils.math_verify import _factored_display, display_relation

    for idx, (model_case, solver_case) in enumerate(_wm_case_pairs(nl, solved)):
        accent = WM_AMBER if idx == 0 else WM_TEAL
        tint = WM_AMBER_TINT if idx == 0 else WM_TEAL_TINT
        top = y
        body: list[str] = []
        body.append(_wm_rect(WM_M, top, WM_CW, 10, 10, tint))
        body.append(_wm_rect(WM_M + 14, top + 14, 54, 20, 5, accent))
        body.append(_wm_text(WM_M + 41, top + 28, f"CASE {idx + 1}", 8.5, "#ffffff", "700", "middle"))
        cond = (model_case.condition.strip() if model_case is not None else "")
        condition = f"If {cond}:" if cond else "This branch"
        body.append(_wm_text(WM_M + 80, top + 28, condition, 15, INK, "700"))
        direction = (
            "base > 1 → keep direction"
            if solver_case is not None and solver_case.regime_kind == ">1"
            else "0 < base < 1 → flip direction"
        )
        body.append(_wm_text(VIEW_W - WM_M - 14, top + 24, direction, 9.5, accent, "700", "end"))
        branch = (
            model_case.solution.strip()
            if model_case is not None and model_case.solution.strip()
            else (_wm_set_text(solver_case.solution) if solver_case is not None and solver_case.solution is not None else "")
        )
        if branch:
            body.append(_wm_text(VIEW_W - WM_M - 14, top + 43, f"x ∈ {branch}", 11, accent, "700", "end"))

        ty = top + 58
        transformed = (model_case.transformed.strip() if model_case is not None else "")
        if transformed:
            for j, line in enumerate(_wrap(_wm_math(transformed), 68)):
                if j == 0:
                    body.append(_wm_text(WM_M + 14, ty, "TRANSFORM", 7.5, WM_SOFT, "700", "start", 1.0))
                body.append(_wm_text(WM_M + 86, ty, line, 13, INK, "600"))
                ty += 18
        if solver_case is not None:
            simplified = display_relation(solver_case.transformed_num, solver_case.transformed_rel)
            factored = _factored_display(solver_case.transformed_num)
            if factored:
                simplified = f"{simplified}   →   {factored} {_wm_rel(solver_case.transformed_rel)} 0"
            for j, line in enumerate(_wrap(simplified, 68)):
                if j == 0:
                    body.append(_wm_text(WM_M + 14, ty, "SIMPLIFY", 7.5, WM_SOFT, "700", "start", 1.0))
                body.append(_wm_text(WM_M + 86, ty, line, 12, accent, "700"))
                ty += 17
            raw = solver_case.raw_solution
            if raw is not None and solver_case.solution is not None and not raw.equals(solver_case.solution):
                body.append(
                    _wm_text(WM_M + 86, ty + 2, f"from the transformation: {_wm_phrase(raw)}", 9, WM_SOFT)
                )
                ty += 16
        card_h = ty - top + 6
        body[0] = _wm_rect(WM_M, top, WM_CW, card_h, 12, "#ffffff", WM_LINE, 1)
        body.insert(1, _wm_rect(WM_M, top, 5, card_h, 2.5, accent))
        parts.append("".join(body))
        y = top + card_h + 10
    return "".join(parts), y


def _wm_key_ideas_light(y: int, solved: "SolveResult", number: str = "04") -> tuple[str, int]:
    header, y = _wm_light_section(y, number, "WHY THIS WORKS", WM_INDIGO)
    parts = [header]
    if solved.base_display and not solved.constant_base:
        ideas = [
            ("01", "Domain first", "Argument > 0, base > 0 and base ≠ 1 hold at once.", WM_INDIGO),
            ("02", "Base decides", "base > 1 keeps the sign; 0 < base < 1 reverses it.", WM_AMBER),
            ("03", "Intersect, then union", "Keep only the parts inside the domain, then combine.", WM_TEAL),
        ]
    elif solved.base_display:
        ideas = [
            ("01", "Domain first", "The fixed base still needs a positive argument.", WM_INDIGO),
            ("02", "Direction is fixed", "A constant base never flips the inequality.", WM_AMBER),
            ("03", "Then solve", "Solve, then keep only the part inside the domain.", WM_TEAL),
        ]
    else:
        ideas = [
            ("01", "Critical points", "Roots and poles split the number line into intervals.", WM_INDIGO),
            ("02", "Test each interval", "One test point decides where the inequality holds.", WM_AMBER),
            ("03", "Endpoint rule", "Filled for ≤ / ≥, hollow for < / >.", WM_TEAL),
        ]
    col_w = (WM_CW - 24) / 3
    band_h = 82
    parts.append(_wm_rect(WM_M, y, WM_CW, band_h, 12, "#f8fafc"))
    for i, (num, title, body_text, color) in enumerate(ideas):
        cx = WM_M + 16 + i * (col_w + 6)
        parts.append(_wm_text(cx, y + 20, num, 8.5, color, "700", "start", 1.2))
        parts.append(_wm_text(cx, y + 38, title, 11.5, INK, "700"))
        for j, line in enumerate(_wrap(body_text, 30)[:2]):
            parts.append(_wm_text(cx, y + 54 + j * 12, line, 8.8, WM_SOFT))
    return "".join(parts), y + band_h + 12


def _render_number_line_scene(spec: DeterministicVisual, scene: VisualScene, y: int) -> tuple[str, int]:
    """One composed worked-math poster, fitted to its verified content.

    The visual is a single aligned solution map (domain, both base regimes,
    final union) followed by the exact derivation. It is intentionally not a
    vertical card dump: the map is the anchor, the sections explain it.
    """
    nl = scene.number_line
    assert nl is not None
    # Defense in depth: this layout carries the "MATHEMATICALLY VERIFIED"
    # claim, so it re-runs the exact verifier itself. A scene that no longer
    # verifies can never be drawn as verified, regardless of caller mistakes.
    from app.utils.math_verify import solve_problem, verify_number_line

    if verify_number_line(nl):
        logger.warning(
            "number_line scene failed re-verification at render time; refusing to draw"
        )
        return "", y
    solved = solve_problem(nl.problem)
    if not solved.supported or solved.solution is None:
        logger.warning("number_line re-solve failed at render time; refusing to draw")
        return "", y

    lo, hi = _wm_scales(solved, nl)
    parts: list[str] = []
    title = (spec.title or scene.title or "Worked Mathematics").strip()
    is_log = bool(solved.base_display)

    if is_log and not solved.constant_base:
        key_idea = (
            "A variable base changes the inequality direction: solve base > 1 and "
            "0 < base < 1 separately, then intersect each branch with the domain."
        )
    elif is_log:
        key_idea = "The base is fixed, so the direction is fixed: solve, then keep only the domain."
    else:
        key_idea = (
            "Roots split the number line into intervals; test each interval and "
            "include an endpoint only when the inequality is non-strict."
        )
    html, y = _wm_infographic_header(title, _wm_problem(nl.problem), key_idea)
    parts.append(html)
    y += 18

    html, y = _wm_solution_map(y, solved, nl, lo, hi)
    parts.append(html)
    y += 10

    from app.utils.math_intervals import Interval
    from app.utils.math_intervals import IntervalSet as _IntervalSet

    whole_line = _IntervalSet.of(Interval(None, None, False, False))
    needs_domain = solved.domain is not None and not solved.domain.equals(whole_line)
    section_no = 2
    if needs_domain:
        html, y = _wm_domain_constraints(y, solved)
        parts.append(html)
        section_no += 1

    if nl.cases and solved.cases:
        html, y = _wm_case_rows(y + 8, solved, nl)
        parts.append(html)
        section_no += 1

    html, y = _wm_key_ideas_light(y + 8, solved, f"{section_no:02d}")
    parts.append(html)
    section_no += 1

    if scene.caption.strip():
        cap_html, y = _wm_light_section(y + 4, f"{section_no:02d}", "WHAT THE VISUAL SHOWS", WM_SOFT)
        parts.append(cap_html)
        for line in _wrap(scene.caption.strip(), 100):
            parts.append(_wm_text(WM_M, y, line, 12, INK))
            y += 17
        y += 4

    line_y = y + 6
    parts.append(f'<line x1="{WM_M}" y1="{line_y}" x2="{VIEW_W - WM_M}" y2="{line_y}" stroke="{WM_LINE}" stroke-width="1"/>')
    parts.append(_wm_text(WM_M, line_y + 18, "Exact interval algebra  ·  verified at render time", 9, WM_SOFT, "400", "start", 0.4))
    y = line_y + 28
    return "".join(parts), y


def _render_scene(scene: VisualScene) -> tuple[str, int]:
    """Dispatch a VisualScene to its layout. Returns (svg_html, next_y)."""
    parts: list[str] = []
    legend: list[str] = []
    relation = None
    if scene.scene_kind == "force_diagram":
        diagram, y_after, legend, relation = _render_force_diagram(scene)
    elif scene.scene_kind == "process_flow":
        diagram, y_after, legend, relation = _render_flow(scene)
    elif scene.scene_kind == "plot":
        diagram, y_after, legend, relation = _render_plot(scene)
    elif scene.scene_kind == "comparison":
        diagram, y_after, legend, relation = _render_comparison(scene)
    elif scene.scene_kind == "generic":
        diagram, y_after, legend, relation = _render_generic(scene, fallback_title="")
    else:
        diagram, y_after, legend, relation = "", STAGE_Y + STAGE_H + 24, [], None
    if not diagram.strip():
        return "", y_after

    parts.append(
        f'<rect x="{STAGE_X}" y="{STAGE_Y}" width="{STAGE_W}" height="{STAGE_H}" rx="14" '
        f'fill="{BG_STAGE}" stroke="{STAGE_STROKE}" stroke-width="1.5"/>'
    )
    parts.append(diagram)

    # Long angle/arc captions live BELOW the stage (never inline over geometry).
    if legend:
        legend_hdr, y_after = _section_header(y_after, "WHAT EACH SYMBOL MEANS")
        parts.append(legend_hdr)
        for line in legend:
            parts.append(
                f'<text x="{MARGIN + 4}" y="{y_after}" font-family="{_FONT}" font-size="14" '
                f'fill="{MUTED}" text-anchor="start">{_esc(line)}</text>'
            )
            y_after += 21
        y_after += 6

    if relation is not None and relation.expression.strip():
        rel_html, y_after = _equation_card(
            y_after,
            relation.expression.strip(),
            relation.caption.strip(),
        )
        parts.append(rel_html)

    if scene.caption.strip():
        cap_html, y_after = _section_header(y_after, "WHAT THE VISUAL SHOWS")
        parts.append(cap_html)
        for line in _wrap(scene.caption.strip(), 90):
            parts.append(
                f'<text x="{MARGIN}" y="{y_after}" font-family="{_FONT}" font-size="16" '
                f'fill="{INK}" text-anchor="start">{_esc(line)}</text>'
            )
            y_after += 22
        y_after += 6

    return "".join(parts), y_after


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def render_deterministic_visual(spec: DeterministicVisual) -> str:
    """Render a bounded DeterministicVisual into a sanitized, deterministic SVG.

    If a `scene` is present it becomes the centerpiece (real diagram from
    universal primitives). Otherwise the classic study card (equations +
    steps + points) is rendered. Returns "" if there is nothing meaningful.
    """
    parts: list[str] = ["<title>Educational visual</title>"]

    meaningful = any(eq.expression.strip() for eq in spec.equations) or any(
        s.strip() for s in spec.steps
    ) or any(p.strip() for p in spec.points) or (spec.scene is not None) or bool((spec.title or "").strip())
    if not meaningful:
        return ""

    # Worked mathematics owns its full study-sheet header (title, question,
    # key idea), so dispatch it BEFORE the generic centered title. It is still
    # content-fitted: the height is computed from the verified sections.
    if (
        spec.scene is not None
        and spec.scene.scene_kind == "number_line"
        and spec.scene.number_line is not None
    ):
        html, y = _render_number_line_scene(spec, spec.scene, 16)
        if not html.strip():
            return ""
        parts.append(html)
        return _assemble_svg(parts, max(320, y + 8))

    y = 56
    title = (spec.title or "Key concept").strip()
    html, y = _title(y, title)
    parts.append(html)

    scene_rendered = False
    if spec.scene is not None:
        scene_html, y_after = _render_scene(spec.scene)
        if scene_html:
            parts.append(scene_html)
            y = y_after
            scene_rendered = True
    # Guaranteed visual: if no scene was rendered but we have title/points/equations,
    # synthesize a generic explanatory diagram so every upload yields an image.
    if not scene_rendered and (spec.title.strip() or spec.points or spec.equations or spec.steps):
        callouts: list[str] = []
        for p in spec.points[:4]:
            if p.strip():
                callouts.append(p.strip())
        for eq in spec.equations[:2]:
            if eq.expression.strip():
                callouts.append(eq.expression.strip())
        for s in spec.steps[:2]:
            if s.strip():
                callouts.append(s.strip())
        callouts = callouts[:4]
        if not callouts:
            callouts = [(spec.title or "Key concept").strip()]
        synth_scene = VisualScene(
            scene_kind="generic",
            title=title,
            caption="",
            generic=VisualGeneric(central_label=title, callouts=callouts),
        )
        scene_html, y_after = _render_scene(synth_scene)
        if scene_html:
            parts.append(scene_html)
            y = y_after
            scene_rendered = True

    # Explain Visually is a VISUAL ARTIFACT, not a second study sheet.
    # When a scene is the centerpiece we render ONLY the scene geometry
    # (+ its caption, symbol legend and relation equation, which are part of
    # the scene itself). Detailed Key Formulas / Understand It / Key Points
    # live in the normal study sections and must NOT be duplicated inside the
    # white visual. The card layout is used only when no scene exists.
    if scene_rendered:
        pass
    else:
        if spec.equations:
            html, y = _section_header(y, "KEY EQUATIONS")
            parts.append(html)
            for eq in spec.equations:
                if not eq.expression.strip():
                    continue
                html, y = _equation_card(y, eq.expression.strip(), eq.meaning.strip())
                parts.append(html)

        if spec.steps:
            html, y = _section_header(y, "HOW IT WORKS")
            parts.append(html)
            step_y = y
            for i, step in enumerate(spec.steps, start=1):
                if not step.strip():
                    continue
                html, step_y = _step_box(step_y, i, step.strip())
                parts.append(html)
                if i < len(spec.steps):
                    parts.append(_step_arrow(step_y + 8))
                    step_y += 34
            y = step_y + 16

        if spec.points:
            html, y = _section_header(y, "KEY POINTS")
            parts.append(html)
            for point in spec.points:
                if not point.strip():
                    continue
                html, y = _point_line(y, point.strip())
                parts.append(html)

    height = max(860, y + 32)
    return _assemble_svg(parts, height)


def _assemble_svg(parts: list[str], height: int) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {height}" width="100%">'
    svg += f'<rect x="0" y="0" width="{VIEW_W}" height="{height}" rx="16" fill="white"/>'
    svg += "".join(parts)
    svg += "</svg>"
    svg = sanitize_svg(svg)
    if not svg:
        return ""
    return svg


def _hero_diagram_bbox(diagram: str) -> tuple[float, float, float, float] | None:
    """Approximate the drawn content bounds so a wide, short hero can be cropped.

    The stage is a fixed 736x592 frame; a process-flow diagram only occupies a
    horizontal band in the middle, leaving large blank top/bottom margins when
    composed. Text bounds are estimated from font size (deterministic; no font
    measurement). Paths are skipped - axes/lines already establish plot bounds.
    """
    try:
        root = ElementTree.fromstring(f"<g>{diagram}</g>")
    except ElementTree.ParseError:
        return None

    xs: list[float] = []
    ys: list[float] = []

    def add(x0: float, y0: float, x1: float, y1: float) -> None:
        xs.extend((x0, x1))
        ys.extend((y0, y1))

    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        attrs = elem.attrib
        try:
            if tag == "rect":
                x, y = float(attrs["x"]), float(attrs["y"])
                add(x, y, x + float(attrs["width"]), y + float(attrs["height"]))
            elif tag == "circle":
                cx, cy, r = float(attrs["cx"]), float(attrs["cy"]), float(attrs["r"])
                add(cx - r, cy - r, cx + r, cy + r)
            elif tag == "ellipse":
                cx, cy = float(attrs["cx"]), float(attrs["cy"])
                rx, ry = float(attrs["rx"]), float(attrs["ry"])
                add(cx - rx, cy - ry, cx + rx, cy + ry)
            elif tag in ("line",):
                add(
                    float(attrs["x1"]), float(attrs["y1"]),
                    float(attrs["x2"]), float(attrs["y2"]),
                )
            elif tag in ("polygon", "polyline"):
                points = [float(v) for v in attrs.get("points", "").split()]
                for i in range(0, len(points) - 1, 2):
                    add(points[i], points[i + 1], points[i], points[i + 1])
            elif tag == "text":
                x, y = float(attrs["x"]), float(attrs["y"])
                size = float(attrs.get("font-size", 12))
                text = "".join(elem.itertext())
                width = max(size * 0.62 * len(text), size)
                anchor = attrs.get("text-anchor", "start")
                if anchor == "middle":
                    x -= width / 2
                elif anchor == "end":
                    x -= width
                add(x, y - size, x + width, y + size * 0.25)
        except (KeyError, ValueError, TypeError):
            continue

    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def render_hero_geometry(spec: DeterministicVisual) -> str:
    """Geometry-only hero for composition engines (e.g. Typst).

    Generic across scene kinds: keeps the stage rect + diagram geometry
    (vectors, axes, curves, labels that belong to the geometry itself) and
    drops ALL prose — no title, no legend, no relation/explanation cards, no
    caption, no supplements. The composition layer (Typst) owns 100% of text.
    Returns "" when there is no scene to draw.
    """
    if spec.scene is None:
        return ""
    scene = spec.scene
    if scene.scene_kind == "force_diagram":
        diagram, _, _, _ = _render_force_diagram(scene)
    elif scene.scene_kind == "process_flow":
        diagram, _, _, _ = _render_flow(scene)
    elif scene.scene_kind == "plot":
        diagram, _, _, _ = _render_plot(scene)
    elif scene.scene_kind == "generic":
        diagram, _, _, _ = _render_generic(scene, fallback_title="")
    elif scene.scene_kind == "comparison":
        diagram, _, _, _ = _render_comparison(scene)
    else:
        return ""
    if not diagram.strip():
        return ""
    pad = 16
    vx, vy = STAGE_X - pad, STAGE_Y - pad
    vw, vh = STAGE_W + 2 * pad, STAGE_H + 2 * pad
    bbox = _hero_diagram_bbox(diagram)
    if bbox is not None:
        _, by, bw, bh = bbox
        if bw >= STAGE_W * 0.5 and bh < STAGE_H * 0.72:
            top = max(vy, by - 18)
            bottom = min(vy + vh, by + bh + 18)
            if bottom - top > 80:
                vy, vh = top, bottom - top
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx} {vy} {vw} {vh}" width="100%">'
        f'<rect x="{vx}" y="{vy}" width="{vw}" height="{vh}" rx="14" fill="white"/>'
        f'<rect x="{STAGE_X}" y="{STAGE_Y}" width="{STAGE_W}" height="{STAGE_H}" rx="14" '
        f'fill="{BG_STAGE}" stroke="{STAGE_STROKE}" stroke-width="1.5"/>'
        f"{diagram}</svg>"
    )
    svg = sanitize_svg(svg)
    return svg or ""


def _step_arrow(y_mid: int) -> str:
    """Downward arrow between two step boxes (polygon arrowhead)."""
    cx = VIEW_W / 2
    return (
        f'<line x1="{cx:.0f}" y1="{y_mid}" x2="{cx:.0f}" y2="{y_mid + 18}" stroke="{ACCENT}" stroke-width="2"/>'
        f'<polygon points="{cx - 5:.0f},{y_mid + 22} {cx + 5:.0f},{y_mid + 22} {cx:.0f},{y_mid + 32}" fill="{ACCENT}"/>'
    )


def _step_box(y: int, index: int, text: str) -> tuple[str, int]:
    lines = _wrap(text, 44)
    box_h = max(48, 24 + 24 * len(lines))
    cx = VIEW_W / 2
    box_w = min(CONTENT_W - 120, 520)
    x0 = cx - box_w / 2

    parts = [
        f'<rect x="{x0:.0f}" y="{y}" width="{box_w:.0f}" height="{box_h}" rx="10" fill="{CARD_FILL}" stroke="{ACCENT}" stroke-width="1.5"/>',
        f'<circle cx="{x0 + 22:.0f}" cy="{y + box_h / 2:.0f}" r="13" fill="{ACCENT}"/>',
        f'<text x="{x0 + 22:.0f}" y="{y + box_h / 2 + 5:.0f}" font-family="{_FONT}" font-size="14" font-weight="700" fill="#ffffff" text-anchor="middle">{index}</text>',
    ]
    ty = y + 14 + box_h / 2 - (len(lines) - 1) * 12
    for line in lines:
        parts.append(
            f'<text x="{cx:.0f}" y="{ty:.0f}" font-family="{_FONT}" font-size="14" font-weight="600" '
            f'fill="{INK}" text-anchor="middle">{_esc(line)}</text>'
        )
        ty += 24
    return "".join(parts), y + box_h
