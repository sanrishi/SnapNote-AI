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

import math

from app.models.schemas import (
    DeterministicVisual,
    FlowConnector,
    FlowNode,
    ForceDiagram,
    VisualAngle,
    VisualCurve,
    VisualGeneric,
    VisualObject,
    VisualPlot,
    VisualScene,
    VisualVector,
)
from app.utils.svg_safe import sanitize_svg

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


def _wrap(text: str, max_chars: int, hyphenate: bool = False) -> list[str]:
    """Deterministic word-wrap (no font metrics). Breaks long tokens when needed.

    hyphenate=True marks mid-word breaks with a visible "-" (narrow flow
    boxes): "Proportional" -> ["Proporti-", "onal"]. Spec values are never
    altered — the hyphen is presentation only, like newspaper columns.
    """
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for word in words:
        while len(word) > max_chars:
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
    return f'<text x="{VIEW_W/2:.0f}" y="{y}" font-family="{_FONT}" font-size="26" font-weight="700" fill="{INK}" text-anchor="middle">{_esc(text)}</text>', y + 44


def _section_header(y: int, text: str) -> tuple[str, int]:
    return (
        f'<text x="{MARGIN}" y="{y}" font-family="{_FONT}" font-size="13" font-weight="700" '
        f'fill="{ACCENT}" text-anchor="start" letter-spacing="1">{_esc(text)}</text>',
        y + 24,
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
            f'<text x="{MARGIN + 20}" y="{ty}" font-family="{_FONT}" font-size="22" font-weight="700" '
            f'fill="{INK}" text-anchor="start">{_esc(line)}</text>'
        )
        ty += 30
    for line in meaning_lines:
        parts.append(
            f'<text x="{MARGIN + 20}" y="{ty}" font-family="{_FONT}" font-size="13" '
            f'fill="{MUTED}" text-anchor="start">{_esc(line)}</text>'
        )
        ty += 18
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
            f'<text x="{text_x}" y="{ty}" font-family="{_FONT}" font-size="15" '
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
) -> str:
    """Draw one vector: line + arrowhead + label at its midpoint."""
    deg = _vector_deg(v)
    length = BASE_LEN * max(0.4, min(1.6, v.length))
    head = _polar(tail[0], tail[1], length, deg)
    mid = _polar(tail[0], tail[1], length * 0.52, deg)
    # Label sits to the left of travel direction (perpendicular offset).
    rad = math.radians(deg)
    lx, ly = mid[0] + 14 * math.sin(rad), mid[1] + 14 * math.cos(rad)
    parts = [
        _line(tail[0], tail[1], head[0], head[1], color, width=3.2),
        _arrowhead(head[0], head[1], deg),
    ]
    if v.label:
        parts.append(
            f'<text x="{_f(lx)}" y="{_f(ly)}" font-family="{_FONT}" font-size="18" '
            f'font-weight="700" fill="{color}" text-anchor="middle">{_esc(v.label)}</text>'
        )
    return "".join(parts)


def _render_object(obj: VisualObject, x: float, y: float) -> str:
    kind = obj.kind
    parts: list[str] = []
    if kind == "disk":
        parts.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="52" fill="none" stroke="{INK}" stroke-width="2.5"/>')
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
        parts.append(
            f'<text x="{_f(x)}" y="{_f(y + 38)}" font-family="{_FONT}" font-size="15" '
            f'font-weight="700" fill="{INK}" text-anchor="middle">{_esc(obj.label)}</text>'
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
        parts.append(
            f'<text x="{_f(label_pos[0])}" y="{_f(label_pos[1])}" font-family="{_FONT}" '
            f'font-size="17" font-weight="700" fill="{ACCENT}" text-anchor="middle">{_esc(angle.label)}</text>'
        )
    return "".join(parts)


def _rotation_arc(arc: VisualArc, around: tuple[float, float]) -> str:
    """Curved rotation arrow around a point (e.g. torque ↻)."""
    r = 30.0
    start = 220.0 if arc.direction == "ccw" else -40.0
    sweep = 240.0 if arc.direction == "ccw" else -240.0
    pts = _arc_points(around[0], around[1], r, start, start + sweep, steps=28)
    tip = pts[-1]
    tip_deg = start + sweep
    mid_deg = start + sweep * 0.5
    parts = [
        _polyline(pts, GREEN, width=2.6),
        _arrowhead(tip[0], tip[1], tip_deg, 11),
    ]
    if arc.label:
        # Label sits OUTSIDE the arc ring (radius r+24 along the arc midpoint),
        # never on top of the arc line itself.
        lpos = _polar(around[0], around[1], r + 24, mid_deg)
        parts.append(
            f'<text x="{_f(lpos[0])}" y="{_f(lpos[1])}" font-family="{_FONT}" font-size="16" '
            f'font-weight="700" fill="{GREEN}" text-anchor="middle">{_esc(arc.label)}</text>'
        )
    return "".join(parts)


def _legend_line(y: int, label: str, text: str) -> str:
    """One muted legend line (label chip + explanation) below the stage."""
    parts = []
    if label:
        parts.append(
            f'<text x="{MARGIN}" y="{y}" font-family="{_FONT}" font-size="12.5" font-weight="700" '
            f'fill="{ACCENT}" text-anchor="start">{_esc(label)}</text>'
        )
        tx = MARGIN + 30
    else:
        tx = MARGIN
    parts.append(
        f'<text x="{tx}" y="{y}" font-family="{_FONT}" font-size="12.5" fill="{MUTED}" '
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
    # Fit guarantee (R1): the old max(92, ...) floor let wide rows overflow the
    # stage (8-node PID hit x0 < 0). Boxes now size from their own text and the
    # row ALWAYS fits: wrap harder / shrink gap+font until it does. Pure
    # character-count arithmetic — no font metrics, stays byte-deterministic.
    font_size = 14 if n <= 6 else 12
    # Conservative mean glyph advance for 600-weight sans (measured against
    # the real PID board: 0.58 let 8-char lines spill past box borders).
    est_char = font_size * 0.64
    gap = 30 if n <= 4 else (20 if n <= 6 else 12)
    wrapped: list[list[str]] = []
    box_w = 150
    for max_chars in (16, 14, 12, 10, 8):
        wrapped = [_wrap(node.label, max_chars, hyphenate=True) for node in nodes]
        need = max(len(line) for lines in wrapped for line in lines) * est_char + 30
        box_w = min(170, need)
        if n * box_w + (n - 1) * gap <= CONTENT_W - 40:
            break
    else:
        wrapped = [_wrap(node.label, 8, hyphenate=True) for node in nodes]
    box_w = min(box_w, (CONTENT_W - 40 - (n - 1) * gap) / n)
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
    for conn in flow.connectors:
        if not (0 <= conn.source < n and 0 <= conn.target < n):
            continue
        key = (conn.source, conn.target, bool(conn.feedback))
        if key in seen_edges:
            continue
        seen_edges.add(key)
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
            if conn.label:
                # Label BELOW the V point — never inside the node band (R2).
                parts.append(
                    f'<text x="{_f((sx + tx) / 2)}" y="{_f(vy + 18)}" '
                    f'font-family="{_FONT}" font-size="12.5" fill="{RED}" text-anchor="middle">{_esc(conn.label)}</text>'
                )
            bottom = max(bottom, vy + 34)
        elif abs(conn.target - conn.source) == 1:
            from_x, to_x = s[0], t[0]
            arrow_deg = 0.0 if to_x >= from_x else 180.0
            y = y_center
            parts.append(_line(from_x + box_w / 2, y, to_x - box_w / 2, y, ACCENT, width=2.2))
            tip_x = to_x - box_w / 2 if arrow_deg == 0.0 else to_x + box_w / 2
            parts.append(_arrowhead(tip_x, y, arrow_deg, 11))
            if conn.label:
                # Label ABOVE the boxes — the gap midpoint sits inside the
                # node band, so centering there renders through boxes (R2).
                parts.append(
                    f'<text x="{_f((from_x + to_x) / 2)}" y="{_f(y_top - 10)}" font-family="{_FONT}" font-size="12.5" '
                    f'fill="{MUTED}" text-anchor="middle">{_esc(conn.label)}</text>'
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
            if conn.label:
                parts.append(
                    f'<text x="{_f((ex + land_x) / 2)}" y="{_f(route_y - 6)}" font-family="{_FONT}" '
                    f'font-size="12.5" fill="{MUTED}" text-anchor="middle">{_esc(conn.label)}</text>'
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


def _render_plot(scene: VisualScene) -> tuple[str, int, list[str], object]:
    plot = scene.plot
    if plot is None or not plot.curves:
        return "", STAGE_Y + STAGE_H + 24, [], None
    # sanitize ranges
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
    parts.append(f'<line x1="{_f(plot_x0)}" y1="{_f(x_axis_y)}" x2="{_f(plot_x0+plot_w)}" y2="{_f(x_axis_y)}" stroke="{INK}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{_f(y_axis_x)}" y1="{_f(plot_y0)}" x2="{_f(y_axis_x)}" y2="{_f(plot_y0+plot_h)}" stroke="{INK}" stroke-width="1.5"/>')
    parts.append(_arrowhead(plot_x0+plot_w, x_axis_y, 0))
    parts.append(_arrowhead(y_axis_x, plot_y0, 90))
    # axis labels + ticks
    parts.append(f'<text x="{_f(plot_x0+plot_w+8)}" y="{_f(x_axis_y+4)}" font-family="{_FONT}" font-size="12" fill="{MUTED}" text-anchor="start">{_esc(plot.x_label or "x")}</text>')
    parts.append(f'<text x="{_f(y_axis_x+6)}" y="{_f(plot_y0-6)}" font-family="{_FONT}" font-size="12" fill="{MUTED}" text-anchor="start">{_esc(plot.y_label or "y")}</text>')
    # Origin tick dedupe (R5): when 0 sits inside both ranges the two mid
    # ticks land on top of each other at the crossing — draw one "0" instead.
    origin_inside = x_min < 0 < x_max and y_min < 0 < y_max
    x_ticks = [x_min, x_max] if origin_inside else [x_min, (x_min + x_max) / 2, x_max]
    y_ticks = [y_min, y_max] if origin_inside else [y_min, (y_min + y_max) / 2, y_max]
    for xv in x_ticks:
        parts.append(f'<text x="{_f(map_x(xv))}" y="{_f(x_axis_y+14)}" font-family="{_FONT}" font-size="10" fill="{MUTED}" text-anchor="middle">{_f(xv)}</text>')
    for yv in y_ticks:
        parts.append(f'<text x="{_f(y_axis_x-8)}" y="{_f(map_y(yv)+3)}" font-family="{_FONT}" font-size="10" fill="{MUTED}" text-anchor="end">{_f(yv)}</text>')
    if origin_inside:
        parts.append(f'<text x="{_f(y_axis_x-8)}" y="{_f(x_axis_y+14)}" font-family="{_FONT}" font-size="10" fill="{MUTED}" text-anchor="end">0</text>')

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
            label_anchors.append((segs[-1][-1], curve.label, color))
    # Generic collision-aware curve labels: labels sharing an anchor point
    # (e.g. "square ABCD" and "A(1,1)" ending at the same corner) get
    # deterministic fallback offsets so they never occupy the same box.
    # First candidate == legacy (+6,-6) placement, so non-colliding labels
    # render byte-identically to before.
    placed_boxes: list[tuple[float, float, float, float]] = []
    for (ax, ay), label, color in label_anchors:
        est_w = len(label) * 6.6 + 8
        est_h = 14.0
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
            for cx, cy in candidates:
                box = (cx, cy, est_w, est_h)
                if not any(_boxes_overlap(box, other, pad=3.0) for other in placed_boxes):
                    chosen = (cx, cy)
                    break
        placed_boxes.append((chosen[0], chosen[1], est_w, est_h))
        # text-anchor=start at box left; baseline chosen so the legacy
        # (+6,-6) candidate renders byte-identically to before.
        parts.append(f'<text x="{_f(chosen[0])}" y="{_f(chosen[1] + est_h / 2)}" font-family="{_FONT}" font-size="11" font-weight="600" fill="{color}" text-anchor="start">{_esc(label)}</text>')
    return "".join(parts), STAGE_Y + STAGE_H + 18, [], None


def _render_generic(scene: VisualScene, fallback_title: str = "") -> tuple[str, int, list[str], object]:
    generic = scene.generic
    central = ""
    callouts: list[str] = []
    if generic is not None:
        central = (generic.central_label or "").strip()
        callouts = [c.strip() for c in (generic.callouts or []) if c.strip()][:4]
    if not central:
        central = (scene.title or fallback_title or "Key Concept").strip()
    if not callouts:
        # synthesize from caption if needed — still guarantees a visual
        cap = (scene.caption or "").strip()
        if cap:
            callouts = _wrap(cap, 28)[:2]
    if not central and not callouts:
        return "", STAGE_Y + STAGE_H + 24, [], None
    # layout: centre box + up to 4 satellites
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
    # satellites
    positions = [(-150, -96), (150, -96), (-150, 96), (150, 96)]
    bw, bh = 168, 46
    for idx, txt in enumerate(callouts[:4]):
        px, py = cx + positions[idx][0], cy + positions[idx][1]
        # connector
        sx = cx + (cw/2 + 8) * (1 if px > cx else -1) if abs(px - cx) > 80 else cx
        sy = cy + (ch/2 + 6) * (1 if py > cy else -1) if abs(py - cy) > 40 else cy
        parts.append(f'<line x1="{_f(sx)}" y1="{_f(sy)}" x2="{_f(px)}" y2="{_f(py)}" stroke="{ACCENT}" stroke-width="1.4" stroke-dasharray="5 4" opacity="0.9"/>')
        parts.append(f'<rect x="{_f(px-bw/2)}" y="{_f(py-bh/2)}" width="{bw}" height="{bh}" rx="10" fill="{CARD_FILL}" stroke="{CARD_STROKE}" stroke-width="1.5"/>')
        tlines = _wrap(txt, 22)
        ty2 = py - (len(tlines) - 1) * 7
        for line in tlines:
            parts.append(f'<text x="{_f(px)}" y="{_f(ty2)}" font-family="{_FONT}" font-size="11.5" fill="{INK}" text-anchor="middle">{_esc(line)}</text>')
            ty2 += 14
    return "".join(parts), STAGE_Y + STAGE_H + 18, [], None


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
                f'<text x="{MARGIN + 4}" y="{y_after}" font-family="{_FONT}" font-size="12.5" '
                f'fill="{MUTED}" text-anchor="start">{_esc(line)}</text>'
            )
            y_after += 19
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
                f'<text x="{MARGIN}" y="{y_after}" font-family="{_FONT}" font-size="14.5" '
                f'fill="{INK}" text-anchor="start">{_esc(line)}</text>'
            )
            y_after += 20
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
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {height}" width="100%">'
    svg += f'<rect x="0" y="0" width="{VIEW_W}" height="{height}" rx="16" fill="white"/>'
    svg += "".join(parts)
    svg += "</svg>"
    svg = sanitize_svg(svg)
    if not svg:
        return ""
    return svg


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
    else:
        return ""
    if not diagram.strip():
        return ""
    pad = 16
    vx, vy = STAGE_X - pad, STAGE_Y - pad
    vw, vh = STAGE_W + 2 * pad, STAGE_H + 2 * pad
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
