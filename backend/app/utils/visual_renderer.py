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
    VisualObject,
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

# Diagram stage geometry (force diagram).
STAGE_X = 56
STAGE_Y = 110
STAGE_W = VIEW_W - 2 * STAGE_X
STAGE_H = 470
PIVOT_X = 240
PIVOT_Y = 470
BASE_LEN = 190
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


def _wrap(text: str, max_chars: int) -> list[str]:
    """Deterministic word-wrap (no font metrics). Breaks long tokens when needed."""
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for word in words:
        while len(word) > max_chars:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
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
    """Arc between two ray directions at `vertex`, labeled, with arrowhead."""
    if len(dirs) < 2:
        return ""
    a, b = dirs[0], dirs[1]
    diff = (b - a + 180.0) % 360.0 - 180.0
    if abs(diff) < 4.0:
        return ""
    r_arc = 34.0
    pts = _arc_points(vertex[0], vertex[1], r_arc, a, b)
    mid_deg = a + diff * 0.5
    label_pos = _polar(vertex[0], vertex[1], r_arc + 16, mid_deg)
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
    if angle.caption:
        cpts = _wrap(angle.caption, 52)
        cy = label_pos[1] + 34
        for i, cl in enumerate(cpts):
            parts.append(
                f'<text x="{_f(label_pos[0])}" y="{_f(cy + i * 16)}" font-family="{_FONT}" '
                f'font-size="12" fill="{MUTED}" text-anchor="middle">{_esc(cl)}</text>'
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
    mid = pts[len(pts) // 2]
    parts = [
        _polyline(pts, GREEN, width=2.6),
        f'<polygon points="{_arrowhead(tip[0], tip[1], tip_deg, 11)}"/>',
    ]
    if arc.label:
        lpos = (mid[0] + 16, mid[1] - 18)
        parts.append(
            f'<text x="{_f(lpos[0])}" y="{_f(lpos[1])}" font-family="{_FONT}" font-size="16" '
            f'font-weight="700" fill="{GREEN}" text-anchor="middle">{_esc(arc.label)}</text>'
        )
    if arc.caption:
        parts.append(
            f'<text x="{_f(mid[0] + 16)}" y="{_f(mid[1])}" font-family="{_FONT}" font-size="11.5" '
            f'fill="{MUTED}" text-anchor="middle">{_esc(arc.caption)}</text>'
        )
    return "".join(parts)


def _render_force_diagram(scene: VisualScene) -> tuple[str, int]:
    """Render the force/vector diagram inside the stage. Returns (svg, next_y)."""
    force: ForceDiagram | None = scene.force
    parts: list[str] = []
    if force is None:
        return "", STAGE_Y + STAGE_H + 24

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

    # Rotation arcs around the pivot.
    for arc in force.arcs:
        parts.append(_rotation_arc(arc, pivot))

    # Relation equation card below the stage.
    relation_html = ""
    y_after = STAGE_Y + STAGE_H + 18
    if force.relation and force.relation.expression.strip():
        relation_html, y_after = _equation_card(
            y_after,
            force.relation.expression.strip(),
            force.relation.caption.strip(),
        )
        parts.append(relation_html)

    return "".join(parts), y_after


def _render_flow(scene: VisualScene) -> tuple[str, int]:
    """Render a process-flow scene: labeled boxes + arrows (+ optional feedback loop)."""
    flow = scene.flow
    parts: list[str] = []
    if flow is None:
        return "", STAGE_Y + STAGE_H + 24
    nodes: list[FlowNode] = [n for n in flow.nodes if n.label.strip()]
    if not nodes:
        return "", STAGE_Y + STAGE_H + 24

    n = len(nodes)
    box_w = min(150, int((CONTENT_W - 40 - (n - 1) * 30) / n))
    box_w = max(92, box_w)
    box_h = 54
    y_center = STAGE_Y + STAGE_H // 2 - 40
    total_w = n * box_w + (n - 1) * 30
    x0 = (VIEW_W - total_w) / 2

    centers: list[tuple[float, float]] = []
    for i, node in enumerate(nodes):
        cx = x0 + i * (box_w + 30) + box_w / 2
        centers.append((cx, y_center))
        lines = _wrap(node.label, 16)
        bx = cx - box_w / 2
        by = y_center - box_h / 2
        parts.append(
            f'<rect x="{_f(bx)}" y="{_f(by)}" width="{box_w}" height="{box_h}" rx="10" '
            f'fill="{CARD_FILL}" stroke="{ACCENT}" stroke-width="1.5"/>'
        )
        ty = y_center - (len(lines) - 1) * 11
        for line in lines:
            parts.append(
                f'<text x="{_f(cx)}" y="{_f(ty)}" font-family="{_FONT}" font-size="14" '
                f'font-weight="600" fill="{INK}" text-anchor="middle">{_esc(line)}</text>'
            )
            ty += 22

    # Connectors (arrows between boxes).
    for conn in flow.connectors:
        if not (0 <= conn.source < n and 0 <= conn.target < n):
            continue
        s = centers[conn.source]
        t = centers[conn.target]
        if conn.feedback:
            # Curved return path below the row.
            pts = [
                (s[0], y_center + box_h / 2 + 8),
                (s[0] + (t[0] - s[0]) * 0.5, y_center + box_h / 2 + 52),
                (t[0], y_center + box_h / 2 + 8),
            ]
            path_pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)
            parts.append(
                f'<polyline points="{path_pts}" fill="none" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>'
            )
            if conn.label:
                parts.append(
                    f'<text x="{_f((s[0] + t[0]) / 2)}" y="{_f(y_center + box_h / 2 + 46)}" '
                    f'font-family="{_FONT}" font-size="12.5" fill="{RED}" text-anchor="middle">{_esc(conn.label)}</text>'
                )
        else:
            from_x, to_x = s[0], t[0]
            arrow_deg = 0.0 if to_x >= from_x else 180.0
            y = y_center
            parts.append(_line(from_x + box_w / 2, y, to_x - box_w / 2, y, ACCENT, width=2.2))
            tip_x = to_x - box_w / 2 if arrow_deg == 0.0 else to_x + box_w / 2
            parts.append(_arrowhead(tip_x, y, arrow_deg, 11))
            if conn.label:
                lx = (from_x + to_x) / 2
                parts.append(
                    f'<text x="{_f(lx)}" y="{_f(y - 12)}" font-family="{_FONT}" font-size="12.5" '
                    f'fill="{MUTED}" text-anchor="middle">{_esc(conn.label)}</text>'
                )

    # Relation card below.
    relation_html = ""
    y_after = y_center + box_h / 2 + 74
    if flow.relation and flow.relation.expression.strip():
        relation_html, y_after = _equation_card(
            y_after,
            flow.relation.expression.strip(),
            flow.relation.caption.strip(),
        )
    return "".join(parts), y_after


def _render_scene(scene: VisualScene) -> tuple[str, int]:
    """Dispatch a VisualScene to its layout. Returns (svg_html, next_y)."""
    parts: list[str] = []
    if scene.scene_kind == "force_diagram":
        diagram, y_after = _render_force_diagram(scene)
    elif scene.scene_kind == "process_flow":
        diagram, y_after = _render_flow(scene)
    else:
        diagram, y_after = "", STAGE_Y + STAGE_H + 24
    if not diagram.strip():
        return "", y_after

    parts.append(
        f'<rect x="{STAGE_X}" y="{STAGE_Y}" width="{STAGE_W}" height="{STAGE_H}" rx="14" '
        f'fill="{BG_STAGE}" stroke="{STAGE_STROKE}" stroke-width="1.5"/>'
    )
    parts.append(diagram)

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
    parts: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} 900" width="100%">']
    parts.append("<title>Educational visual</title>")

    meaningful = any(eq.expression.strip() for eq in spec.equations) or any(
        s.strip() for s in spec.steps
    ) or any(p.strip() for p in spec.points) or (spec.scene is not None)
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

    # When a scene is the centerpiece, the card sections become supplements.
    if scene_rendered:
        if spec.equations:
            html, y = _section_header(y, "KEY EQUATIONS")
            parts.append(html)
            for eq in spec.equations:
                if not eq.expression.strip():
                    continue
                html, y = _equation_card(y, eq.expression.strip(), eq.meaning.strip())
                parts.append(html)
        if spec.points:
            html, y = _section_header(y, "KEY POINTS")
            parts.append(html)
            for point in spec.points:
                if not point.strip():
                    continue
                html, y = _point_line(y, point.strip())
                parts.append(html)
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

    parts.append("</svg>")
    svg = sanitize_svg("".join(parts))
    if not svg:
        return ""
    return svg


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
