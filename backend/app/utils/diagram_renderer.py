"""Deterministic SVG renderer for polar-coordinate integration regions.

"Math, never pixels": the renderer takes a validated, canonical semantic spec
(radii + angles as math values) and converts it to SVG geometry in pure Python.
The same validated spec ALWAYS produces byte-identical SVG — no randomness, no
time, no font measurement, no LLM.

Design constraints honoured (svg_safe allowlist):
  - No <marker> (marker-end gets renamed to markerEnd by the sanitizer and then
    silently fails in browsers). Arrowheads are drawn as <polygon>.
  - No fill-rule attribute (not allow-listed). The shaded annulus uses the
    two-subpath winding trick: outer ring one direction, inner ring opposite,
    which creates the hole under the default nonzero fill rule.
  - No <style> block (stripped by the sanitizer). All styling is inline.
"""

from __future__ import annotations

import math

from app.utils.diagram_validation import CanonicalPolarRegion
from app.utils.math_normalize import display_math
from app.utils.svg_safe import sanitize_svg

VIEW_W = 520
VIEW_H = 520
CX = 260.0
CY = 260.0
OUTER_MAX_PX = 200.0

FILL = "#a78bfa"
STROKE = "#334155"
AXIS = "#94a3b8"
TEXT_COLOR = "#1e293b"
MUTED = "#64748b"


def _fmt(value: float) -> str:
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _point(r_px: float, theta: float) -> tuple[float, float]:
    return (CX + r_px * math.cos(theta), CY - r_px * math.sin(theta))


def _ring_path(r_px: float, t0: float, t1: float, reverse: bool = False) -> str:
    deg0 = math.degrees(t0)
    deg1 = math.degrees(t1)
    n = max(4, int(round(abs(deg1 - deg0))))
    angles = [deg0 + (deg1 - deg0) * i / n for i in range(n + 1)]
    if reverse:
        angles = list(reversed(angles))
    pts = [_fmt(_point(r_px, math.radians(a))[0]) + "," + _fmt(_point(r_px, math.radians(a))[1]) for a in angles]
    return "M " + " L ".join(pts) + " Z"


def _arrowhead(tip_x: float, tip_y: float, base_x: float, base_y: float, size: float = 9.0) -> str:
    dx, dy = tip_x - base_x, tip_y - base_y
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    back = (tip_x - ux * size, tip_y - uy * size)
    return (
        f"{_fmt(back[0] + px * size * 0.45)},{_fmt(back[1] + py * size * 0.45)} "
        f"{_fmt(tip_x)},{_fmt(tip_y)} "
        f"{_fmt(back[0] - px * size * 0.45)},{_fmt(back[1] - py * size * 0.45)}"
    )


def _boundary_label(radius_value: float, scale: float, anchor_deg: float, text: str) -> str:
    radius = radius_value * scale + 14
    px, py = _point(radius, math.radians(anchor_deg))
    return (
        f'<text x="{_fmt(px)}" y="{_fmt(py)}" font-family="sans-serif" font-size="15" '
        f'font-weight="600" fill="{TEXT_COLOR}" text-anchor="middle" dominant-baseline="middle">'
        f"{text}</text>"
    )


def render_polar_region(spec: CanonicalPolarRegion) -> str:
    """Render a validated polar-region spec into a sanitized, deterministic SVG."""
    scale = OUTER_MAX_PX / spec.outer_value
    inner_r = spec.inner_value * scale
    outer_r = spec.outer_value * scale
    t0, t1 = spec.theta_min_value, spec.theta_max_value
    is_full = abs((t1 - t0) - 2.0 * math.pi) < 1e-6

    parts: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {VIEW_H}" width="100%">']
    parts.append("<title>Polar integration region</title>")

    outer_ring = _ring_path(outer_r, t0, t1)
    inner_ring = _ring_path(inner_r, t0, t1, reverse=True)
    parts.append(f'<path d="{outer_ring} {inner_ring}" fill="{FILL}" opacity="0.2"/>')

    parts.append(
        f'<line x1="0" y1="{_fmt(CY)}" x2="{VIEW_W}" y2="{_fmt(CY)}" stroke="{AXIS}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{_fmt(CX)}" y1="0" x2="{_fmt(CX)}" y2="{VIEW_H}" stroke="{AXIS}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<polygon points="{_arrowhead(VIEW_W - 6, CY, VIEW_W - 26, CY)}" fill="{AXIS}"/>'
    )
    parts.append(
        f'<polygon points="{_arrowhead(CX, 6, CX, 26)}" fill="{AXIS}"/>'
    )
    parts.append(
        f'<circle cx="{_fmt(CX)}" cy="{_fmt(CY)}" r="2.5" fill="{AXIS}"/>'
    )
    parts.append(
        f'<text x="{VIEW_W - 14}" y="{CY - 12}" font-family="sans-serif" font-size="14" '
        f'fill="{TEXT_COLOR}" text-anchor="middle">x</text>'
    )
    parts.append(
        f'<text x="{CX + 14}" y="16" font-family="sans-serif" font-size="14" '
        f'fill="{TEXT_COLOR}" text-anchor="middle">y</text>'
    )

    parts.append(f'<circle cx="{_fmt(CX)}" cy="{_fmt(CY)}" r="{_fmt(inner_r)}" fill="none" stroke="{STROKE}" stroke-width="2"/>')
    parts.append(f'<circle cx="{_fmt(CX)}" cy="{_fmt(CY)}" r="{_fmt(outer_r)}" fill="none" stroke="{STROKE}" stroke-width="2"/>')

    if not is_full:
        arc_r = inner_r * 0.75
        angles = [t0 + (t1 - t0) * i / 8 for i in range(9)]
        pts = " ".join(f"{_fmt(_point(arc_r, a)[0])},{_fmt(_point(arc_r, a)[1])}" for a in angles)
        parts.append(
            f'<path d="M {pts.replace(" ", " L ")}" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4"/>'
        )
        mid = math.radians(math.degrees(t0) + (math.degrees(t1) - math.degrees(t0)) / 2.0)
        mx, my = _point(arc_r * 1.35, mid)
        parts.append(
            f'<text x="{_fmt(mx)}" y="{_fmt(my)}" font-family="sans-serif" font-size="14" '
            f'font-style="italic" fill="{MUTED}" text-anchor="middle">θ</text>'
        )

    parts.append(_boundary_label(spec.inner_value, scale, 315, f"r = {display_math(spec.inner)}"))
    parts.append(_boundary_label(spec.outer_value, scale, 45, f"r = {display_math(spec.outer)}"))

    caption = (
        f"region between r = {display_math(spec.inner)} and r = {display_math(spec.outer)}"
        f", \u03b8 from {display_math(spec.theta_min)} to {display_math(spec.theta_max)}"
    )
    parts.append(
        f'<text x="{_fmt(CX)}" y="{VIEW_H - 18}" font-family="sans-serif" font-size="14" '
        f'fill="{MUTED}" text-anchor="middle">{caption}</text>'
    )

    parts.append("</svg>")
    return sanitize_svg("".join(parts))
