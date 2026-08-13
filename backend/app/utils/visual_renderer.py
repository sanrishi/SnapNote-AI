"""Deterministic SVG renderer for the Explain Visually educational visual.

Hybrid architecture rule: generative AI is used for visual creativity, never
for exact mathematical typography. When exact text/symbols/relationships are
the payload, this renderer turns a bounded, structured spec (DeterministicVisual)
into a clean, flat, sanitized SVG. Code owns all layout and geometry — Gemini
only supplies content (title, equations, steps, points). The same spec ALWAYS
produces byte-identical SVG: no randomness, no time, no font measurement, no LLM.

Design constraints honoured (svg_safe allowlist):
  - No <marker> (sanitizer renames marker-end to markerEnd and it silently
    fails). Arrowheads are drawn as <polygon>.
  - No <style> block (stripped by the sanitizer). All styling is inline.
  - Every string is XML-escaped before being embedded.
"""

from __future__ import annotations

from app.models.schemas import DeterministicVisual
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

_FONT = "sans-serif"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


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


def render_deterministic_visual(spec: DeterministicVisual) -> str:
    """Render a bounded DeterministicVisual into a sanitized, deterministic SVG.

    Returns "" if there is nothing meaningful to draw (empty spec).
    """
    parts: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} 900" width="100%">']
    parts.append("<title>Educational visual</title>")

    meaningful = any(eq.expression.strip() for eq in spec.equations) or any(
        s.strip() for s in spec.steps
    ) or any(p.strip() for p in spec.points)
    if not meaningful:
        return ""

    y = 56
    title = (spec.title or "Key concept").strip()
    html, y = _title(y, title)
    parts.append(html)

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