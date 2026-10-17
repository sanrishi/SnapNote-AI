"""Rendered-layout regressions for worked-math number_line scenes.

The CEO requirement: the final visual must be a compact, content-fitted
number-line lesson (no fixed 860px stage), with verified solution sets only.
These tests pin the rendered SVG: exact content, endpoint semantics
(open vs closed), canvas fitting, determinism, and sanitization.
"""
import re

from app.models.schemas import (
    DeterministicVisual,
    VisualInterval,
    VisualMathCase,
    VisualNumberLine,
    VisualNumberLineTrack,
    VisualScene,
    VisualSpec,
)
from app.utils.visual_lesson import validate_legacy_visual
from app.utils.visual_renderer import ACCENT, GREEN, RED, render_deterministic_visual

PROBLEM = "log_{x+3}(x\u00b2 \u2212 x) < 1"
CAPTION = (
    "A logarithmic inequality with a variable base must satisfy the domain "
    "first; the base regime then decides whether the inequality direction flips."
)

GREEN_FILLED_DOT = f'fill="{GREEN}" stroke="{GREEN}"'
GREEN_OPEN_DOT = f'fill="white" stroke="{GREEN}"'


def _track(intervals, points=None, label=""):
    return VisualNumberLineTrack(
        label=label,
        intervals=[
            VisualInterval(lo=lo, hi=hi, lo_closed=lo_c, hi_closed=hi_c)
            for lo, hi, lo_c, hi_c in intervals
        ],
        points=points or [],
    )


def _regression_nl() -> VisualNumberLine:
    return VisualNumberLine(
        problem=PROBLEM,
        domain=_track(
            [
                ("-3", "-2", False, False),
                ("-2", "0", False, False),
                ("1", "inf", False, False),
            ],
            points=["-2"],
            label="x\u00b2 \u2212 x > 0, x + 3 > 0, x + 3 \u2260 1 (so x = \u22122 is excluded)",
        ),
        cases=[
            VisualMathCase(
                condition="x + 3 > 1",
                transformed="x\u00b2 \u2212 x < x + 3",
                solution="(-1, 0) \u222a (1, 3)",
            ),
            VisualMathCase(
                condition="0 < x + 3 < 1",
                transformed="x\u00b2 \u2212 x > x + 3",
                solution="(-3, -2)",
            ),
        ],
        solution=_track(
            [
                ("-3", "-2", False, False),
                ("-1", "0", False, False),
                ("1", "3", False, False),
            ],
            label="Final answer",
        ),
    )


def _spec(nl: VisualNumberLine) -> VisualSpec:
    return VisualSpec(
        concept="logarithmic inequality with a variable base",
        render_mode="deterministic",
        text_required=True,
        deterministic=DeterministicVisual(
            title="Log Inequality \u2014 Full Case Analysis",
            scene=VisualScene(
                scene_kind="number_line",
                title="Log Inequality \u2014 Full Case Analysis",
                caption=CAPTION,
                number_line=nl,
            ),
        ),
    )


def _svg(nl: VisualNumberLine) -> str:
    svg = render_deterministic_visual(_spec(nl).deterministic)
    assert svg, "renderer returned an empty SVG"
    return svg


def test_renders_verified_structure():
    svg = _svg(_regression_nl())
    assert "FINAL ANSWER \u2014 MATHEMATICALLY VERIFIED" in svg
    assert "DOMAIN \u2014 WHERE THE LOG IS DEFINED" in svg
    assert "CASE ANALYSIS" in svg
    assert "WHAT THE VISUAL SHOWS" in svg
    assert "log_{x+3}(x\u00b2 \u2212 x) &lt; 1" in svg
    assert "(-1, 0) \u222a (1, 3)" in svg
    assert "x \u2208 (-3, -2)" in svg
    assert "If x + 3 &gt; 1:" in svg
    assert "If 0 &lt; x + 3 &lt; 1:" in svg
    assert "x \u2260 -2" in svg


def test_open_and_closed_endpoints_drawn_exactly():
    svg = _svg(_regression_nl())
    # Six open endpoints on the final answer track; none filled.
    assert svg.count(GREEN_OPEN_DOT) == 6
    assert svg.count(GREEN_FILLED_DOT) == 0
    # Excluded point -2 marked hollow red exactly once.
    assert svg.count(f'fill="white" stroke="{RED}"') == 1
    assert svg.count("x \u2260 -2") == 1
    # Domain track uses the accent color.
    assert f'stroke="{ACCENT}"' in svg


def test_closed_endpoint_filled_dot():
    nl = VisualNumberLine(
        problem="log_2(x - 1) >= -1",
        solution=_track([("3/2", "inf", True, False)]),
    )
    svg = _svg(nl)
    assert svg.count(GREEN_FILLED_DOT) == 1
    assert svg.count(GREEN_OPEN_DOT) == 0
    assert ">1.5</text>" in svg


def test_content_fitted_height_no_fixed_stage():
    svg = _svg(_regression_nl())
    m = re.search(r'viewBox="0 0 800 (\d+)"', svg)
    assert m is not None
    height = int(m.group(1))
    assert 320 <= height < 860
    assert 'height="560"' not in svg  # no legacy fixed stage
    ys = [int(y) for y in re.findall(r"<text[^>]* y=\"(\d+)\"", svg)]
    assert ys and max(ys) <= height
    xs = [float(x) for x in re.findall(r"<text[^>]* x=\"([\d.]+)\"", svg)]
    assert xs and min(xs) >= 0 and max(xs) <= 800


def test_render_is_byte_deterministic():
    nl = _regression_nl()
    assert _svg(nl) == _svg(nl)


def test_render_is_sanitized():
    svg = _svg(_regression_nl())
    assert svg.startswith("<svg")
    assert "<script" not in svg
    assert "javascript:" not in svg
    assert "onerror" not in svg


def test_correct_spec_validates_and_wrong_spec_is_refused():
    assert validate_legacy_visual(_spec(_regression_nl())) == []
    wrong = _regression_nl()
    wrong.solution = _track([("-3", "-2", False, False), ("-1", "3", False, False)])
    errors = validate_legacy_visual(_spec(wrong))
    assert any("mathematically wrong" in e for e in errors)
