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
        domain=_track([("1", "inf", False, False)], label="x \u2212 1 > 0"),
        solution=_track([("3/2", "inf", True, False)]),
    )
    svg = _svg(nl)
    assert svg.count(GREEN_FILLED_DOT) == 1
    assert svg.count(GREEN_OPEN_DOT) == 0
    # Endpoint labels stay exact rationals, never float-formatted: 3/2, not 1.5.
    assert ">3/2</text>" in svg
    assert ">1.5</text>" not in svg


def test_rational_domain_header_is_not_log_specific():
    """Case 2: a rational inequality's domain track must not be labelled
    with log wording ("DOMAIN — WHERE THE LOG IS DEFINED")."""
    nl = VisualNumberLine(
        problem="(x - 2)/(x + 3) <= 0",
        domain=_track(
            [("-inf", "-3", False, False), ("-3", "inf", False, False)],
            label="x + 3 \u2260 0",
        ),
        solution=_track([("-3", "2", False, True)]),
    )
    svg = _svg(nl)
    assert "DOMAIN \u2014 WHERE THE EXPRESSION IS DEFINED" in svg
    assert "WHERE THE LOG" not in svg


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


def test_empty_solution_rendered_explicitly():
    """An inequality with no solutions renders a complete, honest scene."""
    nl = VisualNumberLine(problem="x - x < 0", solution=_track([]))
    svg = _svg(nl)
    assert "FINAL ANSWER \u2014 MATHEMATICALLY VERIFIED" in svg
    assert "No real solutions" in svg


def test_renderer_refuses_scene_that_fails_verification():
    """Defense in depth: even called directly, the renderer re-runs the exact
    verifier and refuses to draw a wrong scene as 'verified'."""
    wrong = VisualNumberLine(
        problem="x^2 - 5x + 6 > 0",
        solution=_track([("-inf", "3", False, False)]),
    )
    assert render_deterministic_visual(_spec(wrong).deterministic) == ""


def test_long_case_text_wraps_inside_canvas():
    from app.utils.visual_renderer import _esc, _wrap

    # A long, exact zero literal: adds length without AST depth (an added
    # " + 0.0 - 0.0" chain trips the depth bound instead, correctly).
    long_transformed = "(x^2 - x) - (x + 3) + 0." + "0" * 60 + " < 0"
    assert len(long_transformed) > 74  # must actually need wrapping
    nl = _regression_nl()
    nl.cases[0].transformed = long_transformed
    assert validate_legacy_visual(_spec(nl)) == []  # still the exact rewrite
    svg = _svg(nl)
    lines = _wrap(long_transformed, 74)
    assert len(lines) >= 2
    for line in lines:
        assert _esc(line) in svg


def test_close_endpoint_labels_stagger_not_merge():
    """-1/100 and 1/100 land ~5px apart on the axis: both labels must render,
    on different rows \u2014 never silently merged into one."""
    nl = VisualNumberLine(
        problem="(x + 1/100)*(x - 1/100)*(x - 1) <= 0",
        solution=_track(
            [("-inf", "-1/100", False, True), ("1/100", "1", True, True)]
        ),
    )
    svg = _svg(nl)
    first = re.search(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*>-1/100</text>', svg)
    second = re.search(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*>1/100</text>', svg)
    assert first is not None and second is not None
    assert float(second.group(1)) - float(first.group(1)) < 34  # would collide
    assert first.group(2) != second.group(2)  # different rows: no overlap
    assert abs(float(first.group(2)) - float(second.group(2))) == 20.0
