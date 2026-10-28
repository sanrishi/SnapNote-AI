"""Rendered-layout regressions for worked-math number_line scenes.

The product requirement: the final visual is a genuine landscape infographic
(1536x1024), not a portrait card stack. These tests pin the rendered SVG:
integrated sections, exact verified content, endpoint semantics (open vs
closed), canvas fitting, determinism, and sanitization.
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


def _spec(
    nl: VisualNumberLine,
    title: str = "Log Inequality \u2014 Full Case Analysis",
    caption: str = CAPTION,
) -> VisualSpec:
    return VisualSpec(
        concept="logarithmic inequality with a variable base",
        render_mode="deterministic",
        text_required=True,
        deterministic=DeterministicVisual(
            title=title,
            scene=VisualScene(
                scene_kind="number_line",
                title=title,
                caption=caption,
                number_line=nl,
            ),
        ),
    )


def _svg(nl: VisualNumberLine) -> str:
    svg = render_deterministic_visual(_spec(nl).deterministic)
    assert svg, "renderer returned an empty SVG"
    return svg


def test_renders_verified_landscape_structure():
    svg = _svg(_regression_nl())
    assert 'viewBox="0 0 1536 1024"' in svg
    assert 'viewBox="0 0 800 ' not in svg
    assert "KEY IDEAS" in svg
    assert "DOMAIN \u2014 WHERE THE LOG IS DEFINED" in svg
    assert "CASE ANALYSIS" in svg
    assert "CASE 1" in svg and "CASE 2" in svg
    assert "QUICK CHECK" in svg
    assert "WHY THIS WORKS" in svg
    assert svg.count("FINAL ANSWER \u2014 MATHEMATICALLY VERIFIED") == 1
    assert "SOLUTION MAP" not in svg
    assert "log\u2093\u208a\u2083(x\u00b2 \u2212 x) &lt; 1" in svg
    # The scene caption is integrated under the title, not dropped.
    assert "must satisfy the domain" in svg
    assert "regime then decides" in svg
    assert "x \u2208 (\u22121, 0) \u222a (1, 3)" in svg
    assert "x \u2208 (\u22123, \u22122)" in svg
    assert "If x + 3 &gt; 1:" in svg
    assert "If 0 &lt; x + 3 &lt; 1:" in svg
    assert "x \u2260 \u22122" in svg
    assert ">03</text>" in svg  # final answer follows 01 domain / 02 case analysis
    assert ">04</text>" in svg  # quick check
    assert ">05</text>" not in svg


def test_interval_phrases_use_correct_direction():
    svg = _svg(_regression_nl())
    assert "x &lt; 0 or x &gt; 1" in svg
    assert "x &gt; \u22123" in svg
    assert "\u22121 &lt; x &lt; 3" in svg
    assert "x &lt; \u22121 or x &gt; 3" in svg
    assert "x &lt; 0 or x &lt; 1" not in svg
    assert "x &lt; \u22123" not in svg
    assert "x &lt; \u22121 or x &lt; 3" not in svg


def test_open_and_closed_endpoints_drawn_exactly():
    svg = _svg(_regression_nl())
    # Six open endpoints on the final answer track; none filled.
    assert svg.count(GREEN_OPEN_DOT) == 6
    assert svg.count(GREEN_FILLED_DOT) == 0
    # The excluded point is shown on the domain track and both case tracks.
    assert svg.count(f'fill="white" stroke="{RED}"') == 3
    assert svg.count("x \u2260 \u22122") == 1
    # Domain track uses the accent color.
    assert f'stroke="{ACCENT}"' in svg


def test_closed_endpoint_filled_dot_and_constant_base_layout():
    nl = VisualNumberLine(
        problem="log_2(x - 1) >= -1",
        domain=_track([("1", "inf", False, False)], label="x \u2212 1 > 0"),
        solution=_track([("3/2", "inf", True, False)]),
    )
    svg = render_deterministic_visual(
        _spec(
            nl,
            title="Solving a Constant-Base Log Inequality",
            caption="A constant base cannot flip the inequality; only the argument's domain restricts the solution.",
        ).deterministic
    )
    assert svg.count(GREEN_FILLED_DOT) == 1
    assert svg.count(GREEN_OPEN_DOT) == 0
    # Endpoint labels stay exact rationals, never float-formatted: 3/2, not 1.5.
    assert ">3/2</text>" in svg
    assert ">1.5</text>" not in svg
    # A constant base has no base regimes, so no invented CASE 1 card.
    assert "CASE 1" not in svg
    assert "CASE ANALYSIS" not in svg
    assert "SOLUTION MAP" in svg
    assert "constant base is already valid" not in svg
    assert "base is already valid" in svg
    # The exact endpoint is a real critical point, not "none".
    assert "CRITICAL POINTS" in svg
    assert "No finite critical points" not in svg
    assert "x = 5/2" in svg
    assert "Branch check passes." in svg


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
    svg = render_deterministic_visual(
        _spec(
            nl,
            title="Solving a Rational Inequality",
            caption="A rational inequality is undefined at its pole; the sign chart then decides which interval survives.",
        ).deterministic
    )
    assert "DOMAIN \u2014 WHERE THE EXPRESSION IS DEFINED" in svg
    assert "WHERE THE LOG" not in svg
    assert "DENOMINATOR \u2260 0" in svg
    assert "x \u2260 \u22123" in svg
    assert "SOLUTION MAP" in svg
    assert "CASE ANALYSIS" not in svg


def test_landscape_canvas_contains_all_text_and_cards():
    svg = _svg(_regression_nl())
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    assert m is not None
    width, height = float(m.group(1)), float(m.group(2))
    assert (width, height) == (1536.0, 1024.0)
    assert 'height="560"' not in svg  # no legacy fixed stage

    ys = [float(y) for y in re.findall(r'<text[^>]* y="([\d.]+)"', svg)]
    assert ys and 0 <= min(ys) and max(ys) <= height
    xs = [float(x) for x in re.findall(r'<text[^>]* x="([\d.]+)"', svg)]
    assert xs and 0 <= min(xs) and max(xs) <= width

    for rect in re.finditer(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg
    ):
        x, y, w, h = (float(rect.group(i)) for i in range(1, 5))
        assert x >= 0 and y >= 0
        assert x + w <= width + 0.01
        assert y + h <= height + 0.01


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
    svg = render_deterministic_visual(
        _spec(
            nl,
            title="Solving a Constant Inequality",
            caption="The two sides are identical, so the inequality is never true.",
        ).deterministic
    )
    assert "FINAL ANSWER \u2014 MATHEMATICALLY VERIFIED" in svg
    assert "No real solutions" in svg
    assert "QUICK CHECK" in svg
    assert "No point can satisfy the inequality." in svg
    assert "empty set is the verified answer" in svg
    assert "DOMAIN \u2014 ALL REAL NUMBERS" in svg


def test_renderer_refuses_scene_that_fails_verification():
    """Defense in depth: even called directly, the renderer re-runs the exact
    verifier and refuses to draw a wrong scene as 'verified'."""
    wrong = VisualNumberLine(
        problem="x^2 - 5x + 6 > 0",
        solution=_track([("-inf", "3", False, False)]),
    )
    assert render_deterministic_visual(_spec(wrong).deterministic) == ""


def test_long_case_text_wraps_inside_canvas():
    from app.utils.visual_renderer import _esc, _wm_math, _wrap

    # 100 zero literals add length without AST depth (an added " + 0.0 - 0.0"
    # chain trips the depth bound instead, correctly). The source limit is 120
    # characters, so this is the worst case a verified expression can reach.
    long_transformed = "(x^2 - x) - (x + 3) + 0." + "0" * 85 + " < 0"
    assert 104 < len(long_transformed) <= 120
    nl = _regression_nl()
    nl.cases[0].transformed = long_transformed
    assert validate_legacy_visual(_spec(nl)) == []  # still the exact rewrite
    svg = _svg(nl)
    lines = _wrap(_wm_math(long_transformed), 104)
    assert len(lines) == 2
    for line in lines:
        assert _esc(line) in svg


def test_close_endpoint_labels_stagger_not_merge():
    """-1/100 and 1/100 land ~5px apart on the axis: both labels must render,
    on different rows — never silently merged into one."""
    nl = VisualNumberLine(
        problem="(x + 1/100)*(x - 1/100)*(x - 1) <= 0",
        solution=_track(
            [("-inf", "-1/100", False, True), ("1/100", "1", True, True)]
        ),
    )
    svg = _svg(nl)
    first = re.search(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*>\u22121/100</text>', svg)
    second = re.search(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*>1/100</text>', svg)
    assert first is not None and second is not None
    assert float(second.group(1)) - float(first.group(1)) < 34  # would collide
    assert first.group(2) != second.group(2)  # different rows: no overlap
    assert abs(float(first.group(2)) - float(second.group(2))) >= 15.0
