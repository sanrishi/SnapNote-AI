"""Deterministic mathematics verification (math_verify).

The regression under test: log_{x+3}(x² − x) < 1 was once rendered with the
wrong final union (-3, -2) ∪ (-1, 3) (domain intersection lost). The verifier
must re-solve such problems exactly and refuse anything that disagrees.
"""
from fractions import Fraction

from app.models.schemas import (
    VisualInterval,
    VisualMathCase,
    VisualNumberLine,
    VisualNumberLineTrack,
)
from app.utils.math_intervals import format_interval_set
from app.utils.math_verify import solve_problem, verify_number_line

PROBLEM = "log_{x+3}(x^2 - x) < 1"
UNICODE_PROBLEM = "log_{x+3}(x\u00b2 \u2212 x) < 1"


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
        ),
        cases=[
            VisualMathCase(
                condition="x + 3 > 1",
                transformed="(x^2 - x) < x + 3",
                solution="(-1, 0) \u222a (1, 3)",
            ),
            VisualMathCase(
                condition="0 < x + 3 < 1",
                transformed="(x^2 - x) > x + 3",
                solution="(-3, -2)",
            ),
        ],
        solution=_track(
            [
                ("-3", "-2", False, False),
                ("-1", "0", False, False),
                ("1", "3", False, False),
            ]
        ),
    )


def test_solve_regression_exact():
    r = solve_problem(PROBLEM)
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-3, -2) \u222a (-1, 0) \u222a (1, 3)"
    assert format_interval_set(r.domain) == "(-3, -2) \u222a (-2, 0) \u222a (1, inf)"
    assert r.excluded_points == (Fraction(-2),)
    assert [c.condition for c in r.cases] == ["x + 3 > 1", "0 < x + 3 < 1"]
    assert format_interval_set(r.cases[0].solution) == "(-1, 0) \u222a (1, 3)"
    assert format_interval_set(r.cases[1].solution) == "(-3, -2)"
    assert r.cases[0].transformed == "(x^2 - x) < x + 3"
    assert r.cases[1].transformed == "(x^2 - x) > x + 3"
    assert not r.constant_base


def test_unicode_problem_solves_identically():
    r = solve_problem(UNICODE_PROBLEM)
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-3, -2) \u222a (-1, 0) \u222a (1, 3)"
    assert r.problem == PROBLEM  # prepared form is stable


def test_unicode_subscript_base_solves():
    """Live-caught regression: Gemini emits the base as unicode subscripts
    (logx₊₃(...)); the math is correct and must verify, not refuse."""
    r = solve_problem("logx\u208a\u2083(x\u00b2 \u2212 x) < 1")
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-3, -2) \u222a (-1, 0) \u222a (1, 3)"
    assert format_interval_set(r.domain) == "(-3, -2) \u222a (-2, 0) \u222a (1, inf)"
    assert [c.condition for c in r.cases] == ["x + 3 > 1", "0 < x + 3 < 1"]


def test_subscript_style_spec_verifies():
    nl = _regression_nl()
    nl.problem = "logx\u208a\u2083(x\u00b2 \u2212 x) < 1"
    assert verify_number_line(nl) == []


def test_bare_log_without_base_still_unsupported():
    r = solve_problem("log(x - 1) < 3")
    assert not r.supported
    assert "log_{base}(argument)" in r.reason


def test_correct_regression_spec_verifies():
    assert verify_number_line(_regression_nl()) == []


def test_wrong_union_refused_with_exact_correction():
    nl = _regression_nl()
    nl.solution = _track([("-3", "-2", False, False), ("-1", "3", False, False)])
    errors = verify_number_line(nl)
    assert errors == [
        "the final answer (-3, -2) \u222a (-1, 3) is mathematically wrong for "
        "'log_{x+3}(x^2 - x) < 1'; the correct answer is "
        "(-3, -2) \u222a (-1, 0) \u222a (1, 3)"
    ]


def test_missing_domain_exclusion_refused():
    nl = _regression_nl()
    nl.domain = _track(
        [("-3", "0", False, False), ("1", "inf", False, False)]
    )
    errors = verify_number_line(nl)
    assert any("the domain (-3, 0) \u222a (1, inf) is mathematically wrong" in e for e in errors)
    assert any("(-3, -2) \u222a (-2, 0) \u222a (1, inf)" in e for e in errors)


def test_case_solution_mismatch_refused():
    nl = _regression_nl()
    nl.cases[1].solution = "(-3, 0)"
    errors = verify_number_line(nl)
    assert any("does not match the computed case solutions" in e for e in errors)


def test_swapped_case_condition_refused():
    nl = _regression_nl()
    nl.cases[0].condition = "0 < x + 3 < 1"
    nl.cases[1].condition = "x + 3 > 1"
    errors = verify_number_line(nl)
    assert any("claims base < 1" in e or "claims base > 1" in e for e in errors)


def test_wrong_transformed_step_refused():
    nl = _regression_nl()
    nl.cases[0].transformed = "(x^2 - x) > x + 3"
    errors = verify_number_line(nl)
    assert any("yields (-2, -1) \u222a (3, inf) on its branch" in e for e in errors)


def test_missing_case_branch_refused():
    nl = _regression_nl()
    nl.cases = nl.cases[:1]
    errors = verify_number_line(nl)
    assert any("expected 2 case branch(es)" in e for e in errors)
    assert any("missing case branch(es)" in e for e in errors)


def test_unparseable_condition_refused():
    nl = _regression_nl()
    nl.cases[0].condition = "x + 3 > 5"
    errors = verify_number_line(nl)
    assert any("could not be parsed" in e for e in errors)


def test_constant_base_needs_no_case_split():
    r = solve_problem("log_2(x - 1) < 3")
    assert r.supported, r.reason
    assert r.constant_base
    assert format_interval_set(r.solution) == "(1, 9)"
    assert format_interval_set(r.domain) == "(1, inf)"
    nl = VisualNumberLine(
        problem="log_2(x - 1) < 3",
        domain=_track([("1", "inf", False, False)]),
        solution=_track([("1", "9", False, False)]),
    )
    assert verify_number_line(nl) == []
    nl.cases = [VisualMathCase(condition="2 > 1", transformed="x < 9", solution="(1, 9)")]
    errors = verify_number_line(nl)
    assert any("no case split to show" in e for e in errors)


def test_negative_constant_reverses_multiplication():
    r = solve_problem("log_2(x - 1) >= -1")
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "[3/2, inf)"
    nl = VisualNumberLine(
        problem="log_2(x - 1) >= -1",
        domain=_track([("1", "inf", False, False)]),
        solution=_track([("3/2", "inf", True, False)]),
    )
    assert verify_number_line(nl) == []


def test_base_between_zero_and_one_case_reversal():
    r = solve_problem("log_{1 - x}(x + 2) < 1")
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-2, -1/2) \u222a (0, 1)"
    assert format_interval_set(r.domain) == "(-2, 0) \u222a (0, 1)"
    assert r.excluded_points == (Fraction(0),)
    conditions = {c.condition for c in r.cases}
    assert conditions == {"1 - x > 1", "0 < 1 - x < 1"}
    nl = VisualNumberLine(
        problem="log_{1 - x}(x + 2) < 1",
        domain=_track(
            [("-2", "0", False, False), ("0", "1", False, False)],
        ),
        cases=[
            VisualMathCase(
                condition="1 - x > 1",
                transformed="(x + 2) < 1 - x",
                solution="(-2, -1/2)",
            ),
            VisualMathCase(
                condition="0 < 1 - x < 1",
                transformed="(x + 2) > 1 - x",
                solution="(0, 1)",
            ),
        ],
        solution=_track(
            [("-2", "-1/2", False, False), ("0", "1", False, False)]
        ),
    )
    assert verify_number_line(nl) == []


def test_polynomial_inequalities():
    r = solve_problem("x^2 - 5x + 6 > 0")
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-inf, 2) \u222a (3, inf)"


def test_rational_inequality_keeps_closed_endpoint():
    r = solve_problem("(x - 2)/(x + 3) <= 0")
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-3, 2]"
    assert r.excluded_points == (Fraction(-3),)
    nl = VisualNumberLine(
        problem="(x - 2)/(x + 3) <= 0",
        domain=_track([("-inf", "-3", False, False), ("-3", "inf", False, False)]),
        solution=_track([("-3", "2", False, True)]),
    )
    assert verify_number_line(nl) == []
    wrong = VisualNumberLine(
        problem="(x - 2)/(x + 3) <= 0",
        domain=_track([("-inf", "-3", False, False), ("-3", "inf", False, False)]),
        solution=_track([("-3", "2", False, False)]),
    )
    assert verify_number_line(wrong)


def test_unsupported_problems_are_refused_honestly():
    for problem in (
        "x^2 - 2 < 0",  # irrational critical points
        "x^3 - 2 <= 0",
        "log_{sin(x)}(x + 1) < 1",  # non-polynomial base
        "log_2(x - 1) < 1/2",  # non-integer comparison constant
        "banana split < 3",  # not even an inequality
    ):
        r = solve_problem(problem)
        assert not r.supported, problem
        nl = VisualNumberLine(
            problem=problem,
            solution=_track([("-inf", "inf", False, False)]),
        )
        errors = verify_number_line(nl)
        assert errors and "cannot verify the mathematics" in errors[0]
        assert "never displayed unverified" in errors[0]


def test_unparseable_claimed_answer_refused():
    nl = VisualNumberLine(
        problem=PROBLEM,
        solution=_track([("abc", "3", False, False)]),
    )
    errors = verify_number_line(nl)
    assert any("could not be parsed as intervals" in e for e in errors)


def test_missing_problem_refused():
    assert verify_number_line(VisualNumberLine()) == [
        "number line scene is missing the problem statement"
    ]


# ── completeness: domain track, case list, exact regime conditions ─────────


def test_missing_domain_track_refused():
    """A log problem restricts the domain; omitting the domain track is an
    incomplete explanation, not a pass."""
    nl = _regression_nl()
    nl.domain = None
    errors = verify_number_line(nl)
    assert any("domain track is required" in e for e in errors)
    assert any("(-3, -2) \u222a (-2, 0) \u222a (1, inf)" in e for e in errors)


def test_missing_domain_track_refused_for_rational():
    nl = VisualNumberLine(
        problem="(x - 2)/(x + 3) <= 0",
        solution=_track([("-3", "2", False, True)]),
    )
    errors = verify_number_line(nl)
    assert any("domain track is required" in e for e in errors)


def test_polynomial_without_restrictions_needs_no_domain():
    nl = VisualNumberLine(
        problem="x^2 - 5x + 6 > 0",
        solution=_track([("-inf", "2", False, False), ("3", "inf", False, False)]),
    )
    assert verify_number_line(nl) == []


def test_missing_entire_case_list_refused():
    """An empty case list must not bypass case-count validation."""
    nl = _regression_nl()
    nl.cases = []
    errors = verify_number_line(nl)
    assert any("expected 2 case branch(es)" in e and "found 0" in e for e in errors)
    assert any("missing case branch(es)" in e for e in errors)


def test_incomplete_base_condition_omitting_positivity_refused():
    """Live-review regression: "x + 3 < 1" describes base < 1 but drops the
    required positivity bound of the 0 < base < 1 regime."""
    nl = _regression_nl()
    nl.cases[1].condition = "x + 3 < 1"
    errors = verify_number_line(nl)
    assert any("omits the positivity requirement" in e for e in errors)


def test_gt_condition_on_between_branch_refused():
    nl = _regression_nl()
    nl.cases[1].condition = "x + 3 > 1"
    errors = verify_number_line(nl)
    assert any("claims base > 1" in e and "0 < base < 1" in e for e in errors)


def test_extra_case_branch_refused():
    nl = _regression_nl()
    nl.cases.append(
        VisualMathCase(condition="x + 3 > 1", transformed="", solution="(-1, 3)")
    )
    errors = verify_number_line(nl)
    assert any("expected 2 case branch(es)" in e and "found 3" in e for e in errors)


# ── parser security: restricted grammar, no evaluation ─────────────────────


def test_parser_supports_the_required_grammar():
    from app.utils.math_verify import _parse_expr
    import sympy

    x = sympy.Symbol("x", real=True)
    cases = {
        "2x": 2 * x,
        "5x": 5 * x,
        "0.5x": x / 2,
        ".5x": x / 2,
        "x(x+1)": x * (x + 1),
        "(x+1)(x-2)": (x + 1) * (x - 2),
        "2(x-3)": 2 * (x - 3),
        "3x^2": 3 * x**2,
        "x\u00b2 \u2212 x": x**2 - x,
        "x/2": x / 2,
        "-x + 1": 1 - x,
        "x^(-2)": x**-2,
        "1 - x": 1 - x,
    }
    for text, expected in cases.items():
        parsed = _parse_expr(text)
        assert not isinstance(parsed, str), (text, parsed)
        assert sympy.simplify(parsed - expected) == 0, text


def test_parser_rejects_forbidden_syntax_without_execution(tmp_path):
    from app.utils.math_verify import _parse_expr

    marker = tmp_path / "pwned.txt"
    forbidden = (
        "__import__('os').system('echo pwned')",
        "os.system('calc')",
        f"__import__('pathlib').Path('{marker.as_posix()}').write_text('pwned')",
        "x.__class__",
        "().__class__.__bases__",
        "[x for x in range(3)]",
        "lambda: 1",
        "x if x else 1",
        "'string'",
        "f'{x}'",
        "{1: 2}",
        "open('x')",
        "x; import os",
        "1e3",
        "x # comment",
        "x`y`",
        "sin(x)",
        "sqrt(x)",
        "y + 1",
        "x @ x",
    )
    for text in forbidden:
        parsed = _parse_expr(text)
        assert isinstance(parsed, str), (text, parsed)
        assert (
            "could not parse" in parsed
            or "unsupported" in parsed
            or "empty" in parsed
            or "too long" in parsed
        ), text
    assert not marker.exists()


def test_parser_enforces_size_bounds():
    from app.utils.math_verify import _parse_expr

    assert isinstance(_parse_expr("x + " * 60 + "1"), str)  # too long
    assert isinstance(_parse_expr("x^99"), str)  # exponent bound
    assert isinstance(_parse_expr("x^(1/2)"), str)  # non-integer exponent
    assert isinstance(_parse_expr("x^x"), str)  # non-constant exponent
    assert isinstance(_parse_expr("2 3"), str)  # adjacent numbers
    assert isinstance(_parse_expr("(x+1" + ")" * 40 + "* (x+1)" * 20), str)


def test_malicious_problem_refused():
    for problem in (
        "__import__('os') < 1",
        "x + __import__('os').system('echo x') > 0",
        "os.system('echo x') <= 1",
    ):
        r = solve_problem(problem)
        assert not r.supported, problem


def test_malicious_log_operands_refused():
    for problem in (
        "log_{__import__('os')}(x) < 1",
        "log_{2}(__import__('os') - 1) < 1",
        "log_{x+3}(__import__('os')) < 1",
    ):
        r = solve_problem(problem)
        assert not r.supported, problem


def test_malicious_condition_and_step_refused():
    nl = _regression_nl()
    nl.cases[0].condition = "__import__('os').system('echo x') > 1"
    nl.cases[1].transformed = "__(x) < __import__('os').getcwd()"
    errors = verify_number_line(nl)
    assert errors
    assert any("could not be parsed" in e for e in errors)


# ── domain preservation: cancellation, restrictions, degenerate cases ──────


def test_cancelled_factor_domain_preserved():
    """((x-1)(x+1))/(x-1) must keep x != 1 even though sympy cancels it."""
    r = solve_problem("((x-1)*(x+1))/(x-1) > 0")
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-1, 1) \u222a (1, inf)"
    assert format_interval_set(r.domain) == "(-inf, 1) \u222a (1, inf)"
    assert r.excluded_points == (Fraction(1),)


def test_reciprocal_product_keeps_restriction():
    r = solve_problem("x*x^(-1) > 0")
    assert r.supported, r.reason
    assert format_interval_set(r.solution) == "(-inf, 0) \u222a (0, inf)"
    assert r.excluded_points == (Fraction(0),)


def test_difference_of_identicals_keeps_restriction():
    for problem in ("1/(x-1) - 1/(x-1) <= 0", "0/(x-1) <= 0"):
        r = solve_problem(problem)
        assert r.supported, (problem, r.reason)
        assert format_interval_set(r.solution) == "(-inf, 1) \u222a (1, inf)", problem
        assert r.excluded_points == (Fraction(1),), problem


def test_zero_polynomial_degenerate_relations():
    expectations = {
        "x-x <= 0": "(-inf, inf)",
        "x-x >= 0": "(-inf, inf)",
        "x-x < 0": "\u2205",
        "x-x > 0": "\u2205",
        "x^2 - x^2 <= 0": "(-inf, inf)",
        "x^2 - x^2 < 0": "\u2205",
    }
    for problem, expected in expectations.items():
        r = solve_problem(problem)
        assert r.supported, (problem, r.reason)
        assert format_interval_set(r.solution) == expected, problem


def test_division_by_zero_refused():
    for problem in ("1/0 > 0", "x/(x-x) > 0", "0^(-1) > 0", "1/(0) < 1"):
        r = solve_problem(problem)
        assert not r.supported, problem
        assert "division by zero" in r.reason, problem


def test_unsupported_divisors_refused_explicitly():
    for problem in ("1/(1/x) > 0", "(x+1)/(x^2 - 2) > 0"):
        r = solve_problem(problem)
        assert not r.supported, problem
        assert "cannot be verified exactly" in r.reason, problem


# ── log domain semantics: invalid bases, empty domain ──────────────────────


def test_invalid_constant_log_bases_refused():
    for problem in (
        "log_1(x) < 1",
        "log_0(x) < 1",
        "log_(-2)(x) > 0",
        "log_{x - x + 1}(x) < 1",  # zero polynomial base -> 1, must not pass
        "log_{2 - 2}(x) < 1",
    ):
        r = solve_problem(problem)
        assert not r.supported, problem
        assert "positive number different from 1" in r.reason, problem


def test_log_with_empty_domain_refused():
    r = solve_problem("log_2(-x^2 - 1) < 1")
    assert not r.supported
    assert "never defined" in r.reason


def test_valid_fraction_constant_base_supported():
    r = solve_problem("log_{1/2}(x - 1) < 2")
    assert r.supported, r.reason
    assert r.constant_base
    assert format_interval_set(r.solution) == "(5/4, inf)"
    assert format_interval_set(r.domain) == "(1, inf)"


# ── exact decimal constants (never through a Python float) ─────────────────


def test_decimal_constants_are_exact_not_floats():
    text = "0.10000000000000000555111512312578270211815834045410156251"
    exact = Fraction(text)
    r = solve_problem(f"x > {text}")
    assert r.supported, r.reason
    assert r.solution is not None
    lo = r.solution.normalized().intervals[0].lo
    assert lo == exact
    assert lo != Fraction("0.1")  # the float round-trip would collapse both

    other = "0.10000000000000000555111512312578270211815834045410156252"
    r2 = solve_problem(f"x > {other}")
    assert r2.supported, r2.reason
    assert r2.solution is not None
    lo2 = r2.solution.normalized().intervals[0].lo
    # Both literals parse to the SAME float; exact commits must still differ.
    assert lo2 == Fraction(other)
    assert lo2 != lo


# ── transformed steps: required + algebraic equivalence on the branch ──────


def test_missing_transformed_step_refused():
    nl = _regression_nl()
    nl.cases[0].transformed = ""
    errors = verify_number_line(nl)
    assert any("transformed inequality is required" in e for e in errors)


def test_transformed_equivalent_forms_accepted():
    variants = (
        "x^2 - 2x - 3 < 0",  # additive rearrangement
        "(x - 3)*(x + 1) < 0",  # same expression, factored
        "2*(x^2 - x) < 2*(x + 3)",  # positive scalar multiple
        "-(x^2 - x) > -(x + 3)",  # negative scalar -> sides swapped
        "x + 3 > x^2 - x",  # sides swapped
    )
    for transformed in variants:
        nl = _regression_nl()
        nl.cases[0].transformed = transformed
        assert verify_number_line(nl) == [], transformed


def test_transformed_branch_coincidence_refused():
    """"x < 3" holds on part of the branch only: refused with the exact
    branch-set mismatch, never silently accepted."""
    nl = _regression_nl()
    nl.cases[0].transformed = "x < 3"
    errors = verify_number_line(nl)
    assert any(
        "yields" in e and "correct case solution" in e for e in errors
    ), errors


def test_transformed_rewrite_not_rearrangement_refused():
    """'(x^2 - 2x - 3)/(x + 1)^2 < 0' solves to the exact same branch set,
    but it is a rewrite, not the canonical rearrangement of the derived
    step \u2014 set coincidence is not correctness."""
    nl = _regression_nl()
    nl.cases[0].transformed = "(x^2 - 2x - 3)/(x + 1)^2 < 0"
    errors = verify_number_line(nl)
    assert any("not an algebraic rearrangement" in e for e in errors), errors


def test_transformed_direction_flip_detected():
    from app.utils.math_intervals import parse_interval_set
    from app.utils.math_verify import (
        MathCase,
        _parse_claim_relation,
        _parse_expr,
        _transformed_derivation_mismatch,
    )

    derived_num = _parse_expr("(x^2 - x) - (x + 3)")
    assert not isinstance(derived_num, str)
    branch = parse_interval_set("(-3, -2)")
    assert not isinstance(branch, str)
    case = MathCase(
        condition="0 < x + 3 < 1",
        regime=branch,
        transformed="(x^2 - x) > x + 3",
        transformed_num=derived_num,
        solution=branch,
        transformed_rel=">",
    )
    # -(a) > -(b) means a < b: same relation on a negated multiple must be
    # refused; the correctly flipped form is accepted.
    bad = _parse_claim_relation("-(x^2 - x) > -(x + 3)")
    assert not isinstance(bad, str)
    reason = _transformed_derivation_mismatch(bad, case)
    assert reason is not None and "wrong inequality direction" in reason
    good = _parse_claim_relation("-(x^2 - x) < -(x + 3)")
    assert not isinstance(good, str)
    assert _transformed_derivation_mismatch(good, case) is None


def test_transformed_flipped_step_refused_on_scene():
    nl = _regression_nl()
    nl.cases[1].transformed = "(x^2 - x) < x + 3"  # derived form uses ">"
    errors = verify_number_line(nl)
    assert any("case step" in e for e in errors)


# ── excluded values must be visible; extra markers are mislabels ───────────


def test_marker_only_exclusion_accepted():
    """A domain set with un-split intervals + a marked point is still exact
    and visually apparent: accepted."""
    nl = _regression_nl()
    nl.domain = _track(
        [("-3", "0", False, False), ("1", "inf", False, False)],
        points=["-2"],
    )
    assert verify_number_line(nl) == []


def test_extra_exclusion_marker_refused():
    nl = _regression_nl()
    nl.solution = _track(
        [
            ("-3", "-2", False, False),
            ("-1", "0", False, False),
            ("1", "3", False, False),
        ],
        points=["5"],
    )
    errors = verify_number_line(nl)
    assert any("in the domain of the problem" in e for e in errors)
