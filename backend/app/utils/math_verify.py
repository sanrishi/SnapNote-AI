"""Deterministic mathematics verification for worked interval problems.

The number_line visual family is ONLY allowed to render when this module can
re-solve the problem exactly and confirm the spec's claimed domain, case
branches, and final solution interval set:

  - log inequalities  log_{b(x)}(a(x)) REL c  with polynomial b, a (rational
    coefficients, degree <= 3) and integer c — including the full domain
    reasoning (argument > 0, base > 0, base != 1), the monotonicity regime
    split (base > 1 vs 0 < base < 1, with inequality reversal), and the
    case-by-case domain intersection;
  - polynomial and rational inequalities  p(x)/q(x) REL 0  with rational
    critical points.

Anything else (irrational critical points, non-polynomial operands, unknown
relations) is reported as UNSUPPORTED and the caller refuses to display it —
never "assumed correct", never "validated".

Everything here is pure and deterministic: same problem text in, same sets
out. No LLM, no network. Solving uses sympy for exact polynomial real roots;
set algebra uses math_intervals (Fractions).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.models.schemas import VisualNumberLine, VisualNumberLineTrack
from app.utils.math_intervals import (
    Interval,
    IntervalSet,
    format_interval_set,
    parse_bound,
)


logger = logging.getLogger(__name__)

_X = sympy.Symbol("x", real=True)
_MAX_DEGREE = 3
_MAX_EXPR_LEN = 120

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

_PREP = {
    "\u2212": "-",  # minus
    "\u2013": "-",
    "\u2014": "-",
    "\u00b7": "*",
    "\u00d7": "*",
    "\u00f7": "/",
    "\u2215": "/",
    "\u00b2": "^2",
    "\u00b3": "^3",
    "\u00b9": "^1",
    "\u2070": "^0",
    "\u2074": "^4",
    "\u2075": "^5",
    "\u2076": "^6",
    "\u2077": "^7",
    "\u2078": "^8",
    "\u2079": "^9",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
}

_REVERSE_REL = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}

_LEAD_RE = re.compile(r"^\s*(?:solve|simplify)\b[^0-9x(]*", re.IGNORECASE)


# ── results ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MathCase:
    """One computed case branch of a log inequality solution."""

    condition: str
    regime: IntervalSet  # where this branch applies (inside the domain)
    transformed: str  # human-readable transformed inequality
    transformed_num: sympy.Expr  # numerator of (lhs - rhs) for the branch
    solution: IntervalSet


@dataclass
class SolveResult:
    supported: bool
    reason: str = ""
    problem: str = ""
    domain: IntervalSet | None = None
    excluded_points: tuple[Fraction, ...] = ()
    cases: tuple[MathCase, ...] = ()
    solution: IntervalSet | None = None
    constant_base: bool = False


# ── text preparation ───────────────────────────────────────────────────────


def _prep(text: str) -> str:
    out = text
    for bad, good in _PREP.items():
        out = out.replace(bad, good)
    return out


def _prepare_problem(problem: str) -> str:
    text = _prep(problem).strip()
    text = _LEAD_RE.sub("", text)
    text = text.strip().rstrip("?.!")
    return text.strip()


def _split_relation(text: str) -> tuple[str, str, str] | None:
    """Split at the first comparison at bracket depth 0."""
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            two = text[i : i + 2]
            if two in ("<=", ">="):
                return text[:i], two, text[i + 2 :]
            if ch in "<>":
                return text[:i], ch, text[i + 1 :]
        i += 1
    return None


def _balanced(text: str, start: int, open_ch: str, close_ch: str) -> int | None:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


# ── exact expression parsing ───────────────────────────────────────────────


def _parse_expr(text: str) -> sympy.Expr | str:
    t = _prep(text).strip()
    if not t:
        return "expression is empty"
    if len(t) > _MAX_EXPR_LEN:
        return f"expression is too long: {t[:40]!r}..."
    try:
        expr = parse_expr(
            t, local_dict={"x": _X}, transformations=_TRANSFORMATIONS, evaluate=True
        )
    except Exception as e:  # sympy raises assorted parse errors
        logger.debug("sympy parse failed for %r: %s", t, e)
        return f"could not parse expression {t!r}"
    if not expr.free_symbols <= {_X}:
        names = ", ".join(sorted(str(s) for s in expr.free_symbols))
        return f"expression uses unsupported symbols ({names}): {t!r}"
    return expr


def _poly(expr: sympy.Expr) -> sympy.Poly | None:
    """Polynomial in x with rational coefficients, bounded degree."""
    if not expr.free_symbols <= {_X}:
        return None
    try:
        poly = sympy.Poly(expr, _X)
    except sympy.PolynomialError:
        return None
    if poly.degree() > _MAX_DEGREE:
        return None
    for coeff in poly.all_coeffs():
        if not coeff.is_rational:
            return None
    return poly


def _rational_roots(poly: sympy.Poly) -> list[Fraction] | None:
    """All distinct real roots as exact Fractions; None when any is irrational."""
    try:
        roots = sympy.real_roots(poly)
    except Exception as e:  # sympy raises assorted errors on exotic input
        logger.debug("real_roots failed for %s: %s", poly, e)
        return None
    out: set[Fraction] = set()
    for r in roots:
        if not r.is_rational:
            return None
        out.add(Fraction(int(sympy.numer(r)), int(sympy.denom(r))))
    return sorted(out)


def _sympy_rational(value: Fraction) -> sympy.Rational:
    return sympy.Rational(value.numerator, value.denominator)


def _sign_at(expr: sympy.Expr, sample: Fraction) -> int:
    val = expr.subs(_X, _sympy_rational(sample))
    if val == 0:
        return 0
    return 1 if val > 0 else -1


def _sample_point(
    lo: Fraction | None, hi: Fraction | None, all_points: list[Fraction]
) -> Fraction:
    if lo is None and hi is None:
        base = Fraction(0)
    elif lo is None:
        base = hi - 1
    elif hi is None:
        base = lo + 1
    else:
        base = (lo + hi) / 2
    while any(p == base for p in all_points):
        base += Fraction(1, 2)
    return base


def _solve_rel(num_expr: sympy.Expr, den_expr: sympy.Expr, rel: str) -> IntervalSet | str:
    """Exact solution set of num_expr/den_expr REL 0. Error string when unsupported."""
    num_poly = _poly(num_expr)
    if num_poly is None:
        return "the inequality expands beyond the supported class (polynomials in x up to degree 3)"
    den_poly = _poly(den_expr)
    if den_poly is None:
        return "the inequality expands beyond the supported class (polynomials in x up to degree 3)"
    num_roots = _rational_roots(num_poly)
    if num_roots is None:
        return "a critical point is irrational, so the solution cannot be verified exactly"
    den_roots = _rational_roots(den_poly)
    if den_roots is None:
        return "a critical point is irrational, so the solution cannot be verified exactly"
    num_root_set = set(num_roots)
    den_root_set = set(den_roots)
    critical = sorted(num_root_set | den_root_set)
    bounds: list[Fraction | None] = [None, *critical, None]
    strict = rel in ("<", ">")
    want_neg = rel in ("<", "<=")
    segments: list[Interval] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        sample = _sample_point(a, b, critical)
        sign = _sign_at(num_expr, sample) * _sign_at(den_expr, sample)
        satisfied = sign < 0 if want_neg else sign > 0
        if not satisfied:
            continue
        lo_closed = (
            a is not None and a in num_root_set and a not in den_root_set and not strict
        )
        hi_closed = (
            b is not None and b in num_root_set and b not in den_root_set and not strict
        )
        iv = Interval(a, b, lo_closed, hi_closed).normalized()
        if not iv.is_empty():
            segments.append(iv)
    result = IntervalSet(tuple(segments)).normalized()
    if not strict:
        for r in num_roots:
            if r in den_root_set:
                continue
            covered = any(
                (iv.lo is not None and iv.lo == r) or (iv.hi is not None and iv.hi == r)
                for iv in result.normalized().intervals
            )
            if not covered:
                result = result.union(IntervalSet.of(Interval(r, r, True, True)))
    return result.normalized()


# ── log inequality solving ─────────────────────────────────────────────────


def _find_log(text: str) -> tuple[str, str, int] | None:
    """Locate log_{base}(arg); returns (base_text, arg_text, end_index)."""
    m = re.search(r"\blog\s*_", text, flags=re.IGNORECASE)
    if m is None:
        return None
    rest = text[m.end() :]
    if rest.startswith("{"):
        close = _balanced(rest, 0, "{", "}")
        if close is None:
            return None
        base_text = rest[1:close]
        after = rest[close + 1 :]
    elif rest.startswith("("):
        close = _balanced(rest, 0, "(", ")")
        if close is None:
            return None
        base_text = rest[1:close]
        after = rest[close + 1 :]
    else:
        paren = after = None
        for i, ch in enumerate(rest):
            if ch == "(":
                paren = i
                break
        if paren is None:
            return None
        base_text = rest[:paren]
        after = rest[paren:]
    after = after.lstrip()
    if not after.startswith("("):
        return None
    close = _balanced(after, 0, "(", ")")
    if close is None:
        return None
    arg_text = after[1:close]
    end = m.end() + (len(rest) - len(after)) + close + 1
    return base_text.strip(), arg_text.strip(), end


def _solve_log(text: str) -> SolveResult:
    found = _find_log(text)
    if found is None:
        return SolveResult(False, "the logarithm is not written as log_{base}(argument)")
    base_text, arg_text, end = found
    split = _split_relation(text)
    if split is None:
        return SolveResult(False, "no comparison relation (<, <=, >, >=) found")
    left, rel, right = split
    if left.strip() != text[:end].strip():
        return SolveResult(
            False, "the comparison must involve only the logarithm and a constant"
        )
    right = right.strip()
    try:
        const = Fraction(right)
    except (ValueError, ZeroDivisionError):
        return SolveResult(False, f"the comparison constant {right!r} is not a number")
    if const.denominator != 1:
        return SolveResult(False, "the comparison constant must be a whole number")
    base_expr = _parse_expr(base_text)
    if isinstance(base_expr, str):
        return SolveResult(False, base_expr)
    arg_expr = _parse_expr(arg_text)
    if isinstance(arg_expr, str):
        return SolveResult(False, arg_expr)
    if _poly(base_expr) is None or _poly(arg_expr) is None:
        return SolveResult(
            False, "the logarithm base and argument must be polynomials in x"
        )
    base_minus_1 = sympy.expand(base_expr - 1)
    if _poly(base_minus_1) is None:
        return SolveResult(False, "the logarithm base must be a polynomial in x")
    base1_roots = _rational_roots(sympy.Poly(base_minus_1, _X))
    if base1_roots is None:
        return SolveResult(
            False,
            "the base crosses 1 at an irrational point, so the case split cannot be verified",
        )
    base_pos = _solve_rel(base_expr, sympy.Integer(1), ">")
    if isinstance(base_pos, str):
        return SolveResult(False, base_pos)
    arg_pos = _solve_rel(arg_expr, sympy.Integer(1), ">")
    if isinstance(arg_pos, str):
        return SolveResult(False, arg_pos)
    domain = base_pos.intersect(arg_pos).without_points(base1_roots)
    constant_base = not base_expr.free_symbols
    base_gt1 = _solve_rel(base_minus_1, sympy.Integer(1), ">")
    if isinstance(base_gt1, str):
        return SolveResult(False, base_gt1)
    base_lt1 = _solve_rel(base_minus_1, sympy.Integer(1), "<")
    if isinstance(base_lt1, str):
        return SolveResult(False, base_lt1)
    base_str = sympy.sstr(base_expr)
    arg_str = _arg_str(arg_text)
    c = int(const)
    k = abs(c)
    cases: list[MathCase] = []
    final = IntervalSet()
    for regime_kind, regime in (
        (">1", domain.intersect(base_gt1)),
        ("<1", domain.intersect(base_lt1)),
    ):
        if regime.is_empty():
            continue
        if c >= 0:
            num_t = sympy.expand(arg_expr - base_expr**c)
            den_t = sympy.Integer(1)
            rhs_txt = base_str if c == 1 else f"({base_str})^{c}"
            transformed = f"{arg_str} {rel if regime_kind == '>1' else _REVERSE_REL[rel]} {rhs_txt}"
        else:
            num_t = sympy.expand(arg_expr * base_expr**k - 1)
            den_t = sympy.Integer(1)
            transformed = (
                f"{arg_str} * ({base_str})^{k} "
                f"{rel if regime_kind == '>1' else _REVERSE_REL[rel]} 1"
            )
        rel_t = rel if regime_kind == ">1" else _REVERSE_REL[rel]
        solved = _solve_rel(num_t, den_t, rel_t)
        if isinstance(solved, str):
            return SolveResult(False, solved)
        case_solution = solved.intersect(regime)
        if constant_base:
            condition = ""
            transformed = ""
        elif regime_kind == ">1":
            condition = f"{base_str} > 1"
        else:
            condition = f"0 < {base_str} < 1"
        cases.append(
            MathCase(
                condition=condition,
                regime=regime,
                transformed=transformed,
                transformed_num=num_t,
                solution=case_solution,
            )
        )
        final = final.union(case_solution)
    return SolveResult(
        supported=True,
        problem=text,
        domain=domain,
        excluded_points=tuple(base1_roots),
        cases=tuple(cases),
        solution=final.normalized(),
        constant_base=constant_base,
    )


def _arg_str(arg_text: str) -> str:
    """Display form of the argument; parentheses added when it has spaces+signs."""
    t = arg_text.strip()
    if "+" in t or ("-" in t and not t.startswith("-")):
        return f"({t})"
    return t


# ── polynomial / rational inequality solving ───────────────────────────────


def _solve_poly_problem(text: str) -> SolveResult:
    split = _split_relation(text)
    if split is None:
        return SolveResult(False, "no comparison relation (<, <=, >, >=) found")
    left, rel, right = split
    if not left.strip() or not right.strip():
        return SolveResult(False, "one side of the comparison is empty")
    lhs = _parse_expr(left)
    if isinstance(lhs, str):
        return SolveResult(False, lhs)
    rhs = _parse_expr(right)
    if isinstance(rhs, str):
        return SolveResult(False, rhs)
    try:
        num, den = sympy.fraction(sympy.together(lhs - rhs))
    except Exception as e:  # sympy can raise on exotic expressions
        logger.debug("fraction/together failed for %s: %s", text, e)
        return SolveResult(False, "the expression could not be brought to a rational form")
    solved = _solve_rel(num, den, rel)
    if isinstance(solved, str):
        return SolveResult(False, solved)
    den_poly = _poly(den)
    if den_poly is None:
        return SolveResult(False, "the expression is not a rational function of x")
    den_roots = _rational_roots(den_poly)
    if den_roots is None:
        return SolveResult(False, "a critical point is irrational, so the solution cannot be verified exactly")
    whole = IntervalSet.of(Interval(None, None, False, False))
    domain = whole.without_points(den_roots) if den_roots else whole
    return SolveResult(
        supported=True,
        problem=text,
        domain=domain,
        excluded_points=tuple(den_roots),
        cases=(),
        solution=solved.normalized(),
    )


def solve_problem(problem: str) -> SolveResult:
    """Re-solve ONE interval inequality problem exactly. Pure."""
    text = _prepare_problem(problem)
    if not text:
        return SolveResult(False, "the problem statement is empty")
    if len(text) > 200:
        return SolveResult(False, "the problem statement is too long to verify")
    if re.search(r"\blog", text, flags=re.IGNORECASE):
        return _solve_log(text)
    return _solve_poly_problem(text)


# ── claimed-spec verification ──────────────────────────────────────────────


def _track_set(track: VisualNumberLineTrack) -> IntervalSet | str:
    intervals: list[Interval] = []
    for vi in track.intervals:
        lo = parse_bound(vi.lo, minus_side=True)
        if isinstance(lo, str):
            return f"{track.label or 'track'}: {lo}"
        hi = parse_bound(vi.hi, minus_side=False)
        if isinstance(hi, str):
            return f"{track.label or 'track'}: {hi}"
        iv = Interval(lo, hi, vi.lo_closed, vi.hi_closed).normalized()
        if iv.is_empty():
            return f"{track.label or 'track'}: degenerate interval ({vi.lo}, {vi.hi})"
        intervals.append(iv)
    result = IntervalSet(tuple(intervals)).normalized()
    points: list[Fraction] = []
    for p in track.points:
        parsed = parse_bound(p, minus_side=True)
        if isinstance(parsed, str) or parsed is None:
            return f"{track.label or 'track'}: unparseable excluded value {p!r}"
        points.append(parsed)
    if points:
        result = result.without_points(points)
    return result


_COND_BETWEEN_RE = re.compile(r"^0\s*<\s*(.+?)\s*<\s*1$")
_COND_GT_RE = re.compile(r"^(.+?)\s*>\s*1$")
_COND_LT_RE = re.compile(r"^(.+?)\s*<\s*1$")


def _parse_condition(text: str) -> tuple[str, sympy.Expr] | None:
    t = _prep(text).strip()
    m = _COND_BETWEEN_RE.match(t)
    if m:
        expr = _parse_expr(m.group(1))
        if isinstance(expr, str):
            return None
        return "between", expr
    m = _COND_GT_RE.match(t)
    if m:
        expr = _parse_expr(m.group(1))
        if isinstance(expr, str):
            return None
        return "gt", expr
    m = _COND_LT_RE.match(t)
    if m:
        expr = _parse_expr(m.group(1))
        if isinstance(expr, str):
            return None
        return "lt", expr
    return None


def _same_expr(a: sympy.Expr, b: sympy.Expr) -> bool:
    try:
        return sympy.simplify(a - b) == 0
    except Exception as e:  # sympy simplify can raise on exotic input
        logger.debug("simplify failed: %s", e)
        return False


def _solve_claimed_transformed(text: str, regime: IntervalSet) -> IntervalSet | str:
    split = _split_relation(text)
    if split is None:
        return f"no comparison relation found in {text!r}"
    left, rel, right = split
    lhs = _parse_expr(left)
    if isinstance(lhs, str):
        return lhs
    rhs = _parse_expr(right)
    if isinstance(rhs, str):
        return rhs
    try:
        num, den = sympy.fraction(sympy.together(lhs - rhs))
    except Exception as e:  # sympy can raise on exotic expressions
        logger.debug("fraction/together failed for %r: %s", text, e)
        return f"could not parse the transformed inequality {text!r}"
    solved = _solve_rel(num, den, rel)
    if isinstance(solved, str):
        return solved
    return solved.intersect(regime)


def verify_number_line(nl: VisualNumberLine) -> list[str]:
    """Verify a number_line scene end to end. Returns [] when fully verified.

    Every returned string is a precise, human-readable refusal/repair reason.
    """
    problem = nl.problem.strip()
    if not problem:
        return ["number line scene is missing the problem statement"]
    result = solve_problem(problem)
    if not result.supported:
        return [
            f"cannot verify the mathematics of {problem!r} ({result.reason}); "
            "worked math is never displayed unverified"
        ]
    assert result.solution is not None  # guaranteed when supported
    errors: list[str] = []

    claimed = _track_set(nl.solution)
    if isinstance(claimed, str):
        errors.append(f"the final answer could not be parsed as intervals: {claimed}")
    elif not claimed.equals(result.solution):
        errors.append(
            f"the final answer {format_interval_set(claimed)} is mathematically wrong "
            f"for {problem!r}; the correct answer is {format_interval_set(result.solution)}"
        )

    if nl.domain is not None:
        claimed_domain = _track_set(nl.domain)
        if isinstance(claimed_domain, str):
            errors.append(f"the domain track could not be parsed: {claimed_domain}")
        elif result.domain is not None and not claimed_domain.equals(result.domain):
            errors.append(
                f"the domain {format_interval_set(claimed_domain)} is mathematically wrong; "
                f"the correct domain is {format_interval_set(result.domain)}"
            )

    if result.constant_base:
        if nl.cases:
            errors.append(
                "the logarithm base is a constant, so there is no case split to show; "
                "remove the cases"
            )
        return errors

    computed_cases = [c for c in result.cases]
    if nl.cases:
        if len(nl.cases) != len(computed_cases):
            errors.append(
                f"expected {len(computed_cases)} case branch(es) for this problem, "
                f"found {len(nl.cases)}"
            )
        remaining = list(computed_cases)
        for mc in nl.cases:
            parsed = _parse_interval_set_text(mc.solution)
            if isinstance(parsed, str):
                errors.append(f"case solution {mc.solution!r} could not be parsed: {parsed}")
                continue
            match = None
            for candidate in remaining:
                if parsed.equals(candidate.solution):
                    match = candidate
                    break
            if match is None:
                valid = "; ".join(
                    f"{format_interval_set(c.solution)} for {c.condition or 'the base regime'}"
                    for c in computed_cases
                )
                errors.append(
                    f"case solution {mc.solution!r} does not match the computed case "
                    f"solutions ({valid})"
                )
                continue
            remaining.remove(match)
            cond = _parse_condition(mc.condition)
            if cond is None:
                errors.append(f"case condition {mc.condition!r} could not be parsed")
            else:
                kind, expr = cond
                base_expr = _base_expr(result)
                if base_expr is not None and not _same_expr(expr, base_expr):
                    errors.append(
                        f"case condition {mc.condition!r} does not describe the "
                        "logarithm base"
                    )
                elif match.condition.startswith("0 <") and kind == "gt":
                    errors.append(
                        f"case condition {mc.condition!r} claims base > 1, but this "
                        "branch is the 0 < base < 1 regime"
                    )
                elif match.condition.endswith("> 1") and kind in ("lt", "between"):
                    errors.append(
                        f"case condition {mc.condition!r} claims base < 1, but this "
                        "branch is the base > 1 regime"
                    )
            if mc.transformed.strip():
                checked = _solve_claimed_transformed(mc.transformed, match.regime)
                if isinstance(checked, str):
                    errors.append(f"case step {mc.transformed!r}: {checked}")
                elif not checked.equals(match.solution):
                    errors.append(
                        f"case step {mc.transformed!r} yields {format_interval_set(checked)} "
                        f"on its branch, but the correct case solution is "
                        f"{format_interval_set(match.solution)}"
                    )
        if remaining:
            missing = "; ".join(
                f"{format_interval_set(c.solution)} for {c.condition}" for c in remaining
            )
            errors.append(f"missing case branch(es): {missing}")
    return errors


def _base_expr(result: SolveResult) -> sympy.Expr | None:
    """Re-extract the log base expression from the stored problem text."""
    found = _find_log(result.problem)
    if found is None:
        return None
    expr = _parse_expr(found[0])
    return None if isinstance(expr, str) else expr


def _parse_interval_set_text(text: str) -> IntervalSet | str:
    from app.utils.math_intervals import parse_interval_set

    return parse_interval_set(text)
