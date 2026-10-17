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
set algebra uses math_intervals (Fractions). Expression text is parsed
through a bounded, AST-validated mathematical grammar — never Python eval
and never sympy.parse_expr on untrusted input. Decimal literals are committed
exactly from their source text (never through a Python float), and every
divisor (denominator, negative power) is tracked as a domain restriction so
simplification can never erase the original domain.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from fractions import Fraction

import sympy

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
_MAX_POW = 10
_MAX_EXPR_NODES = 120
_MAX_EXPR_DEPTH = 12

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
    # Unicode subscripts (model output style: logx₊₃(...))
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u208a": "+",
    "\u208b": "-",
}

_REVERSE_REL = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}

_LEAD_RE = re.compile(r"^\s*(?:solve|simplify)\b[^0-9x(]*", re.IGNORECASE)

# "logx+3(...)" (no explicit base marker) -> "log_{x+3}(...)". The base text
# between "log" and the argument parenthesis is unambiguous, and the problem
# is verified exactly afterward, so this normalization is safe. Plain
# "log(...)" keeps its (unsupported) bare form — the class needs 1+ chars.
_LOG_SUBSCRIPT_RE = re.compile(r"\blog([^_({(]+?)\(", re.IGNORECASE)


# ── results ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MathCase:
    """One computed case branch of a log inequality solution."""

    condition: str
    regime: IntervalSet  # where this branch applies (inside the domain)
    transformed: str  # human-readable transformed inequality
    transformed_num: sympy.Expr  # numerator of (lhs - rhs) for the branch
    solution: IntervalSet
    transformed_rel: str = ""  # relation of the derived transformation


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
    text = _LOG_SUBSCRIPT_RE.sub(r"log_{\1}(", text)
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


# ── exact expression parsing (safe, no eval) ───────────────────────────────
#
# Expression text is source- and model-derived, so it must never reach Python
# eval or sympy.parse_expr (sympy's parse_expr evaluates the source string and
# is documented as unsafe for untrusted input). Instead:
#   1. tokenize against a strict mathematical grammar,
#   2. rewrite implicit multiplication ("2x", "x(x+1)", "(x+1)(x-2)") into
#      explicit Python syntax,
#   3. parse with ast.parse(mode="eval") — which only builds a syntax tree
#      and never executes anything,
#   4. construct sympy expressions from a closed allowlist of AST nodes:
#      numbers, the variable x, unary signs, +, -, *, /, and bounded integer
#      powers.
# Function calls, attribute access, imports, strings, and every other syntax
# are rejected outright. Size, node count, depth, and exponent magnitude are
# bounded to prevent computational abuse. Decimal literals are rebuilt from
# their ORIGINAL source text (ast.get_source_segment) so constants that need
# more precision than a float are committed exactly, not rounded. Division by
# zero is refused at build time, and every divisor is recorded as a domain
# restriction (a factor that must be != 0) so cancellation cannot erase the
# original domain.

_NUMBER_RE = re.compile(r"\d+\.\d*|\.\d+|\d+")

_EXPR_TOKEN_RE = re.compile(
    r"(?P<num>\d+\.\d*|\.\d+|\d+)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>\*\*|[+\-*/^()])"
    r"|(?P<ws>\s+)"
)


def _tokenize_expr(text: str) -> list[tuple[str, str]] | str:
    """Strict tokenizer. Returns tokens or a human-readable rejection reason."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _EXPR_TOKEN_RE.match(text, pos)
        if m is None:
            return f"unsupported character {text[pos]!r}"
        pos = m.end()
        if m.lastgroup == "ws":
            continue
        tokens.append((m.lastgroup or "", m.group()))
    return tokens


def _to_python_source(tokens: list[tuple[str, str]]) -> tuple[str, str]:
    """Join tokens into Python syntax, inserting explicit multiplication for
    implicit products. Returns (source, "") or ("", reason)."""
    parts: list[str] = []
    prev_kind: str | None = None
    prev_tok: str | None = None
    for kind, tok in tokens:
        starts_factor = kind in ("num", "name") or tok == "("
        ends_factor = prev_kind in ("num", "name") or prev_tok == ")"
        if prev_kind is not None and starts_factor and ends_factor:
            if prev_kind == "num" and kind == "num":
                return "", "adjacent numbers"
            if prev_kind == "name" and kind == "num":
                return "", "unsupported symbol sequence"
            parts.append("*")
        parts.append("**" if tok == "^" else tok)
        prev_kind, prev_tok = kind, tok
    return "".join(parts), ""


def _build_expr(
    node: ast.AST,
    depth: int,
    budget: list[int],
    source: str,
    restrictions: list[sympy.Expr],
) -> sympy.Expr | str:
    """Construct a sympy expression from an allowlisted AST node.

    `restrictions` collects every divisor (denominator / negative-power base)
    that must be nonzero for the original expression to be defined.
    """
    budget[0] -= 1
    if budget[0] < 0:
        return "expression is too complex to verify"
    if depth > _MAX_EXPR_DEPTH:
        return "expression is nested too deeply to verify"
    if isinstance(node, ast.Expression):
        return _build_expr(node.body, depth, budget, source, restrictions)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "unsupported literal"
        if isinstance(value, int):
            return sympy.Integer(value)
        segment = ast.get_source_segment(source, node)
        if segment is not None and _NUMBER_RE.fullmatch(segment.strip()):
            frac = Fraction(segment.strip())
        else:
            frac = Fraction(str(value))
        return sympy.Rational(frac.numerator, frac.denominator)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return _X
        return f"unsupported symbols ({node.id})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _build_expr(node.operand, depth + 1, budget, source, restrictions)
        if isinstance(operand, str):
            return operand
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _build_expr(node.left, depth + 1, budget, source, restrictions)
        if isinstance(left, str):
            return left
        right = _build_expr(node.right, depth + 1, budget, source, restrictions)
        if isinstance(right, str):
            return right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                return "division by zero"
            restrictions.append(right)
            return left / right
        if isinstance(node.op, ast.Pow):
            if not right.is_rational or right.q != 1:
                return "exponents must be whole numbers"
            exponent = int(right)
            if abs(exponent) > _MAX_POW:
                return f"exponents are limited to +/-{_MAX_POW}"
            if exponent < 0:
                if left == 0:
                    return "division by zero"
                restrictions.append(left)
            return left**exponent
        return "unsupported operator"
    return "unsupported syntax"


def _parse_expr_with_restrictions(
    text: str,
) -> tuple[sympy.Expr, tuple[sympy.Expr, ...]] | str:
    """Parse expression text into (expr, domain restrictions).

    Restrictions are the divisors the expression requires to be nonzero.
    Returns an error string (never raises) for anything outside the grammar.
    """
    t = _prep(text).strip()
    if not t:
        return "expression is empty"
    if len(t) > _MAX_EXPR_LEN:
        return f"expression is too long: {t[:40]!r}..."
    tokens = _tokenize_expr(t)
    if isinstance(tokens, str):
        return f"could not parse expression {t!r}: {tokens}"
    if not tokens:
        return "expression is empty"
    source, err = _to_python_source(tokens)
    if err:
        return f"could not parse expression {t!r}: {err}"
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return f"could not parse expression {t!r}"
    restrictions: list[sympy.Expr] = []
    expr = _build_expr(tree, 0, [_MAX_EXPR_NODES], source, restrictions)
    if isinstance(expr, str):
        return f"could not parse expression {t!r}: {expr}"
    if not expr.free_symbols <= {_X}:  # defense in depth; unreachable
        names = ", ".join(sorted(str(s) for s in expr.free_symbols))
        return f"expression uses unsupported symbols ({names}): {t!r}"
    return expr, tuple(restrictions)


def _parse_expr(text: str) -> sympy.Expr | str:
    parsed = _parse_expr_with_restrictions(text)
    if isinstance(parsed, str):
        return parsed
    return parsed[0]


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
    if poly.is_zero:
        return []  # the zero polynomial vanishes nowhere *exclusively*
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


def _restriction_zeros(restrictions: tuple[sympy.Expr, ...]) -> list[Fraction] | str:
    """Exact zero points of every tracked divisor, or a refusal reason.

    A divisor that is not a polynomial (or whose zero is irrational) cannot be
    excluded exactly, so the problem is refused instead of guessed.
    """
    roots: set[Fraction] = set()
    for divisor in restrictions:
        poly = _poly(divisor)
        if poly is None:
            return "a divisor does not reduce to a polynomial in x, so the domain cannot be verified exactly"
        found = _rational_roots(poly)
        if found is None:
            return "a divisor vanishes at an irrational point, so the domain cannot be verified exactly"
        roots.update(found)
    return sorted(roots)


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
    if den_poly.is_zero:
        return "division by zero"
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
        if want_neg:
            satisfied = sign < 0 or (sign == 0 and not strict)
        else:
            satisfied = sign > 0 or (sign == 0 and not strict)
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
    constant_base = not base_expr.free_symbols
    # A constant base that is <= 0 or == 1 is not a valid logarithm: there is
    # no domain to solve over. Refuse explicitly instead of returning an
    # empty-domain "solution" that could be confused with a real answer.
    if constant_base and (base_expr <= 0 or base_expr == 1):
        return SolveResult(
            False,
            "the logarithm base must be a positive number different from 1",
        )
    if not constant_base and abs(int(const)) > _MAX_POW:
        return SolveResult(
            False,
            f"the comparison constant is too large for a variable-base logarithm (limit {_MAX_POW})",
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
    if domain.is_empty():
        return SolveResult(
            False,
            "the logarithm is never defined for any real x in this problem",
        )
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
                transformed_rel=rel_t,
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
    parsed_left = _parse_expr_with_restrictions(left)
    if isinstance(parsed_left, str):
        return SolveResult(False, parsed_left)
    parsed_right = _parse_expr_with_restrictions(right)
    if isinstance(parsed_right, str):
        return SolveResult(False, parsed_right)
    lhs, lhs_restrictions = parsed_left
    rhs, rhs_restrictions = parsed_right
    restrictions = lhs_restrictions + rhs_restrictions
    # The ORIGINAL divisors are the domain authority. `together` may cancel
    # them (e.g. ((x-1)(x+1))/(x-1) simplifies), so their zeros are computed
    # from the tracked restrictions, never from the simplified denominator.
    restriction_roots = _restriction_zeros(restrictions)
    if isinstance(restriction_roots, str):
        return SolveResult(False, restriction_roots)
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
    excluded = sorted(set(restriction_roots) | set(den_roots))
    whole = IntervalSet.of(Interval(None, None, False, False))
    domain = whole.without_points(excluded) if excluded else whole
    return SolveResult(
        supported=True,
        problem=text,
        domain=domain,
        excluded_points=tuple(excluded),
        cases=(),
        solution=solved.intersect(domain).normalized(),
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


def claimed_track_set(track: VisualNumberLineTrack) -> IntervalSet | str:
    """The interval set a rendered track CLAIMS (intervals minus marked points)."""
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


def _parse_claim_relation(
    text: str,
) -> tuple[sympy.Expr, str, sympy.Expr, tuple[sympy.Expr, ...]] | str:
    """Parse a model-claimed inequality into (lhs, rel, rhs, restrictions)."""
    split = _split_relation(_prep(text))
    if split is None:
        return f"no comparison relation found in {text!r}"
    left, rel, right = split
    parsed_left = _parse_expr_with_restrictions(left)
    if isinstance(parsed_left, str):
        return parsed_left
    parsed_right = _parse_expr_with_restrictions(right)
    if isinstance(parsed_right, str):
        return parsed_right
    lhs, lhs_restrictions = parsed_left
    rhs, rhs_restrictions = parsed_right
    return lhs, rel, rhs, lhs_restrictions + rhs_restrictions


def _solve_claim_over_regime(
    parsed: tuple[sympy.Expr, str, sympy.Expr, tuple[sympy.Expr, ...]],
    regime: IntervalSet,
) -> IntervalSet | str:
    lhs, rel, rhs, restrictions = parsed
    zeros: list[Fraction] = []
    if restrictions:
        computed_zeros = _restriction_zeros(restrictions)
        if isinstance(computed_zeros, str):
            return computed_zeros
        zeros = computed_zeros
    try:
        num, den = sympy.fraction(sympy.together(lhs - rhs))
    except Exception as e:  # sympy can raise on exotic expressions
        logger.debug("fraction/together failed for %r: %s", lhs - rhs, e)
        return "could not parse the transformed inequality"
    solved = _solve_rel(num, den, rel)
    if isinstance(solved, str):
        return solved
    if zeros:
        solved = solved.without_points(zeros)
    return solved.intersect(regime)


def _transformed_derivation_mismatch(
    parsed: tuple[sympy.Expr, str, sympy.Expr, tuple[sympy.Expr, ...]],
    match: MathCase,
) -> str | None:
    """Check the claimed transformation against the DERIVED one algebraically.

    The claimed inequality must be the derived transformation up to a nonzero
    rational multiple (any additive rearrangement passes; multiplying by a
    negative constant must flip the direction). A step whose solution set
    merely coincides with the correct one after intersecting the branch is
    NOT accepted. Returns a mismatch reason or None.
    """
    lhs, rel, rhs, _ = parsed
    try:
        claimed_diff = sympy.expand(lhs - rhs)
        derived_diff = sympy.expand(match.transformed_num)
    except Exception as e:  # sympy can raise on exotic expressions
        logger.debug("expand failed for transformed comparison: %s", e)
        return "could not be compared with the derived transformation"
    if derived_diff == 0:
        if claimed_diff != 0:
            return f"is not equivalent to the derived transformation {match.transformed!r}"
        return None
    if claimed_diff == 0:
        return f"is not equivalent to the derived transformation {match.transformed!r}"
    try:
        ratio = sympy.cancel(claimed_diff / derived_diff)
    except Exception as e:
        logger.debug("cancel failed for transformed comparison: %s", e)
        return "could not be compared with the derived transformation"
    if ratio.free_symbols or not ratio.is_rational:
        return (
            f"is not an algebraic rearrangement of the derived transformation "
            f"{match.transformed!r} (bring every term to one side of the same "
            "inequality instead of rewriting it)"
        )
    expected_rel = match.transformed_rel
    if ratio < 0:
        expected_rel = _REVERSE_REL[expected_rel]
    if rel != expected_rel:
        return (
            f"uses the wrong inequality direction for the derived transformation "
            f"{match.transformed!r}"
        )
    return None


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

    claimed = claimed_track_set(nl.solution)
    if isinstance(claimed, str):
        errors.append(f"the final answer could not be parsed as intervals: {claimed}")
    elif not claimed.equals(result.solution):
        errors.append(
            f"the final answer {format_interval_set(claimed)} is mathematically wrong "
            f"for {problem!r}; the correct answer is {format_interval_set(result.solution)}"
        )

    # A domain track is REQUIRED whenever the problem restricts the domain
    # (log argument/base, denominator zeros). Only problems whose domain is
    # the whole real line may omit the track.
    whole_line = IntervalSet.of(Interval(None, None, False, False))
    domain_required = result.domain is not None and not result.domain.equals(whole_line)
    if nl.domain is None:
        if domain_required:
            errors.append(
                f"the domain track is required for {problem!r}: the domain is "
                f"{format_interval_set(result.domain)}, and it is part of a "
                "verified worked solution"
            )
    else:
        claimed_domain = claimed_track_set(nl.domain)
        if isinstance(claimed_domain, str):
            errors.append(f"the domain track could not be parsed: {claimed_domain}")
        elif result.domain is not None and not claimed_domain.equals(result.domain):
            errors.append(
                f"the domain {format_interval_set(claimed_domain)} is mathematically wrong; "
                f"the correct domain is {format_interval_set(result.domain)}"
            )

    # Every excluded value must be visually apparent: either marked as a point
    # on a track or shown as a domain interval endpoint. Markers for values
    # that are NOT excluded are mislabels and are refused too.
    marked: set[Fraction] = set()
    for track in (nl.domain, nl.solution):
        if track is None:
            continue
        for raw_point in track.points:
            parsed_point = parse_bound(raw_point, minus_side=True)
            if not isinstance(parsed_point, str) and parsed_point is not None:
                marked.add(parsed_point)
    excluded = set(result.excluded_points)
    extra_marks = sorted(marked - excluded)
    if extra_marks:
        values = ", ".join(f"x = {v}" for v in extra_marks)
        errors.append(
            f"the scene marks {values} as excluded, but "
            + ("that value is" if len(extra_marks) == 1 else "those values are")
            + " in the domain of the problem"
        )
    hidden: list[Fraction] = []
    for point in sorted(excluded - marked):
        visible = False
        if nl.domain is not None:
            for vi in nl.domain.intervals:
                lo_b = parse_bound(vi.lo, minus_side=True)
                hi_b = parse_bound(vi.hi, minus_side=False)
                if (not isinstance(lo_b, str) and lo_b == point) or (
                    not isinstance(hi_b, str) and hi_b == point
                ):
                    visible = True
                    break
        if not visible:
            hidden.append(point)
    if hidden:
        values = ", ".join(f"x = {v}" for v in hidden)
        errors.append(
            f"the excluded value(s) {values} must be shown on the number line "
            "(as a marked point or an open domain endpoint)"
        )

    if result.constant_base:
        if nl.cases:
            errors.append(
                "the logarithm base is a constant, so there is no case split to show; "
                "remove the cases"
            )
        return errors

    # Variable-base log: EVERY computed base-regime branch must be shown. An
    # empty case list is a missing explanation, never a pass.
    computed_cases = [c for c in result.cases]
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
            elif match.condition.startswith("0 <"):
                if kind == "gt":
                    errors.append(
                        f"case condition {mc.condition!r} claims base > 1, but this "
                        "branch is the 0 < base < 1 regime"
                    )
                elif kind == "lt":
                    errors.append(
                        f"case condition {mc.condition!r} is incomplete: it omits the "
                        "positivity requirement; this branch is the 0 < base < 1 regime"
                    )
            elif kind != "gt":
                errors.append(
                    f"case condition {mc.condition!r} claims base < 1, but this "
                    "branch is the base > 1 regime"
                )
        if mc.transformed.strip():
            parsed_claim = _parse_claim_relation(mc.transformed)
            if isinstance(parsed_claim, str):
                errors.append(f"case step {mc.transformed!r}: {parsed_claim}")
            else:
                checked = _solve_claim_over_regime(parsed_claim, match.regime)
                if isinstance(checked, str):
                    errors.append(f"case step {mc.transformed!r}: {checked}")
                elif not checked.equals(match.solution):
                    errors.append(
                        f"case step {mc.transformed!r} yields {format_interval_set(checked)} "
                        f"on its branch, but the correct case solution is "
                        f"{format_interval_set(match.solution)}"
                    )
                else:
                    mismatch = _transformed_derivation_mismatch(parsed_claim, match)
                    if mismatch is not None:
                        errors.append(f"case step {mc.transformed!r} {mismatch}")
        else:
            errors.append(
                f"case step for {match.condition or 'the base regime'} is missing; "
                "the transformed inequality is required for every branch"
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
