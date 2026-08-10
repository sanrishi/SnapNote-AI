"""Normalize free-form math strings from Gemini into a canonical internal form.

Purpose: "sqrt5", "sqrt(5)", "√5" must all become the same canonical string
("sqrt(5)"), and "2pi"/"2π"/"2*pi" become "2*pi". The renderer and validator
only ever work with canonical strings plus their numeric value — never with
raw Gemini output. Pure, deterministic, no external deps.
"""

from __future__ import annotations

import ast
import math
import re

_PI = math.pi
_EPS = 1e-9

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_UNICODE_TOKENS = {
    "π": "pi",
    "√": "sqrt",
    "×": "*",
    "·": "*",
    "−": "-",
    "–": "-",
    "∕": "/",
}

_SQRT_IMPLICIT = re.compile(r"sqrt(\d+(?:\.\d+)?)")
_SQRT_BRACE = re.compile(r"sqrt\{([^}]*)\}")
_DIGIT_PI = re.compile(r"(\d)pi")
_CLOSE_PI = re.compile(r"\)pi")
_DIGIT_SQRT = re.compile(r"(\d)sqrt")
_CLOSE_SQRT = re.compile(r"\)sqrt")
_DIGIT_SQRT_OPEN = re.compile(r"(\d)sqrt\(")
_DIGIT_OPEN = re.compile(r"(\d)\(")
_CLOSE_DIGIT = re.compile(r"\)(\d)")
_CLOSE_OPEN = re.compile(r"\)\(")


def normalize_math(expr: str) -> str | None:
    """Return the canonical form of a math expression, or None if unparseable.

    Canonical examples: "1", "sqrt(5)", "2*pi", "pi/2", "sqrt(2)/2".
    """
    if not expr:
        return None
    s = expr.strip().lower().replace(" ", "").replace("\u200b", "").replace("\u2009", "")
    s = s.translate(_SUPERSCRIPT)
    for src, dst in _UNICODE_TOKENS.items():
        s = s.replace(src, dst)
    s = s.replace("**", "^")

    s = _SQRT_BRACE.sub(r"sqrt(\1)", s)
    s = _SQRT_IMPLICIT.sub(r"sqrt(\1)", s)
    s = _DIGIT_SQRT_OPEN.sub(r"\1*sqrt(", s)
    s = _DIGIT_PI.sub(r"\1*pi", s)
    s = _CLOSE_PI.sub(r")*pi", s)
    s = _DIGIT_SQRT.sub(r"\1*sqrt", s)
    s = _CLOSE_SQRT.sub(r")*sqrt", s)
    s = _DIGIT_OPEN.sub(r"\1*(", s)
    s = _CLOSE_DIGIT.sub(r")*\1", s)
    s = _CLOSE_OPEN.sub(r")*(", s)

    if not s or not re.fullmatch(r"[0-9a-zA-Z+*/\-^().,]+", s):
        return None
    if math_value(s) is None:
        return None
    return s


def math_value(expr: str) -> float | None:
    """Safely evaluate a math string (no eval) into a float, or None on failure."""
    if not expr:
        return None
    py_expr = expr.replace("^", "**")
    try:
        tree = ast.parse(py_expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float | None:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return _PI
            if node.id == "e":
                return math.e
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _eval(node.operand)
            if val is None:
                return None
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if abs(right) < _EPS:
                    return None
                return left / right
            if isinstance(node.op, ast.Pow):
                if left < 0 and abs(right - round(right)) > _EPS:
                    return None
                return left ** right
            return None
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "sqrt"
                and len(node.args) == 1
                and not node.keywords
            ):
                val = _eval(node.args[0])
                if val is None or val < 0:
                    return None
                return math.sqrt(val)
            return None
        return None

    try:
        return _eval(tree)
    except Exception:
        return None


def display_math(expr: str) -> str:
    """Render a canonical math string for display (√5, 2π, π/2)."""
    if not expr:
        return ""
    out = expr
    out = re.sub(r"sqrt\(([^()]*)\)", r"√\1", out)
    out = out.replace("pi", "π")
    out = out.replace("*", "")
    return out
