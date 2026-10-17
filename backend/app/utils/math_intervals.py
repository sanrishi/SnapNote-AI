"""Exact interval-set algebra for worked mathematics (number_line family).

Pure, dependency-free (fractions only). Intervals have Fraction endpoints;
an unbounded side is represented as None. Used by the math verifier
(math_verify) to compare the spec's claimed solution sets against the
independently computed ones, and by the number-line renderer to lay out
endpoints exactly.

This module owns ONLY set algebra and parsing/formatting of the canonical
interval-set text form, e.g. "(-3, -2) ∪ (-1, 0) ∪ (1, 3)". It never solves
anything; solving lives in math_verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Interval:
    """One real interval. lo/hi: Fraction, or None for ±infinity."""

    lo: Fraction | None  # None == -infinity
    hi: Fraction | None  # None == +infinity
    lo_closed: bool = False
    hi_closed: bool = False

    def normalized(self) -> Interval:
        """Drop closures on unbounded sides; collapse impossible spans."""
        lo = self.lo
        hi = self.hi
        lo_closed = self.lo_closed if lo is not None else False
        hi_closed = self.hi_closed if hi is not None else False
        if lo is not None and hi is not None and lo > hi:
            return EMPTY_INTERVAL
        if lo is not None and hi is not None and lo == hi and not (lo_closed and hi_closed):
            return EMPTY_INTERVAL
        return Interval(lo, hi, lo_closed, hi_closed)

    def is_empty(self) -> bool:
        return self.normalized() is EMPTY_INTERVAL

    def is_point(self) -> bool:
        n = self.normalized()
        return n.lo is not None and n.lo == n.hi


EMPTY_INTERVAL = Interval(Fraction(0), Fraction(0), False, False)


def format_bound(value: Fraction | None, *, minus_side: bool) -> str:
    if value is None:
        return "-inf" if minus_side else "inf"
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_interval(iv: Interval) -> str:
    n = iv.normalized()
    left = "[" if n.lo_closed else "("
    right = "]" if n.hi_closed else ")"
    return (
        f"{left}{format_bound(n.lo, minus_side=True)}, "
        f"{format_bound(n.hi, minus_side=False)}{right}"
    )


def format_interval_set(s: IntervalSet) -> str:
    if not s.normalized().intervals:
        return "∅"
    return " ∪ ".join(format_interval(iv) for iv in s.normalized().intervals)


@dataclass(frozen=True)
class IntervalSet:
    """A finite union of disjoint, sorted, normalized intervals."""

    intervals: tuple[Interval, ...] = ()

    @staticmethod
    def of(*intervals: Interval) -> IntervalSet:
        return IntervalSet(tuple(intervals)).normalized()

    def normalized(self) -> IntervalSet:
        cleaned = [iv.normalized() for iv in self.intervals]
        cleaned = [iv for iv in cleaned if not iv.is_empty()]
        cleaned.sort(key=lambda iv: (iv.lo is not None, iv.lo if iv.lo is not None else 0))
        merged: list[Interval] = []
        for iv in cleaned:
            if not merged:
                merged.append(iv)
                continue
            prev = merged[-1]
            if _touches_or_overlaps(prev, iv):
                merged[-1] = _merge(prev, iv)
            else:
                merged.append(iv)
        return IntervalSet(tuple(merged))

    def is_empty(self) -> bool:
        return not self.normalized().intervals

    def equals(self, other: IntervalSet) -> bool:
        return self.normalized().intervals == other.normalized().intervals

    def intersect(self, other: IntervalSet) -> IntervalSet:
        a = self.normalized().intervals
        b = other.normalized().intervals
        out: list[Interval] = []
        for x in a:
            for y in b:
                lo = max_bound(x.lo, y.lo)
                hi = min_bound(x.hi, y.hi)
                if lo is not None and hi is not None and lo > hi:
                    continue
                if lo == x.lo == y.lo:
                    lo_closed = x.lo_closed and y.lo_closed
                elif lo == x.lo:
                    lo_closed = x.lo_closed
                else:
                    lo_closed = y.lo_closed
                if lo is None:
                    lo_closed = False
                if hi == x.hi == y.hi:
                    hi_closed = x.hi_closed and y.hi_closed
                elif hi == x.hi:
                    hi_closed = x.hi_closed
                else:
                    hi_closed = y.hi_closed
                if hi is None:
                    hi_closed = False
                iv = Interval(lo, hi, lo_closed, hi_closed).normalized()
                if not iv.is_empty():
                    out.append(iv)
        return IntervalSet(tuple(out)).normalized()

    def union(self, other: IntervalSet) -> IntervalSet:
        return IntervalSet(self.normalized().intervals + other.normalized().intervals).normalized()

    def without_points(self, points: list[Fraction]) -> IntervalSet:
        """Remove isolated points, splitting intervals around them."""
        result = self.normalized()
        for p in points:
            new: list[Interval] = []
            for iv in result.intervals:
                lo, hi = iv.lo, iv.hi
                if lo is not None and p < lo:
                    new.append(iv)
                    continue
                if hi is not None and p > hi:
                    new.append(iv)
                    continue
                if lo is not None and p == lo:
                    candidate = Interval(lo, hi, False, iv.hi_closed).normalized()
                    if not candidate.is_empty():
                        new.append(candidate)
                    continue
                if hi is not None and p == hi:
                    candidate = Interval(lo, hi, iv.lo_closed, False).normalized()
                    if not candidate.is_empty():
                        new.append(candidate)
                    continue
                left = Interval(lo, p, iv.lo_closed, False).normalized()
                right = Interval(p, hi, False, iv.hi_closed).normalized()
                if not left.is_empty():
                    new.append(left)
                if not right.is_empty():
                    new.append(right)
            result = IntervalSet(tuple(new)).normalized()
        return result


def max_bound(a: Fraction | None, b: Fraction | None) -> Fraction | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if a >= b else b


def min_bound(a: Fraction | None, b: Fraction | None) -> Fraction | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def _touches_or_overlaps(a: Interval, b: Interval) -> bool:
    if a.hi is None or b.lo is None:
        return True
    if a.hi > b.lo:
        return True
    if a.hi == b.lo:
        return a.hi_closed or b.lo_closed
    return False


def _merge(a: Interval, b: Interval) -> Interval:
    """Merge two intervals known to touch/overlap, given a sorts before b."""
    if a.lo is None:
        lo, lo_closed = None, False
    elif a.lo == b.lo:
        lo, lo_closed = a.lo, a.lo_closed or b.lo_closed
    else:
        lo, lo_closed = a.lo, a.lo_closed
    if a.hi is None or b.hi is None:
        hi, hi_closed = None, False
    elif a.hi == b.hi:
        hi, hi_closed = a.hi, a.hi_closed or b.hi_closed
    elif a.hi > b.hi:
        hi, hi_closed = a.hi, a.hi_closed
    else:
        hi, hi_closed = b.hi, b.hi_closed
    return Interval(lo, hi, lo_closed, hi_closed)


# ── parsing the canonical text form ────────────────────────────────────────

_UNICODE_FIXUPS = {
    "\u2212": "-",  # minus sign
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u221e": "inf",
    "\u222a": "U",  # union
    "\u22c3": "U",  # n-ary union
}

_EMPTY_MARKERS = {"∅", "empty", "none", "no solution", "no solutions"}


def _clean(text: str) -> str:
    out = text
    for bad, good in _UNICODE_FIXUPS.items():
        out = out.replace(bad, good)
    return out


def parse_bound(text: str, *, minus_side: bool) -> Fraction | None | str:
    """Parse one endpoint. Returns Fraction, None (infinity), or an error string."""
    t = text.strip()
    if t.startswith("+"):
        t = t[1:].strip()
    low = t.lower()
    if low in ("inf", "infinity"):
        if minus_side:
            return "positive infinity cannot be a lower bound"
        return None
    if low in ("-inf", "-infinity"):
        if not minus_side:
            return "negative infinity cannot be an upper bound"
        return None
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return f"unparseable endpoint: {text!r}"


_INTERVAL_RE = re.compile(r"^\s*([\[\(])\s*([^,]+?)\s*,\s*([^,]+?)\s*([\]\)])\s*$")


def parse_interval(text: str) -> Interval | str:
    """Parse one interval like "(-3, -2)" or "[0, 1/2]". Error string on failure."""
    m = _INTERVAL_RE.match(_clean(text))
    if not m:
        return f"unparseable interval: {text!r}"
    left, lo_txt, hi_txt, right = m.group(1), m.group(2), m.group(3), m.group(4)
    lo = parse_bound(lo_txt, minus_side=True)
    if isinstance(lo, str):
        return lo
    hi = parse_bound(hi_txt, minus_side=False)
    if isinstance(hi, str):
        return hi
    iv = Interval(lo, hi, left == "[", right == "]").normalized()
    if iv.is_empty():
        return f"degenerate interval: {text!r}"
    return iv


def parse_interval_set(text: str) -> IntervalSet | str:
    """Parse a union like "(-3, -2) ∪ (-1, 0) ∪ (1, 3)". Error string on failure.

    Accepts "U", "u", "∪" as union separators. Never treats a bare comma as a
    union separator (commas separate interval endpoints).
    """
    cleaned = _clean(text).strip()
    if not cleaned:
        return "empty interval-set text"
    if cleaned.lower() in _EMPTY_MARKERS:
        return IntervalSet()
    cleaned = re.sub(r"\bunited\b|\bor\b|\bunion\b", "U", cleaned, flags=re.IGNORECASE)
    raw_parts = re.split(r"[Uu]", cleaned)
    parts = [p.strip() for p in raw_parts]
    if any(not part for part in parts):
        return f"malformed union in interval set: {text!r}"
    intervals: list[Interval] = []
    for part in parts:
        parsed = parse_interval(part)
        if isinstance(parsed, str):
            return parsed
        intervals.append(parsed)
    return IntervalSet(tuple(intervals)).normalized()
