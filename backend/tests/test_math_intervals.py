"""Exact interval-set algebra (math_intervals): parsing, merging, set ops."""
from fractions import Fraction

from app.utils.math_intervals import (
    Interval,
    IntervalSet,
    format_interval_set,
    parse_interval,
    parse_interval_set,
)


def _set(text: str) -> IntervalSet:
    parsed = parse_interval_set(text)
    assert not isinstance(parsed, str), parsed
    return parsed


def test_parse_unicode_union_and_minus():
    s = _set("(\u22123, \u22122) \u222a (\u22121, 0) \u222a (1, 3)")
    assert format_interval_set(s) == "(-3, -2) ∪ (-1, 0) ∪ (1, 3)"


def test_parse_infinity_endpoints():
    s = _set("(-inf, 3) \u222a (5, inf)")
    assert format_interval_set(s) == "(-inf, 3) ∪ (5, inf)"


def test_overlapping_infinity_intervals_merge():
    s = _set("(-inf, 3) \u222a (1, inf)")
    assert format_interval_set(s) == "(-inf, inf)"


def test_parse_fractions_and_decimals_are_exact():
    a = _set("[1/2, 1.5]")
    b = _set("[0.5, 3/2]")
    assert a.equals(b)
    assert format_interval_set(a) == "[1/2, 3/2]"


def test_merge_touching_closed_intervals():
    s = _set("[1, 2) \u222a [2, 3]")
    assert format_interval_set(s) == "[1, 3]"


def test_open_touching_intervals_do_not_merge():
    s = _set("(-3, -2) \u222a (-2, 0)")
    assert format_interval_set(s) == "(-3, -2) ∪ (-2, 0)"


def test_whole_line_and_point_interval():
    assert format_interval_set(_set("(-inf, inf)")) == "(-inf, inf)"
    assert format_interval_set(_set("[2, 2]")) == "[2, 2]"


def test_empty_markers():
    assert _set("∅").is_empty()
    assert _set("no solution").is_empty()
    assert format_interval_set(_set("∅")) == "∅"


def test_intersect():
    a = _set("(-3, 0) \u222a (2, 5)")
    b = _set("(-1, 3)")
    assert format_interval_set(a.intersect(b)) == "(-1, 0) ∪ (2, 3)"
    assert a.intersect(_set("(5, 9)")).is_empty()


def test_without_points_splits_intervals():
    a = _set("(-3, 0)")
    result = a.without_points([Fraction(-2)])
    assert format_interval_set(result) == "(-3, -2) ∪ (-2, 0)"
    untouched = a.without_points([Fraction(7)])
    assert format_interval_set(untouched) == "(-3, 0)"


def test_equality_is_exact():
    assert _set("(-3, -2) \u222a (-1, 0)").equals(_set("(-1, 0) \u222a (-3, -2)"))
    assert not _set("(-3, -2]").equals(_set("(-3, -2)"))


def test_degenerate_and_bad_inputs_return_errors():
    assert isinstance(parse_interval_set("(3, 3)"), str)
    assert isinstance(parse_interval_set(""), str)
    assert isinstance(parse_interval_set("abc"), str)
    assert isinstance(parse_interval("(1, -inf)"), str)
    assert isinstance(parse_interval("(inf, 3)"), str)


def test_interval_normalization_drops_bad_closures():
    iv = Interval(None, Fraction(3), True, True)
    assert iv.normalized().lo_closed is False


# ── boundary and edge semantics ────────────────────────────────────────────


def test_without_points_unbounded_closed_and_repeated():
    whole = _set("(-inf, inf)")
    assert format_interval_set(whole.without_points([Fraction(2)])) == "(-inf, 2) \u222a (2, inf)"
    closed = _set("[1, 2]")
    assert format_interval_set(closed.without_points([Fraction(1)])) == "(1, 2]"
    assert format_interval_set(closed.without_points([Fraction(2)])) == "[1, 2)"
    assert closed.without_points([Fraction(1), Fraction(1)]).equals(
        closed.without_points([Fraction(1)])
    )
    assert closed.without_points([]).equals(closed)
    assert format_interval_set(_set("[2, 2]").without_points([Fraction(2)])) == "\u2205"
    assert format_interval_set(_set("(-2, 2)").without_points([Fraction(0)])) == "(-2, 0) \u222a (0, 2)"
    # a point outside every interval leaves the set untouched
    assert _set("(-1, 1)").without_points([Fraction(7)]).equals(_set("(-1, 1)"))


def test_union_intersect_edge_semantics():
    # (0, 1] meets [1, 2) exactly at the single point 1 ...
    assert _set("(0, 1]").intersect(_set("[1, 2)")).equals(_set("[1, 1]"))
    # ... but (0, 1) excludes it, so the intersection is empty.
    assert _set("(0, 1)").intersect(_set("[1, 2)")).is_empty()
    assert _set("[0, 1]").union(_set("[1, 2]")).equals(_set("[0, 2]"))
    assert _set("(-inf, 0)").union(_set("(0, inf)")).equals(_set("(-inf, 0) \u222a (0, inf)"))
    assert _set("\u2205").union(_set("[1, 2]")).equals(_set("[1, 2]"))
    assert _set("[1, 2]").intersect(_set("\u2205")).is_empty()
    assert _set("(-inf, inf)").intersect(_set("[3, 4]")).equals(_set("[3, 4]"))


def test_malformed_union_separators_rejected():
    assert isinstance(parse_interval_set("(1, 2) \u222a"), str)
    assert isinstance(parse_interval_set("\u222a (1, 2)"), str)
    assert isinstance(parse_interval_set("(1, 2) \u222a \u222a (3, 4)"), str)


def test_decimal_and_fraction_bounds_are_equal_exactly():
    assert _set("[0.1, 0.3]").equals(_set("[1/10, 3/10]"))
    long_decimal = "0.1000000000000000000000000000000000001"
    long_fraction = "1000000000000000000000000000000000001/10000000000000000000000000000000000000"
    assert _set(f"[{long_decimal}, 1]").equals(_set(f"[{long_fraction}, 1]"))
    assert not _set("[0.1, 0.3]").equals(_set("[0.1000000000000000055511151231257827021181583404541015625, 0.3]"))
