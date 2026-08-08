import pytest

from app.utils.svg_safe import sanitize_svg, svg_data_uri


def test_sanitize_keeps_clean_svg():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect x="1" y="1" width="20" height="20" fill="red"/><text x="5" y="30">A</text></svg>'
    out = sanitize_svg(svg)
    assert "<rect" in out
    assert "<text" in out
    assert out.startswith("<svg")


def test_sanitize_strips_script():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script><rect x="0" y="0" width="10" height="10"/></svg>'
    assert sanitize_svg(svg) == ""


def test_sanitize_strips_event_handlers():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width="10" height="10" onmouseover="alert(1)"/></svg>'
    out = sanitize_svg(svg)
    assert "onmouseover" not in out


def test_sanitize_strips_javascript_href():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><a href="javascript:alert(1)"><rect x="0" y="0" width="10" height="10"/></a></svg>'
    out = sanitize_svg(svg)
    assert "javascript" not in out


def test_sanitize_rejects_non_svg():
    assert sanitize_svg("<html><body>x</body></html>") == ""


def test_sanitize_rejects_malformed():
    assert sanitize_svg("<svg><unclosed>") == ""


def test_sanitize_empty():
    assert sanitize_svg("") == ""
    assert sanitize_svg(None) == ""


def test_svg_data_uri():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect x="1" y="1" width="5" height="5"/></svg>'
    uri = svg_data_uri(svg)
    assert uri.startswith("data:image/svg+xml")
    assert "%3Csvg" in uri
    assert "<svg" not in uri
