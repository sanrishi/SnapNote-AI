"""Explain Visually must stay a lazy, on-click, once-per-diagram benefit.

Rule 8.3: the visual is never generated automatically at extraction time.
A previous frontend revision auto-called ``handleExplainVisual()`` after every
diagram upload, which contradicted the product contract and spent image-model
work nobody asked for. These are static regression checks on the shipped page.
"""

from __future__ import annotations

from pathlib import Path

WEBSITE = Path(__file__).resolve().parents[2] / "website" / "index.html"


def _html() -> str:
    return WEBSITE.read_text(encoding="utf-8")


def test_explain_visually_is_never_auto_started_after_extraction():
    html = _html()
    assert "setTimeout(() => handleExplainVisual()" not in html
    assert "setTimeout(() => handleExplainVisual" not in html
    assert "auto-generate on every diagram upload" not in html


def test_explain_visually_is_wired_to_the_button_click():
    html = _html()
    assert 'explainBtn.addEventListener("click", handleExplainVisual)' in html
    assert 'id="explain-btn"' in html
    assert 'fetch(`${BASE}/api/extract/visual`' in html


def test_explain_visual_copy_states_the_bundled_no_extra_credit_contract():
    html = _html()
    assert "Free — bundled with your diagram" in html
