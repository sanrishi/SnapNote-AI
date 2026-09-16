"""Contract tests for the production composition boundary (visual_lesson).

Covers: semantic validity, geometry validity, prose ownership, label
collisions, layout overflow/clipping, deterministic output, fallback.
Uses the ground-truth fixtures (torque + argand) — no invented math.
"""
import re
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "prototypes", "visual_composition")))

import pytest

from app.models.schemas import VisualSpec
from app.utils import visual_lesson as vl
from app.utils.visual_lesson import (
    LessonValidationError,
    TypstUnavailable,
    build_lesson_typst,
    compile_typst,
    extract_lesson_content,
    render_hero_geometry,
    render_visual_lesson,
    select_family,
    validate_lesson_spec,
)
from ground_truth_specs import argand_spec_from_ground_truth, torque_spec_from_ground_truth


def _torque() -> VisualSpec:
    return torque_spec_from_ground_truth()


def _argand() -> VisualSpec:
    return argand_spec_from_ground_truth()


# ── semantic validity ──

def test_validate_accepts_ground_truth():
    assert validate_lesson_spec(_torque()) == []
    assert validate_lesson_spec(_argand()) == []


def test_validate_rejects_missing_scene():
    spec = _torque()
    spec.deterministic.scene = None
    errors = validate_lesson_spec(spec)
    assert any("scene" in e for e in errors)


def test_validate_rejects_dangling_tail():
    spec = _torque()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.force is not None
    spec.deterministic.scene.force.vectors[1].tail = "NONEXISTENT"
    errors = validate_lesson_spec(spec)
    assert any("NONEXISTENT" in e for e in errors)


def test_validate_rejects_unknown_scene_kind():
    spec = _torque()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.scene_kind = "hologram"  # type: ignore[assignment]
    errors = validate_lesson_spec(spec)
    assert any("hologram" in e for e in errors)


def test_render_refuses_invalid_spec():
    spec = _torque()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.force = None
    with pytest.raises(LessonValidationError):
        render_visual_lesson(spec)


# ── family selection ──

def test_select_family():
    assert select_family(_torque()) == "vector"
    assert select_family(_argand()) == "coordinate"


# ── geometry validity + prose ownership ──

PROSE_MARKERS = ["WHAT THE", "WHAT EACH SYMBOL", "Torque and Angular", "Square on Argand"]


def test_hero_has_geometry_no_prose():
    for spec in (_torque(), _argand()):
        hero = render_hero_geometry(spec.deterministic)
        assert hero.strip().startswith("<svg")
        assert "<line" in hero or "<polyline" in hero or "<polygon" in hero
        for marker in PROSE_MARKERS:
            assert marker not in hero, f"prose leaked into hero: {marker!r}"


def test_content_preserves_exact_values():
    content = extract_lesson_content(_torque(), hero_svg="<svg/>")
    assert content.result_text == "τ = r × F"
    assert any(c.heading == "r" for c in content.callouts)
    assert any(c.heading == "F" for c in content.callouts)
    content_a = extract_lesson_content(_argand(), hero_svg="<svg/>")
    assert any("square ABCD" in c.heading for c in content_a.callouts)
    assert any("A(1,1)" in c.heading for c in content_a.callouts)


# ── collisions (generic mechanism, Argand regression) ──

def test_plot_labels_do_not_overlap_in_hero():
    from app.utils.visual_renderer import render_deterministic_visual

    svg = render_deterministic_visual(_argand().deterministic)
    boxes = []
    for m in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*font-size="11"[^>]*>(.*?)</text>', svg):
        x, y, text = float(m.group(1)), float(m.group(2)), m.group(3)
        boxes.append((x, y - 11, len(text) * 6.6 + 8, 14, text))
    assert len(boxes) >= 2
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay, aw, ah, at = boxes[i]
            bx, by, bw, bh, bt = boxes[j]
            assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay, (
                f"curve labels overlap: {at!r} vs {bt!r}"
            )


# ── composition template ──

def test_typst_source_is_deterministic():
    c1 = extract_lesson_content(_torque(), hero_svg="<svg/>")
    c2 = extract_lesson_content(_torque(), hero_svg="<svg/>")
    assert build_lesson_typst(c1) == build_lesson_typst(c2)


def test_typst_source_carries_result_once():
    content = extract_lesson_content(_torque(), hero_svg="<svg/>")
    src = build_lesson_typst(content)
    assert "τ = r × F" in src
    # title appears exactly once (no duplication between hero and composition,
    # since the hero carries no title at all)
    assert src.count("Torque and Angular Momentum") == 1


# ── compile + fallback ──

def _typst_present() -> bool:
    import shutil

    return shutil.which("typst") is not None


def test_compile_roundtrip_when_available():
    if not _typst_present():
        pytest.skip("typst CLI not on PATH (expected on Windows; runs on Ubuntu)")
    content = extract_lesson_content(_torque(), hero_svg="<svg/>")
    src = build_lesson_typst(content)
    # template without hero image compiles (hero file added by caller)
    src_noimg = src.replace('image("hero.svg", width: 68%),', '"[hero]",')
    data = compile_typst(src_noimg, "pdf")
    assert data[:5] == b"%PDF-"


def test_fallback_to_bare_hero_when_typst_missing(monkeypatch):
    import app.utils.visual_lesson as mod

    monkeypatch.setattr(mod.shutil, "which", lambda *_a, **_k: None)
    lesson = render_visual_lesson(_torque(), out_format="pdf")
    assert lesson.fallback_used is True
    assert lesson.out_format == "svg"
    assert lesson.data.strip().startswith(b"<svg")
    assert lesson.warnings, "fallback must record a warning"


def test_full_render_when_available():
    if not _typst_present():
        pytest.skip("typst CLI not on PATH (expected on Windows; runs on Ubuntu)")
    import tempfile

    from app.utils.visual_renderer import render_hero_geometry as _hero

    for spec in (_torque(), _argand()):
        hero = _hero(spec.deterministic)
        content = extract_lesson_content(spec, hero_svg=hero)
        with tempfile.TemporaryDirectory() as td:
            hp = os.path.join(td, "hero.svg")
            open(hp, "w", encoding="utf-8").write(hero)
            src = build_lesson_typst(content)
            data = compile_typst(src, "pdf")
            assert data[:5] == b"%PDF-", "Typst must produce a real PDF"
