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

from app.models.schemas import (
    CompositionCallout,
    CompositionReasoningStep,
    CompositionResult,
    LessonComposition,
    VisualSpec,
)
from app.utils import visual_lesson as vl
from app.utils.visual_lesson import (
    LessonValidationError,
    TypstUnavailable,
    build_lesson_typst,
    compile_typst,
    extract_lesson_content,
    render_hero_geometry,
    render_v3_visual,
    render_visual_lesson,
    select_family,
    should_use_v3,
    v3_store_mode,
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


# ── arrowhead nesting (all families, not just force) ──

def _flow_spec():
    from app.models.schemas import FlowConnector, FlowNode, ProcessFlow, VisualScene

    spec = _torque()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.scene_kind = "process_flow"  # type: ignore[assignment]
    spec.deterministic.scene.flow = ProcessFlow(
        nodes=[FlowNode(label="A"), FlowNode(label="B")],
        connectors=[FlowConnector(source=0, target=1, label="go")],
    )
    return spec


def test_arrowheads_never_nested_any_family():
    """Regression (real-input find): plot axis arrowheads were double-wrapped
    as <polygon points="<polygon ...>"/> — valid XML after escaping, so the
    sanitizer passed it and Chromium drew nothing. Every family must emit
    standalone <polygon> elements with numeric points only."""
    import re

    from app.utils.visual_renderer import render_deterministic_visual, render_hero_geometry

    specs = [_torque(), _argand(), _flow_spec(), _v3_torque(), _v3_argand()]
    assert len(specs) == 5
    for spec in specs:
        for svg in (render_deterministic_visual(spec.deterministic), render_hero_geometry(spec.deterministic)):
            assert "<polygon points=\"<polygon" not in svg
            assert "&lt;polygon" not in svg
            for m in re.finditer(r'<polygon points="([^"]+)"', svg):
                vals = m.group(1).split()
                assert len(vals) % 2 == 0
                for v in vals:
                    float(v)


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


# ── v3 composition input ──

def _v3_torque() -> VisualSpec:
    from ground_truth_specs import torque_v3_spec_from_ground_truth

    return torque_v3_spec_from_ground_truth()


def _v3_argand() -> VisualSpec:
    from ground_truth_specs import argand_v3_spec_from_ground_truth

    return argand_v3_spec_from_ground_truth()


def test_v3_validates_clean():
    assert validate_lesson_spec(_v3_torque()) == []
    assert validate_lesson_spec(_v3_argand()) == []


def test_v3_extract_prefers_composition_exact():
    content = extract_lesson_content(_v3_torque(), hero_svg="<svg/>")
    assert content.title == "Torque and Angular Momentum"
    assert content.subtitle == "Pivot → r → F → θ → τ — the turning effect"
    assert [c.heading for c in content.callouts] == ["O", "r", "F", "θ = 35°"]
    assert "θ = 90° − 55° = 35°" in content.reasoning
    assert "τ = r × F" in content.reasoning
    assert content.result_text == "τ = r × F"
    assert content.takeaway.startswith("A force far from the pivot")

    content_a = extract_lesson_content(_v3_argand(), hero_svg="<svg/>")
    assert [c.heading for c in content_a.callouts] == ["A", "B", "C", "D"]
    assert "z = 1+i" in content_a.callouts[0].body
    assert "(1,1)" in content_a.callouts[0].body
    assert "Area = 4" in content_a.result_text
    assert "side s = |B − A| = 2" in content_a.takeaway


def test_v3_rejects_duplicate_ids():
    from pydantic import ValidationError

    spec = _v3_torque()
    assert spec.deterministic.composition is not None
    spec.deterministic.composition.callouts[1].id = spec.deterministic.composition.callouts[0].id
    errors = validate_lesson_spec(spec)
    assert any("duplicated" in e for e in errors)


def test_v3_rejects_empty_composition():
    spec = _v3_torque()
    spec.deterministic.composition = LessonComposition()
    errors = validate_lesson_spec(spec)
    assert any("empty" in e for e in errors)


def test_v3_rejects_empty_callout_label_and_result():
    spec = _v3_torque()
    assert spec.deterministic.composition is not None
    spec.deterministic.composition.callouts[0].label = "  "
    spec.deterministic.composition.result = CompositionResult(expression="  ")
    errors = validate_lesson_spec(spec)
    assert any("empty label" in e for e in errors)
    assert any("empty expression" in e for e in errors)


def test_v3_rejects_unknown_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CompositionCallout(id="x", label="y", value="z", pixel_x=12)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        LessonComposition(title="t", unknown_block="x")  # type: ignore[call-arg]


def test_v3_rejects_too_many_callouts_and_long_label():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LessonComposition(
            callouts=[CompositionCallout(id=f"c{i}", label="l", value="v") for i in range(7)]
        )
    with pytest.raises(ValidationError):
        CompositionCallout(id="c0", label="x" * 25, value="v")


def test_v3_typst_source_carries_composition_once():
    content = extract_lesson_content(_v3_argand(), hero_svg="<svg/>")
    src = build_lesson_typst(content)
    assert "Square on Argand Plane" in src
    assert "Area = 4" in src
    assert src.count("Square on Argand Plane") == 1
    # composition body values pass through byte-identical
    assert "z = 1+i" in src


# ── v3 route wiring (flag + render_v3_visual contract) ──
# These exercise the exact decision + render path the Explain Visually route
# uses, without importing FastAPI modules (see module docstring rationale).

def test_flag_defaults_off():
    from app.config import Settings

    assert Settings.model_fields["EXPLAIN_VISUALLY_V3"].default is False


def test_should_use_v3_default_off():
    # Flag off (default) -> legacy path even for deterministic specs.
    assert should_use_v3(_torque()) is False
    assert should_use_v3(_v3_torque()) is False


def test_should_use_v3_flag_on_deterministic(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "EXPLAIN_VISUALLY_V3", True)
    assert should_use_v3(_torque()) is True
    assert should_use_v3(_v3_torque()) is True
    assert should_use_v3(_v3_argand()) is True


def test_should_use_v3_flag_on_generative_stays_legacy(monkeypatch):
    from app.config import settings
    from app.models.schemas import VisualRenderMode

    monkeypatch.setattr(settings, "EXPLAIN_VISUALLY_V3", True)
    spec = _torque()
    spec.render_mode = VisualRenderMode.GENERATIVE
    assert should_use_v3(spec) is False


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_v3_render_success_fallback_svg():
    # No Typst CLI on Windows -> honest bare-hero SVG fallback, same shape as
    # the legacy deterministic tuple ("svg", svg_string).
    mode, payload = _run(render_v3_visual(_v3_torque()))
    assert mode == "svg"
    assert isinstance(payload, str) and payload.strip().startswith("<svg")
    assert "WHAT THE" not in payload  # bare hero carries no prose


def test_v3_render_validation_failure_returns_none():
    spec = _v3_torque()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.force = None  # fails semantic validation
    assert _run(render_v3_visual(spec)) is None


def test_v3_render_no_double_execution(monkeypatch):
    # The v3 path must never call the legacy deterministic renderer.
    def _boom(*_a, **_k):
        raise AssertionError("legacy generate path must not run under v3")

    monkeypatch.setattr("app.utils.visual_renderer.render_deterministic_visual", _boom)
    mode, payload = _run(render_v3_visual(_v3_argand()))
    assert mode == "svg"
    assert payload.strip().startswith("<svg")


def test_v3_render_deterministic_result(monkeypatch):
    # Force the fallback deterministically (independent of Typst presence):
    # same spec twice -> byte-identical payload.
    from app.utils import visual_lesson as mod

    def _no_typst(*_a, **_k):
        raise mod.TypstCompileError("forced for determinism check")

    monkeypatch.setattr(mod, "compile_typst", _no_typst)
    first = _run(render_v3_visual(_v3_torque()))
    second = _run(render_v3_visual(_v3_torque()))
    assert first == second


def test_v3_render_touches_no_credits(monkeypatch):
    import app.utils.credits_store as store

    def _boom(*_a, **_k):
        raise AssertionError("v3 render must not touch credits")

    monkeypatch.setattr(store, "use_credits", _boom)
    monkeypatch.setattr(store, "add_credits", _boom)
    mode, _ = _run(render_v3_visual(_v3_argand()))
    assert mode == "svg"


def test_v3_store_mode_labels_composed_png_deterministic():
    assert v3_store_mode(True, "png") == "deterministic"
    assert v3_store_mode(True, "svg") == "deterministic"
    assert v3_store_mode(False, "png") == "generative"
    assert v3_store_mode(False, "svg") == "deterministic"


# ── v3 delivery path: render -> exact bytes -> real upload_image -> URL ──

def _fake_imgbb_success(captured: dict):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"success": True, "data": {"url": "https://imgbb.test/v3-lesson.png"}}

    def _post(url, data=None, files=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        captured["calls"] = captured.get("calls", 0) + 1
        return _Resp()

    return _post


def test_v3_delivery_uploads_exact_png_bytes(monkeypatch):
    if not _typst_present():
        import pytest as _pytest

        _pytest.skip("needs Typst CLI for composed PNG (runs on Ubuntu)")
    import httpx

    from app.services.storage_service import upload_image

    mode, payload = _run(render_v3_visual(_v3_torque()))
    assert mode == "png"
    assert isinstance(payload, (bytes, bytearray)) and payload[:8] == b"\x89PNG\r\n\x1a\n"
    captured: dict = {}
    monkeypatch.setattr(httpx, "post", _fake_imgbb_success(captured))
    monkeypatch.setattr("app.config.settings.IMGBB_API_KEY", "test-key", raising=False)
    url = upload_image(bytes(payload), {"title": "explain-visually"})
    assert url == "https://imgbb.test/v3-lesson.png"
    assert captured.get("calls", 0) == 1  # exactly one upload, no orphans/doubles
    assert captured["files"]["image"] == bytes(payload)  # byte-identical file


def test_v3_delivery_storage_failure_returns_none(monkeypatch):
    if not _typst_present():
        import pytest as _pytest

        _pytest.skip("needs Typst CLI for composed PNG (runs on Ubuntu)")
    import httpx

    from app.services.storage_service import upload_image

    mode, payload = _run(render_v3_visual(_v3_argand()))
    assert mode == "png"

    def _boom(*_a, **_k):
        raise ConnectionError("imgbb down")

    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr("app.config.settings.IMGBB_API_KEY", "test-key", raising=False)
    # Existing contract: storage failure -> None -> route raises the
    # controlled UpstreamError (same lines as the generative path).
    assert upload_image(bytes(payload), {"title": "explain-visually"}) is None


def test_v3_delivery_touches_no_credits(monkeypatch):
    import app.utils.credits_store as store

    def _boom(*_a, **_k):
        raise AssertionError("delivery path must not touch credits")

    monkeypatch.setattr(store, "use_credits", _boom)
    monkeypatch.setattr(store, "add_credits", _boom)
    mode, payload = _run(render_v3_visual(_v3_torque()))
    assert mode in ("svg", "png")


# ── production fixture suite: Torque + Argand through the v3 boundary ──

def test_fixture_suite_v3_torque_argand():
    # Proves the route's v3 path actually invokes the composition boundary
    # for both fixture families (vector + coordinate).
    for spec in (_v3_torque(), _v3_argand()):
        assert validate_lesson_spec(spec) == []
        mode, payload = _run(render_v3_visual(spec))
        assert mode == "svg"  # bare-hero fallback where Typst is absent
        assert payload.strip().startswith("<svg")
    # Exact content survives the boundary.
    mode_t, _ = _run(render_v3_visual(_v3_torque()))
    assert mode_t == "svg"
    content = extract_lesson_content(_v3_torque(), hero_svg="<svg/>")
    assert content.result_text == "τ = r × F"
