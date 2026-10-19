"""Contract tests for the production composition boundary (visual_lesson).

Covers: semantic validity, geometry validity, prose ownership, label
collisions, layout overflow/clipping, deterministic output, fallback.
Uses the ground-truth fixtures (torque + argand) — no invented math.
"""
import re
import sys
import os
import asyncio

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
from ground_truth_specs import (
    argand_spec_from_ground_truth,
    torque_spec_from_ground_truth,
    torque_v3_spec_from_ground_truth,
)


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


def test_scene_relation_must_be_an_equation():
    """Case 7 defect: a reconcile repair chose the node label "Plant G(s)"
    as the highlighted relation; labels are not equations."""
    spec = _torque()
    force = spec.deterministic.scene.force  # type: ignore[union-attr]
    assert force is not None and force.relation is not None
    force.relation.expression = "Plant G(s)"
    errors = validate_lesson_spec(spec)
    assert any("not an equation" in e for e in errors)
    force.relation.expression = "τ = r × F"
    assert validate_lesson_spec(spec) == []


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

    specs = [_torque(), _argand(), _flow_spec(), _v3_torque(), _v3_argand(), _flow_v3_spec()]
    assert len(specs) == 6
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
    for m in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*font-size="16"[^>]*>(.*?)</text>', svg):
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

    from app.utils.visual_renderer import render_hero_geometry as _hero

    for spec in (_torque(), _argand()):
        hero = _hero(spec.deterministic)
        content = extract_lesson_content(spec, hero_svg=hero)
        src = build_lesson_typst(content)
        data = compile_typst(src, "pdf", extra_files={"hero.svg": hero})
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


def _assert_v3_shape(mode: str, payload: object) -> None:
    # Same tuple shape as the legacy dispatcher in both branches.
    if mode == "svg":
        assert isinstance(payload, str) and payload.strip().startswith("<svg")
        assert "WHAT THE" not in payload  # bare hero carries no prose
    else:
        assert mode == "png"
        assert isinstance(payload, (bytes, bytearray)) and bytes(payload)[:8] == b"\x89PNG\r\n\x1a\n"


def test_v3_render_success_shape():
    # Typst absent -> ("svg", hero); Typst present -> ("png", bytes).
    # Passes identically on Windows and Ubuntu.
    mode, payload = _run(render_v3_visual(_v3_torque()))
    _assert_v3_shape(mode, payload)


def test_v3_render_fallback_svg_when_typst_fails(monkeypatch):
    # Forced Typst failure -> honest bare-hero SVG fallback, everywhere.
    from app.utils import visual_lesson as mod

    def _no_typst(*_a, **_k):
        raise mod.TypstCompileError("forced for fallback check")

    monkeypatch.setattr(mod, "compile_typst", _no_typst)
    mode, payload = _run(render_v3_visual(_v3_torque()))
    assert mode == "svg"
    assert isinstance(payload, str) and payload.strip().startswith("<svg")
    assert "WHAT THE" not in payload


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
    _assert_v3_shape(mode, payload)


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
    mode, payload = _run(render_v3_visual(_v3_argand()))
    _assert_v3_shape(mode, payload)


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
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "imgbb", raising=False)
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
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "imgbb", raising=False)
    # Existing contract: storage failure -> None -> route raises the
    # controlled UpstreamError (same lines as the generative path).
    assert upload_image(bytes(payload), {"title": "explain-visually"}) is None


def test_v3_delivery_imgbb_http_error_reports_body_without_key(monkeypatch, caplog):
    import httpx

    from app.services import storage_service

    class _Resp:
        status_code = 400
        text = '{"success": false, "error": "bad key SECRET-KEY"}'

    def _post(*_a, **_k):
        return _Resp()

    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setattr("app.config.settings.IMGBB_API_KEY", " SECRET-KEY ", raising=False)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "imgbb", raising=False)
    with caplog.at_level("WARNING", logger="app.services.storage_service"):
        assert storage_service.upload_image(b"not-an-image") is None
    assert "ImgBB HTTP 400" in caplog.text
    assert "SECRET-KEY" not in caplog.text


def test_v3_delivery_touches_no_credits(monkeypatch):
    import app.utils.credits_store as store

    def _boom(*_a, **_k):
        raise AssertionError("delivery path must not touch credits")

    monkeypatch.setattr(store, "use_credits", _boom)
    monkeypatch.setattr(store, "add_credits", _boom)
    mode, payload = _run(render_v3_visual(_v3_torque()))
    _assert_v3_shape(mode, payload)


# ── production fixture suite: Torque + Argand through the v3 boundary ──

def test_fixture_suite_v3_torque_argand():
    # Proves the route's v3 path actually invokes the composition boundary
    # for both fixture families (vector + coordinate).
    for spec in (_v3_torque(), _v3_argand()):
        assert validate_lesson_spec(spec) == []
        mode, payload = _run(render_v3_visual(spec))
        _assert_v3_shape(mode, payload)
    # Exact content survives the boundary.
    content = extract_lesson_content(_v3_torque(), hero_svg="<svg/>")
    assert content.result_text == "τ = r × F"


# ── req 10: engineering process_flow family with explicit v3 composition ──

def _flow_v3_spec() -> VisualSpec:
    """Engineering closed-loop control flow + explicit composition.

    The composition result EQUALS the flow relation expression — the
    contradiction gate (req 6) requires this whenever both are present.
    All values are scene/node labels, not invented facts."""
    from app.models.schemas import (
        CompositionCallout,
        CompositionReasoningStep,
        CompositionResult,
        FlowConnector,
        FlowNode,
        LessonComposition,
        ProcessFlow,
        VisualRelation,
    )

    spec = _torque()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.scene_kind = "process_flow"  # type: ignore[assignment]
    spec.deterministic.scene.force = None
    spec.deterministic.title = "Closed-Loop Control"
    spec.deterministic.scene.caption = "A closed control loop: reference compared with feedback drives the plant."
    spec.deterministic.scene.flow = ProcessFlow(
        nodes=[
            FlowNode(label="r"),
            FlowNode(label="Controller"),
            FlowNode(label="Plant"),
            FlowNode(label="y"),
        ],
        connectors=[
            FlowConnector(source=0, target=1, label="e"),
            FlowConnector(source=1, target=2, label="u"),
            FlowConnector(source=2, target=3, label=""),
            FlowConnector(source=3, target=1, label="feedback", feedback=True),
        ],
        relation=VisualRelation(expression="e = r − y", caption="error = reference minus output"),
    )
    spec.deterministic.composition = LessonComposition(
        title="Closed-Loop Control",
        framing="Reference → error → control → plant → feedback",
        callouts=[
            CompositionCallout(id="node-r", label="r", value="reference input"),
            CompositionCallout(id="node-e", label="e", value="error signal"),
            CompositionCallout(id="node-u", label="u", value="control effort"),
            CompositionCallout(id="node-y", label="y", value="measured output"),
        ],
        reasoning=[
            CompositionReasoningStep(id="rs-err", expression="e = r − y", explanation="compare reference with feedback"),
            CompositionReasoningStep(id="rs-act", expression="u drives plant", explanation="controller acts on the error"),
        ],
        result=CompositionResult(expression="e = r − y", emphasis=True),
        takeaway="Feedback turns the error into the control signal that drives the plant toward the reference.",
    )
    return spec


def test_v3_flow_validates_clean_and_extracts_exact():
    spec = _flow_v3_spec()
    assert validate_lesson_spec(spec) == []
    content = extract_lesson_content(spec, hero_svg="<svg/>")
    assert content.result_text == "e = r − y"
    assert [c.heading for c in content.callouts] == ["r", "e", "u", "y"]
    assert "compare reference with feedback" in content.reasoning
    assert content.takeaway.startswith("Feedback turns the error")


def test_v3_flow_renders_through_boundary():
    spec = _flow_v3_spec()
    mode, payload = _run(render_v3_visual(spec))
    _assert_v3_shape(mode, payload)


# ── req 6: contradiction gate (composition result vs scene relation) ──

def test_v3_rejects_result_contradicting_scene_relation():
    spec = _v3_torque()
    assert spec.deterministic.scene is not None
    assert spec.deterministic.scene.force is not None
    assert spec.deterministic.scene.force.relation is not None
    assert spec.deterministic.composition is not None
    # Sanity: fixture agrees with its own scene relation.
    assert spec.deterministic.scene.force.relation.expression == "τ = r × F"
    # Contradict it: a different "answer" than the scene's own relation.
    spec.deterministic.composition.result = CompositionResult(expression="τ = Iα", emphasis=True)
    errors = validate_lesson_spec(spec)
    assert any("contradicts scene relation" in e for e in errors)


def test_v3_whitespace_only_difference_still_passes():
    spec = _v3_torque()
    assert spec.deterministic.composition is not None
    assert spec.deterministic.composition.result is not None
    spec.deterministic.composition.result.expression = "τ  =  r × F"
    assert validate_lesson_spec(spec) == []


def test_v3_plot_composition_without_scene_relation_passes():
    # Argand has a composition result but the plot scene carries no relation:
    # the gate only fires when BOTH are present, so this must stay valid.
    assert validate_lesson_spec(_v3_argand()) == []


# ── req 12 (part 1): no/unsafe composition → derived fallback, not failure ──

def test_v3_absent_composition_uses_derived_fallback():
    """A spec with NO composition must still render via the v2 derived path
    (equations/steps/points) — never fail merely because composition is absent.
    This is the fallback req 7/12 relies on when evidence is insufficient."""
    spec = _flow_v3_spec()
    spec.deterministic.composition = None
    assert validate_lesson_spec(spec) == []
    content = extract_lesson_content(spec, hero_svg="<svg/>")
    # Derived path: explicit framing is gone, scene-derived content remains,
    # and the visual still renders — absence of composition is never a failure.
    assert "Reference → error" not in content.subtitle
    assert "Feedback turns the error" not in content.takeaway
    assert content.takeaway != ""
    mode, payload = _run(render_v3_visual(spec))
    _assert_v3_shape(mode, payload)


# ── visual-quality hardening: flow layout (R1–R4) ──

def _pid_like_spec():
    """8-node fan-out/fan-in flow mirroring the real PID board spec."""
    from app.models.schemas import FlowConnector, FlowNode, ProcessFlow

    spec = _torque()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.scene_kind = "process_flow"  # type: ignore[assignment]
    spec.deterministic.scene.force = None
    spec.deterministic.scene.flow = ProcessFlow(
        nodes=[FlowNode(label=l) for l in [
            "Set Point r(t)", "Error e(t)", "Proportional Kp × e(t)", "Integral Ki × ∫ e(t) dt",
            "Derivative Kd × de/dt", "Summation S", "Plant G(s)", "Output y(t)"]],
        connectors=[FlowConnector(source=0, target=1, label="r(t)"),
                    FlowConnector(source=1, target=2, label="e(t)"),
                    FlowConnector(source=1, target=3, label="e(t)"),
                    FlowConnector(source=1, target=4, label="e(t)"),
                    FlowConnector(source=2, target=5), FlowConnector(source=3, target=5),
                    FlowConnector(source=4, target=5), FlowConnector(source=5, target=6),
                    FlowConnector(source=6, target=7),
                    FlowConnector(source=7, target=1, label="Feedback (Sensor)", feedback=True)],
    )
    return spec


def _flow_rects(svg: str) -> list[tuple[float, float, float, float]]:
    import re

    out = []
    for m in re.finditer(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg):
        out.append(tuple(float(m.group(i)) for i in range(1, 5)))
    return out


def test_flow_row_always_fits_stage():
    """R1 regression (real PID board): the old max(92, ...) floor pushed an
    8-node row to x0 < 0. Node boxes must stay inside the stage band."""
    from app.utils.visual_renderer import STAGE_W, STAGE_X, render_hero_geometry

    svg = render_hero_geometry(_pid_like_spec().deterministic)
    rects = _flow_rects(svg)
    node_rects = [r for r in rects if r[2] < 300]  # node boxes, not bg/stage
    assert len(node_rects) == 8
    for x, _y, w, _h in node_rects:
        assert x >= STAGE_X - 1, f"box starts left of stage: {x}"
        assert x + w <= STAGE_X + STAGE_W + 1, f"box ends right of stage: {x + w}"


def test_flow_long_labels_wrapped_with_hyphens():
    from app.utils.visual_renderer import _wrap

    lines = _wrap("Proportional Kp × e(t)", 8, hyphenate=True)
    assert lines[0].endswith("-")
    assert max(len(line) for line in lines) <= 8
    # Default path unchanged (no hyphens injected elsewhere).
    assert _wrap("Proportional Kp × e(t)", 8) == ["Proporti", "onal Kp", "× e(t)"]


def test_flow_connector_labels_clear_of_node_band():
    """R2 regression: straight labels sit above boxes; feedback label sits
    below the V — never inside the node band where boxes render."""
    import re

    from app.utils.visual_renderer import render_hero_geometry

    svg = render_hero_geometry(_pid_like_spec().deterministic)
    rects = _flow_rects(svg)
    node_rects = [r for r in rects if r[2] < 300]
    top = min(y for _x, y, _w, _h in node_rects)
    bottom = max(yy + h for _x, yy, _w, h in node_rects)
    texts = [(float(m.group(1)), float(m.group(2)), m.group(3))
             for m in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*font-size="14"[^>]*>(.*?)</text>', svg)]
    assert texts, "expected connector labels"
    for _x, y, label in texts:
        assert y < top - 2 or y > bottom + 2, f"label {label!r} inside node band"


def test_flow_nonadjacent_edges_route_above():
    """Non-adjacent (fan-out/fan-in) edges route above the row; adjacent
    edges stay straight. R4: exact-duplicate edges draw once."""
    import re

    from app.models.schemas import FlowConnector
    from app.utils.visual_renderer import render_hero_geometry

    spec = _pid_like_spec()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.flow is not None
    spec.deterministic.scene.flow.connectors.append(FlowConnector(source=0, target=1, label="r(t)"))
    svg = render_hero_geometry(spec.deterministic)
    elbows = [m.group(0) for m in re.finditer(r"<polyline[^>]*>", svg)]
    # 4 non-adjacent edges ((1,3),(1,4),(2,5),(3,5); (4,5) is adjacent) +
    # 1 feedback V = 5 polylines; the duplicate (0,1) edge must not add
    # another straight line+arrowhead.
    assert len(elbows) == 5
    heads = re.findall(r"<polygon", svg)
    assert len(heads) == 5 + 4 + 0  # 5 straight + 4 elbows, feedback has none


# ── visual-quality hardening: plot (R5 + polar semantics) ──

def _polar_spec() -> VisualSpec:
    from app.models.schemas import VisualCurve, VisualPlot

    spec = _torque()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.scene_kind = "plot"  # type: ignore[assignment]
    spec.deterministic.scene.force = None
    spec.deterministic.scene.plot = VisualPlot(
        x_label="x", y_label="y", x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0,
        curves=[VisualCurve(label="r = 1", expr="1"),
                VisualCurve(label="r = √5", expr="sqrt(5)", color="green"),
                VisualCurve(label="y = 2 ref", expr="2")],
    )
    return spec


def test_polar_radial_bounds_render_as_circles():
    """Polar semantics regression (real math board): r = const bounds must
    draw as circles, not misleading Cartesian horizontal lines."""
    import re

    from app.utils.visual_renderer import render_hero_geometry

    svg = render_hero_geometry(_polar_spec().deterministic)
    assert len(re.findall(r"<ellipse", svg)) == 2
    # The non-radial constant ("y = 2 ref") keeps its honest horizontal line.
    assert len(re.findall(r"<polyline", svg)) == 1


def test_polar_descriptive_radius_labels_become_circles():
    """Generalization (real math board, round 2): bounds labeled
    "Inner Radius g(θ) = 1" must also draw as circles, not lines."""
    import re

    from app.models.schemas import VisualCurve
    from app.utils.visual_renderer import render_hero_geometry

    spec = _polar_spec()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.plot is not None
    spec.deterministic.scene.plot.curves = [
        VisualCurve(label="Inner Radius g(θ) = 1", expr="1"),
        VisualCurve(label="Outer Radius g(θ) = 2", expr="2", color="red"),
    ]
    svg = render_hero_geometry(spec.deterministic)
    assert len(re.findall(r"<ellipse", svg)) == 2
    assert len(re.findall(r"<polyline", svg)) == 0


def test_curve_labels_stay_inside_plot_area():
    """Long curve labels must not spill past the plot/card edge (real math
    board: "Outer Radius..." clipped)."""
    import re

    from app.models.schemas import VisualCurve
    from app.utils.visual_renderer import render_hero_geometry

    spec = _polar_spec()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.plot is not None
    spec.deterministic.scene.plot.curves = [
        VisualCurve(label="Outer Radius g(θ) = some very long bound description here", expr="2"),
    ]
    svg = render_hero_geometry(spec.deterministic)
    for m in re.finditer(r'<text x="([\d.]+)"[^>]*font-size="16"[^>]*font-weight="600"[^>]*>(.*?)</text>', svg):
        x, label = float(m.group(1)), m.group(2)
        est_right = x + len(label) * 9.6 + 8
        assert est_right <= 752 + 4, f"label spills past plot: {label!r}"


def test_polar_function_curves_unaffected():
    import re

    from app.models.schemas import VisualCurve
    from app.utils.visual_renderer import render_hero_geometry

    spec = _polar_spec()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.plot is not None
    spec.deterministic.scene.plot.curves.append(VisualCurve(label="y=x", expr="x"))
    svg = render_hero_geometry(spec.deterministic)
    assert len(re.findall(r"<ellipse", svg)) == 2
    assert len(re.findall(r"<polyline", svg)) == 2


def test_plot_origin_tick_drawn_once():
    """R5 regression (real math board): symmetric ranges jumbled "0.0"/"00"
    at the crossing — now a single 0."""
    import re

    from app.utils.visual_renderer import render_hero_geometry

    svg = render_hero_geometry(_polar_spec().deterministic)
    zeros = re.findall(r">0</text>", svg)
    assert len(zeros) == 1


# ── visual-quality hardening: Typst composition (dedup + page height) ──

def test_typst_figure_caption_does_not_repeat_takeaway():
    from app.utils.visual_lesson import build_lesson_typst, extract_lesson_content

    content = extract_lesson_content(_v3_torque(), hero_svg="<svg/>")
    src = build_lesson_typst(content)
    assert src.count(content.takeaway) == 1
    assert "Figure 1" not in src or content.takeaway not in src.split("Figure 1")[1].split("]")[0]


def test_typst_subtitle_matching_title_dropped():
    from app.models.schemas import LessonComposition
    from app.utils.visual_lesson import build_lesson_typst, extract_lesson_content

    spec = _v3_torque()
    assert spec.deterministic.composition is not None
    spec.deterministic.composition.title = "Torque and Cross Product"
    spec.deterministic.composition.framing = "Torque and Cross Product"
    content = extract_lesson_content(spec, hero_svg="<svg/>")
    src = build_lesson_typst(content)
    assert src.count("Torque and Cross Product") == 1  # title only


def test_typst_page_height_is_content_aware():
    from app.utils.visual_lesson import build_lesson_typst, extract_lesson_content

    src = build_lesson_typst(extract_lesson_content(_v3_torque(), hero_svg="<svg/>"))
    assert "height: auto" in src
    assert "height: 900pt" not in src


def test_derived_flow_reasoning_drops_bare_word_labels():
    """R9: connector labels that merely restate arrows ("allocates") are
    filtered from derived reasoning; symbolic ones ("e(t)") are kept."""
    from app.models.schemas import FlowConnector
    from app.utils.visual_lesson import extract_lesson_content

    spec = _flow_v3_spec()
    spec.deterministic.composition = None
    assert spec.deterministic.scene is not None and spec.deterministic.scene.flow is not None
    spec.deterministic.scene.flow.connectors = [
        FlowConnector(source=0, target=1, label="allocates"),
        FlowConnector(source=1, target=2, label="e(t)"),
        FlowConnector(source=2, target=3, label="Feedback (sensor)"),
    ]
    content = extract_lesson_content(spec, hero_svg="<svg/>")
    assert "allocates" not in content.reasoning
    assert "e(t)" in content.reasoning
    assert "Feedback (sensor)" in content.reasoning


# ── comparison family + generic density gate (production hotfix) ──
# A central box with two formula chips composed over a full page is the
# observed sparse-output failure. Comparison scenes must validate, render a
# real hero, and select; thin generic scenes must refuse honestly.

def _comparison_spec() -> VisualSpec:
    from app.models.schemas import (
        ComparisonPanel,
        DeterministicVisual,
        VisualArc,
        VisualComparison,
        VisualObject,
        VisualRelation,
        VisualScene,
        VisualVector,
    )

    spec = _torque()
    spec.deterministic.scene = VisualScene(
        scene_kind="comparison",
        title="Angular Speed of a Rigid Body",
        caption="Translation moves points together; rotation turns them.",
        comparison=VisualComparison(
            panels=[
                ComparisonPanel(
                    label="Pure translation",
                    object=VisualObject(kind="block"),
                    vectors=[VisualVector(label="v", angle_deg=0, length=1.0)],
                    arcs=[],
                    note="all points share v",
                ),
                ComparisonPanel(
                    label="Pure rotation",
                    object=VisualObject(kind="disk"),
                    vectors=[],
                    arcs=[VisualArc(label="w", direction="ccw")],
                    note="points circle the axis",
                ),
                ComparisonPanel(
                    label="Combined motion",
                    object=VisualObject(kind="disk"),
                    vectors=[VisualVector(label="v", angle_deg=0, length=1.0)],
                    arcs=[VisualArc(label="w", direction="ccw")],
                    note="rolling",
                ),
            ],
            relation=VisualRelation(expression="v = wr", caption="rolling link"),
        ),
    )
    return spec


def test_select_family_comparison():
    assert select_family(_comparison_spec()) == "comparison"


def test_validate_accepts_comparison():
    assert validate_lesson_spec(_comparison_spec()) == []


def test_validate_rejects_thin_comparison():
    spec = _comparison_spec()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.comparison is not None
    spec.deterministic.scene.comparison.panels = spec.deterministic.scene.comparison.panels[:1]
    errors = validate_lesson_spec(spec)
    assert any("2 labeled panels" in e for e in errors)


def test_validate_rejects_empty_comparison_panel():
    from app.models.schemas import ComparisonPanel

    spec = _comparison_spec()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.comparison is not None
    spec.deterministic.scene.comparison.panels[0].vectors = []
    errors = validate_lesson_spec(spec)
    assert any("no vectors or arcs" in e for e in errors)


def test_hero_geometry_comparison_nonempty():
    hero = render_hero_geometry(_comparison_spec().deterministic)
    assert hero.strip()
    assert "Pure translation" in hero and "Combined motion" in hero
    assert "<script" not in hero


def _generic_spec(callouts):
    from app.models.schemas import DeterministicVisual, VisualGeneric, VisualScene

    spec = _torque()
    spec.deterministic.scene = VisualScene(
        scene_kind="generic",
        title="Some Lesson",
        generic=VisualGeneric(central_label="Some Lesson", callouts=list(callouts)),
    )
    return spec


def test_validate_rejects_thin_generic():
    # The reported failure shape: central box + two formula chips.
    errors = validate_lesson_spec(_generic_spec(["chip one", "chip two"]))
    assert any("at least 3 callouts" in e for e in errors)


def test_validate_accepts_dense_generic():
    assert validate_lesson_spec(_generic_spec(["one", "two", "three"])) == []


def test_generic_hero_renders_up_to_satellite_capacity():
    """Capacity is a render contract: every count up to the cap draws, and
    content beyond the cap is visibly absent (the gate mirrors this)."""
    from app.utils.visual_renderer import GENERIC_SATELLITE_CAP, render_hero_geometry

    callouts = [f"item {i}" for i in range(GENERIC_SATELLITE_CAP)]
    hero = render_hero_geometry(_generic_spec(callouts).deterministic)
    for c in callouts:
        assert c in hero
    overflow = render_hero_geometry(
        _generic_spec(callouts + ["overflow item"]).deterministic
    )
    assert "overflow item" not in overflow


# ── tangential-velocity physics rule lives in the spec prompt ──

def test_spec_prompt_requires_tangential_velocity():
    from app.services import vision_service

    prompt = vision_service.VISUAL_SPEC_PROMPT
    assert "TANGENTIAL" in prompt
    assert "never radial" in prompt
    assert "omit the vector rather than guessing" in prompt
    assert "EQUATION FIDELITY" in prompt
    assert "never rewrite Re(z)" in prompt


# ── thin-spec repair decision (semantic-completeness gate) ──

def test_needs_repair_flags_generic_and_empty():
    from app.models.schemas import VisualRenderMode, VisualSpec
    from app.utils.visual_lesson import spec_needs_repair

    assert spec_needs_repair(_generic_spec(["a", "b"])) is not None
    empty_flow = _flow_v3_spec()
    assert empty_flow.deterministic.scene is not None
    empty_flow.deterministic.scene.flow.nodes = []
    assert spec_needs_repair(empty_flow) is not None


def test_needs_repair_passes_rich_and_generative():
    from app.models.schemas import VisualRenderMode, VisualSpec
    from app.utils.visual_lesson import spec_needs_repair

    assert spec_needs_repair(_torque()) is None
    assert spec_needs_repair(_comparison_spec()) is None
    gen = _torque()
    gen.render_mode = VisualRenderMode.GENERATIVE
    assert spec_needs_repair(gen) is None


def test_repair_thin_spec_returns_repaired_or_none(monkeypatch):
    import json
    from unittest.mock import AsyncMock
    from app.services import vision_service

    rich_json = _comparison_spec().model_dump_json()
    monkeypatch.setattr(
        vision_service, "_call_gemini", AsyncMock(return_value=rich_json))
    out = asyncio.run(vision_service.repair_thin_spec(
        _generic_spec(["a", "b"]), b"fake-image", None, "generic family carries no geometry"))
    assert out is not None
    assert out.deterministic.scene is not None
    assert out.deterministic.scene.scene_kind == "comparison"

    async def _boom(*_a, **_k):
        raise vision_service.UpstreamError(service="x", detail="down")

    monkeypatch.setattr(vision_service, "_call_gemini", _boom)
    assert asyncio.run(vision_service.repair_thin_spec(
        _generic_spec(["a", "b"]), b"fake-image", None, "x")) is None


# ── deterministic task derivation (no-task bypass closure) ──

def _strip_task(spec):
    spec.deterministic.task = None
    return spec


def test_derive_task_from_force_scene():
    from app.utils.visual_lesson import derive_task_from_scene, validate_task_link

    task = derive_task_from_scene(_strip_task(_torque()))
    assert task is not None
    assert task.task_type == "vector_arrangement"
    assert task.acceptable_families == ["force_diagram"]
    assert "derived deterministically" in task.assumptions[0]
    spec = _torque()
    spec.deterministic.task = task
    assert validate_task_link(spec) == []


def test_derive_task_empty_scene_returns_none():
    from app.models.schemas import DeterministicVisual, VisualScene, VisualSpec
    from app.utils.visual_lesson import derive_task_from_scene

    blank = VisualSpec(
        concept="Nothing",
        deterministic=DeterministicVisual(
            title="Nothing",
            scene=VisualScene(scene_kind="generic"),
        ),
    )
    assert derive_task_from_scene(blank) is None


def test_derive_task_marks_provenance():
    from app.utils.visual_lesson import derive_task_from_scene

    task = derive_task_from_scene(_strip_task(_flow_v3_spec()))
    assert task is not None
    assert task.task_type == "connected_process"
    assert any("derived" in a for a in task.assumptions)


# ── worked math (number_line): registries, gates, no v3 ──

def _number_line_lesson() -> VisualSpec:
    from app.models.schemas import (
        DeterministicVisual,
        VisualInterval,
        VisualMathCase,
        VisualNumberLine,
        VisualNumberLineTrack,
        VisualScene,
        VisualTask,
    )

    def track(intervals, points=None, label=""):
        return VisualNumberLineTrack(
            label=label,
            intervals=[
                VisualInterval(lo=lo, hi=hi, lo_closed=lo_c, hi_closed=hi_c)
                for lo, hi, lo_c, hi_c in intervals
            ],
            points=points or [],
        )

    nl = VisualNumberLine(
        problem="log_{x+3}(x^2 - x) < 1",
        domain=track(
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
        solution=track(
            [
                ("-3", "-2", False, False),
                ("-1", "0", False, False),
                ("1", "3", False, False),
            ],
            label="Final answer",
        ),
    )
    return VisualSpec(
        concept="logarithmic inequality with a variable base",
        render_mode="deterministic",
        text_required=True,
        deterministic=DeterministicVisual(
            title="Log Inequality \u2014 Full Case Analysis",
            task=VisualTask(
                task_type="interval_solution",
                learning_outcome="solve a logarithmic inequality with a variable base",
                required_elements=["domain", "final answer"],
                required_relationships=[],
                exact_claims=[],
                acceptable_families=["number_line"],
                assumptions=[],
                missing=[],
            ),
            scene=VisualScene(
                scene_kind="number_line",
                title="Log Inequality \u2014 Full Case Analysis",
                caption=(
                    "A logarithmic inequality with a variable base must satisfy "
                    "the domain first; the base regime decides the direction."
                ),
                number_line=nl,
            ),
        ),
    )


def test_number_line_registries():
    assert select_family(_number_line_lesson()) == "worked_math"
    assert vl.TASK_FAMILIES["interval_solution"] == ["number_line"]
    caps = vl.FAMILY_CAPABILITIES["number_line"]
    assert "interval sets on number lines" in caps["represents"]
    assert "unverifiable mathematics (irrational critical points, non-polynomial operands)" in caps["cannot"]


def test_number_line_verified_math_passes_gates():
    from app.utils.visual_lesson import validate_legacy_visual

    spec = _number_line_lesson()
    assert validate_lesson_spec(spec) == []
    assert validate_legacy_visual(spec) == []


def test_number_line_wrong_math_fails_gate():
    spec = _number_line_lesson()
    scene = spec.deterministic.scene
    assert scene is not None and scene.number_line is not None
    scene.number_line.solution.intervals = scene.number_line.solution.intervals[:2]
    errors = validate_lesson_spec(spec)
    assert any("mathematically wrong" in e for e in errors)


def test_number_line_stays_off_v3(monkeypatch):
    from app.utils.visual_lesson import spec_needs_repair

    monkeypatch.setattr("app.config.settings.EXPLAIN_VISUALLY_V3", True, raising=False)
    spec = _number_line_lesson()
    assert should_use_v3(spec) is False
    assert spec_needs_repair(spec) is None


def test_extract_lesson_content_refuses_worked_math():
    with pytest.raises(LessonValidationError, match="deterministic number-line path"):
        extract_lesson_content(_number_line_lesson(), hero_svg="<svg/>")


def test_derive_task_from_number_line_scene():
    from app.models.schemas import VisualTask
    from app.utils.visual_lesson import derive_task_from_scene, validate_task_link

    spec = _number_line_lesson()
    spec.deterministic.task = None
    task = derive_task_from_scene(spec)
    assert task is not None
    assert task.task_type == "interval_solution"
    assert task.acceptable_families == ["number_line"]
    spec.deterministic.task = task
    assert validate_task_link(spec) == []
    assert isinstance(spec.deterministic.task, VisualTask)


def test_derive_task_from_source_number_line():
    from app.models.schemas import StudyNotes
    from app.utils.visual_lesson import derive_task_from_source

    notes = StudyNotes.model_validate(
        {
            "topic": {"title": "Logarithmic inequalities", "is_probable": False},
            "key_formulas": [
                {
                    "formula": "log_{b}(x) < c",
                    "explanation": "base and domain constraints decide the solution set",
                    "uncertain_symbols": [],
                    "confidence": "clear",
                }
            ],
        }
    )
    task = derive_task_from_source(notes, "number_line")
    assert task is not None
    assert task.task_type == "interval_solution"
    assert task.acceptable_families == ["number_line"]
    assert "derived from StudyNotes" in task.assumptions[0]


# ── number_line exact-claim semantic normalization ──


def test_number_line_exact_claim_interval_set_normalized():
    """A final answer written as a canonical interval set must match the scene
    by SET EQUALITY even when the exact written form differs from scene text."""
    spec = _number_line_lesson()
    spec.deterministic.task.exact_claims = ["(-3, -2) \u222a (-1, 0) \u222a (1, 3)"]
    assert vl.validate_task_link(spec) == []


def test_number_line_exact_claim_unicode_subscript_normalized():
    """\u208a/\u2083/\u00b2/\u2212 in a claim are normalized exactly like the verifier's
    own problem text before the comparison."""
    spec = _number_line_lesson()
    spec.deterministic.task.exact_claims = ["logx\u208a\u2083(x\u00b2 \u2212 x) < 1"]
    assert vl.validate_task_link(spec) == []


def test_number_line_wrong_exact_claim_still_refused():
    spec = _number_line_lesson()
    spec.deterministic.task.exact_claims = ["(-3, -2) \u222a (-1, 3)"]
    errors = vl.validate_task_link(spec)
    assert any("exact claim missing" in e for e in errors)

# ── reliability milestone (benchmark cases 2/4/6/7/8/10) ──────────────────


def test_acceptable_families_task_type_alias_expands():
    """Case 8: the planner wrote a task_type name into acceptable_families;
    alias expansion must map it to the registry families (the hard registry
    check still runs)."""
    from app.models.schemas import VisualTask

    spec = _torque()
    det = spec.deterministic
    assert det.scene is not None
    det.task = VisualTask(
        task_type="vector_arrangement",
        required_elements=[],
        exact_claims=[],
        acceptable_families=["vector_arrangement"],
    )
    assert vl.validate_task_link(spec) == []


def test_unknown_acceptable_family_still_refused():
    from app.models.schemas import VisualTask

    spec = _torque()
    det = spec.deterministic
    det.task = VisualTask(
        task_type="vector_arrangement",
        required_elements=[],
        exact_claims=[],
        acceptable_families=["hologram"],
    )
    errors = vl.validate_task_link(spec)
    assert any("acceptable_families" in e for e in errors)


def test_equations_only_claim_is_invisible_and_not_accepted():
    """Trust layer: with a scene present, deterministic.equations render
    nowhere (card content is for scene-less specs only), so a claim living
    only there must NOT silently pass the gate as 'shown'."""
    from app.models.schemas import VisualTask

    spec = _argand()
    det = spec.deterministic
    det.task = VisualTask(
        task_type="coordinate_construction",
        required_elements=[],
        exact_claims=["Area = 4"],
        acceptable_families=["plot"],
    )
    errors = vl.validate_task_link(spec)
    assert any("exact claim missing" in e for e in errors)
    # No composition chip channel here -> promotion cannot rescue; the route
    # then repairs with the model (or refuses honestly).
    assert vl.promote_equation_claims(spec, errors) == 0


def test_promote_equation_claim_into_composition_callouts():
    """A claim satisfied only by an equation is promoted verbatim into a
    visible composition callout chip — no LLM call, content unchanged."""
    from app.models.schemas import VisualTask

    spec = torque_v3_spec_from_ground_truth()
    det = spec.deterministic
    det.task = VisualTask(
        task_type="vector_arrangement",
        required_elements=[],
        exact_claims=["L = I\u03c9"],
        acceptable_families=["force_diagram"],
    )
    errors = vl.validate_task_link(spec)
    assert any("exact claim missing" in e for e in errors)
    assert vl.promote_equation_claims(spec, errors) == 1
    assert vl.validate_task_link(spec) == []
    assert det.composition is not None
    assert any(c.label == "L = I\u03c9" for c in det.composition.callouts)


def test_promote_equation_claim_into_generic_satellites_and_cap():
    """Generic hero renders at most GENERIC_SATELLITE_CAP satellites: claims
    are promoted while capacity remains, and beyond it promotion refuses so
    the model repair rearranges content instead of silently losing it."""
    from app.models.schemas import VisualEquation, VisualGeneric, VisualTask
    from app.utils.visual_renderer import GENERIC_SATELLITE_CAP

    def _generic_spec(filled: int):
        spec = _torque()
        det = spec.deterministic
        scene = det.scene
        assert scene is not None
        scene.scene_kind = "generic"
        scene.force = None
        scene.generic = VisualGeneric(
            central_label="Angular momentum",
            callouts=[f"item {i}" for i in range(filled)],
        )
        det.equations = [VisualEquation(expression="L = I\u03c9", meaning="angular momentum")]
        det.task = VisualTask(
            task_type="definition_overview",
            required_elements=[],
            exact_claims=["L = I\u03c9"],
            acceptable_families=["generic"],
        )
        return spec

    spec = _generic_spec(GENERIC_SATELLITE_CAP - 2)
    errors = vl.validate_task_link(spec)
    assert any("exact claim missing" in e for e in errors)
    assert vl.promote_equation_claims(spec, errors) == 1
    assert vl.validate_task_link(spec) == []
    assert spec.deterministic.scene.generic.callouts[-1] == "L = I\u03c9"

    full = _generic_spec(GENERIC_SATELLITE_CAP)
    errors = vl.validate_task_link(full)
    assert any("exact claim missing" in e for e in errors)
    assert vl.promote_equation_claims(full, errors) == 0


def test_generic_callouts_beyond_capacity_are_not_visible():
    """The gate only counts generic callouts the hero can actually draw."""
    from app.models.schemas import VisualGeneric, VisualTask
    from app.utils.visual_renderer import GENERIC_SATELLITE_CAP

    spec = _torque()
    det = spec.deterministic
    scene = det.scene
    assert scene is not None
    scene.scene_kind = "generic"
    scene.force = None
    scene.generic = VisualGeneric(
        central_label="List",
        callouts=[f"item {i}" for i in range(GENERIC_SATELLITE_CAP)] + ["hidden = 9"],
    )
    det.task = VisualTask(
        task_type="definition_overview",
        required_elements=[],
        exact_claims=["hidden = 9"],
        acceptable_families=["generic"],
    )
    errors = vl.validate_task_link(spec)
    assert any("exact claim missing" in e for e in errors)


def test_plot_structural_tokens_satisfy_slot_elements():
    """Case 2: placeholder slot nouns ("point", "segment") are satisfiable by
    the scene's structural primitives, not by literal label text."""
    from app.models.schemas import PlotMarker, PlotSegment, VisualTask

    spec = _argand()
    det = spec.deterministic
    assert det.scene is not None and det.scene.plot is not None
    det.scene.plot.markers = [PlotMarker(x=2, y=1, label="F")]
    det.scene.plot.segments = [
        PlotSegment(x1=2, y1=0, x2=2, y2=3, label="lens", style="dashed")
    ]
    det.task = VisualTask(
        task_type="coordinate_construction",
        required_elements=["point", "segment"],
        exact_claims=[],
        acceptable_families=["plot"],
    )
    assert vl.validate_task_link(spec) == []


def test_exact_claim_tolerates_multiplication_spacing():
    """Case 10: the same printed expression with/without an explicit × is one
    claim; different values still fail."""
    from app.models.schemas import VisualGeneric, VisualTask

    spec = _torque()
    det = spec.deterministic
    scene = det.scene
    assert scene is not None
    scene.scene_kind = "generic"
    scene.force = None
    scene.generic = VisualGeneric(
        central_label="Normal distribution",
        callouts=["f(z) = (1/√(2π))exp(-z²/2)"],
    )
    det.task = VisualTask(
        task_type="definition_overview",
        required_elements=[],
        exact_claims=["f(z) = (1/√(2π)) × exp(-z²/2)"],
        acceptable_families=["generic"],
    )
    assert vl.validate_task_link(spec) == []

    det.task.exact_claims = ["f(z) = (1/√(2π)) × exp(-z²/3)"]
    assert any("exact claim missing" in e for e in vl.validate_task_link(spec))


def test_plot_accepts_segments_and_markers_without_curves():
    from app.models.schemas import PlotMarker, PlotSegment

    spec = _argand()
    det = spec.deterministic
    assert det.scene is not None and det.scene.plot is not None
    det.scene.plot.curves = []
    det.scene.plot.markers = [PlotMarker(x=2, y=1, label="F")]
    det.scene.plot.segments = [
        PlotSegment(x1=2, y1=0, x2=2, y2=3, label="lens", style="dashed")
    ]
    assert validate_lesson_spec(spec) == []


def test_plot_renders_markers_and_segments():
    from app.models.schemas import PlotMarker, PlotSegment

    spec = _argand()
    det = spec.deterministic
    assert det.scene is not None and det.scene.plot is not None
    det.scene.plot.markers = [PlotMarker(x=2, y=1, label="F")]
    det.scene.plot.segments = [
        PlotSegment(
            x1=2, y1=0, x2=2, y2=3, label="lens", style="dashed", arrow=True
        )
    ]
    svg = render_hero_geometry(det)
    assert "lens" in svg
    assert "F" in svg
    assert 'stroke-dasharray="6 4"' in svg
    assert "<polygon" in svg


def test_flow_drops_connector_label_echoing_node():
    """Case 7: connector labels restating a node's own text are noise."""
    from app.models.schemas import FlowConnector, FlowNode, ProcessFlow

    spec = _torque()
    det = spec.deterministic
    scene = det.scene
    assert scene is not None
    scene.scene_kind = "process_flow"
    scene.force = None
    scene.flow = ProcessFlow(
        nodes=[FlowNode(label="Error e(t)"), FlowNode(label="Plant G(s)")],
        connectors=[FlowConnector(source=0, target=1, label="e(t)")],
    )
    svg = render_hero_geometry(det)
    assert ">e(t)</text>" not in svg
    assert "Error e(t)" in svg


def test_flow_node_labels_never_hyphenated():
    """Case 7: "Proport-ional"/"Summati-on" mid-word hyphens must not return."""
    svg = render_hero_geometry(_pid_like_spec().deterministic)
    assert "-</text>" not in svg
    assert "Proport-ional" not in svg


def test_flow_node_words_never_split_mid_word():
    """Case 7 follow-up: the font search must keep whole words whole
    ("Proportion"/"al" split was still observed after the hyphen ban)."""
    from app.utils.visual_renderer import _wrap

    assert _wrap("Proportional = Kp × e(t)", 8, break_long_words=False) == [
        "Proportional",
        "= Kp ×",
        "e(t)",
    ]
    svg = render_hero_geometry(_pid_like_spec().deterministic)
    assert "Proportion</text>" not in svg
    assert "Proportional" in svg


def test_generic_renders_five_callouts():
    """Case 10: the fifth formula satellite used to be silently dropped."""
    from app.models.schemas import VisualGeneric

    spec = _torque()
    det = spec.deterministic
    scene = det.scene
    assert scene is not None
    scene.scene_kind = "generic"
    scene.force = None
    scene.generic = VisualGeneric(
        central_label="Formula sheet",
        callouts=[
            "a = 1: first",
            "b = 2: second",
            "c = 3: third",
            "d = 4: fourth",
            "e = 5: fifth",
        ],
    )
    svg = render_hero_geometry(det)
    for text in ("first", "second", "third", "fourth", "fifth"):
        assert text in svg


def test_generic_composition_chips_deduplicated():
    """Case 10: generic hero satellites already ARE the callouts; the v3 chip
    grid repeated every formula verbatim below the scene."""
    from app.models.schemas import VisualGeneric

    spec = _torque()
    det = spec.deterministic
    assert det.scene is not None
    det.scene.scene_kind = "generic"
    det.scene.force = None
    det.scene.generic = VisualGeneric(
        central_label="Formula sheet", callouts=["a = 1: first"]
    )
    content = extract_lesson_content(spec)
    assert content.callouts == []


def test_force_ground_satisfies_surface_requirement():
    """Case 4: "horizontal surface" is satisfied by the code-drawn ground."""
    from app.models.schemas import VisualTask

    spec = _torque()
    det = spec.deterministic
    assert det.scene is not None and det.scene.force is not None
    det.scene.force.ground = True
    det.task = VisualTask(
        task_type="vector_arrangement",
        required_elements=["horizontal surface"],
        exact_claims=[],
        acceptable_families=["force_diagram"],
    )
    assert vl.validate_task_link(spec) == []


def test_force_ground_draws_surface():
    """Case 4: a rolling disk needs its contact surface, not a floating body."""
    spec = _torque()
    det = spec.deterministic
    assert det.scene is not None and det.scene.force is not None
    det.scene.force.ground = True
    svg = render_hero_geometry(det)
    assert "surface" in svg
    assert svg.count('stroke-width="2.5"') >= 1


def test_deterministic_visual_coerces_legacy_card_strings():
    """Case 6: repair emitted equations as plain strings -> pydantic crash."""
    from app.models.schemas import DeterministicVisual

    det = DeterministicVisual(
        title="t",
        equations=["1/v - 1/u = 1/f"],
        steps=["apply lens formula"],
        points=["f is focal length"],
    )
    assert det.equations[0].expression == "1/v - 1/u = 1/f"
    assert det.equations[0].meaning == ""
    assert det.steps == ["apply lens formula"]
    assert det.points == ["f is focal length"]
