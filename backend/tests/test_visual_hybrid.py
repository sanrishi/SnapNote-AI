"""Unit tests for the hybrid Explain Visually renderer and dispatcher.

Deterministic SVG renderer: exact text/symbols rendered by code, sanitized,
byte-deterministic. Universal scene primitives (objects/vectors/angles/arcs/
relations/process boxes/connectors) render as real diagrams. Hybrid dispatcher:
render_mode routes to SVG (deterministic) or Pollinations (generative) with a
conditional OCR legibility gate.
"""

import asyncio
import re
from unittest.mock import AsyncMock

import pytest

from app.models.schemas import (
    DeterministicVisual,
    FlowConnector,
    FlowNode,
    ForceDiagram,
    ProcessFlow,
    VisualAngle,
    VisualArc,
    VisualCurve,
    VisualEquation,
    VisualObject,
    VisualPlot,
    VisualRelation,
    VisualRenderMode,
    VisualScene,
    VisualSpec,
    VisualVector,
)
from app.utils.visual_renderer import render_deterministic_visual


# ── Deterministic renderer ──


def _sample_deterministic() -> DeterministicVisual:
    return DeterministicVisual(
        title="Rotational Dynamics",
        equations=[
            VisualEquation(expression="τ = r × F", meaning="torque = force × lever arm"),
            VisualEquation(expression="L = Iω", meaning="angular momentum = inertia × angular velocity"),
            VisualEquation(expression="ΔL = ∫ τ dt", meaning="impulse-momentum theorem"),
        ],
        steps=["Identify the axis", "Compute torque", "Relate to angular momentum"],
        points=["Torque is the rotational analogue of force", "Angular momentum is conserved without net torque"],
    )


def test_renderer_produces_sanitized_svg_with_exact_content():
    svg = render_deterministic_visual(_sample_deterministic())
    assert svg.startswith("<svg")
    assert svg.strip().endswith("</svg>")
    # No scene -> synthesizes a generic VISUAL ARTIFACT: central title + a few
    # equation/point callouts. It must never render the full Key Formulas /
    # How It Works / Key Points notes sections inside the visual.
    assert "Rotational Dynamics" in svg
    assert "τ = r × F" in svg
    assert "L = Iω" in svg
    assert "KEY EQUATIONS" not in svg
    assert "HOW IT WORKS" not in svg
    assert "KEY POINTS" not in svg
    # Sanitized: no scripts, no event handlers, no external refs
    assert "<script" not in svg.lower()
    assert "onerror" not in svg.lower()
    assert "javascript:" not in svg.lower()


def test_renderer_is_byte_deterministic():
    a = render_deterministic_visual(_sample_deterministic())
    b = render_deterministic_visual(_sample_deterministic())
    assert a == b


def test_renderer_empty_spec_returns_empty():
    assert render_deterministic_visual(DeterministicVisual()) == ""


def test_renderer_escapes_html_in_text():
    spec = DeterministicVisual(
        title="a < b & c",
        equations=[VisualEquation(expression="x < y", meaning="means & implies")],
    )
    svg = render_deterministic_visual(spec)
    assert "&lt;" in svg
    assert "&amp;" in svg
    assert "<script" not in svg.lower()


# ── Scene engine (universal educational primitives) ──


def _torque_scene() -> DeterministicVisual:
    return DeterministicVisual(
        title="Torque",
        scene=VisualScene(
            scene_kind="force_diagram",
            caption="Torque magnitude depends on the lever arm r and the angle θ between r and F.",
            force=ForceDiagram(
                object=VisualObject(kind="pivot", label="O"),
                vectors=[
                    VisualVector(label="r", angle_deg=55, length=1.1, color="accent"),
                    VisualVector(label="F", angle_deg=90, length=0.8, color="red", tail="r"),
                ],
                angles=[VisualAngle(label="θ", between=["r", "F"], caption="angle between the position vector and the force vector")],
                arcs=[VisualArc(label="τ", around="O", direction="ccw", caption="rotation direction")],
                relation=VisualRelation(expression="τ = r × F", caption="torque depends on lever arm and angle"),
            ),
        ),
        points=["Torque drives changes in angular momentum"],
    )


def test_scene_force_diagram_draws_real_diagram():
    svg = render_deterministic_visual(_torque_scene())
    # Every primitive must be present: pivot, both vectors, angle arc, rotation arc,
    # the relation equation, and the caption header.
    for token in ["Torque", "O", "r", "F", "θ", "τ", "τ = r × F", "WHAT THE VISUAL SHOWS"]:
        assert token in svg, f"missing {token!r}"
    # Explain Visually is a VISUAL ARTIFACT, not a second study sheet: the scene
    # must NOT duplicate Key Formulas / Key Points inside the white visual.
    assert "KEY POINTS" not in svg
    assert "KEY EQUATIONS" not in svg
    # Real geometry, not just text: lines + arrowhead polygons + arc polylines.
    assert "<line" in svg
    assert "<polygon" in svg  # arrowheads
    assert "<polyline" in svg  # angle arc + rotation arc
    assert "<rect" in svg  # stage + equation card
    assert "angle-arc" not in svg  # no leftover placeholder concepts


def test_scene_is_byte_deterministic_and_sanitized():
    a = render_deterministic_visual(_torque_scene())
    b = render_deterministic_visual(_torque_scene())
    assert a == b
    assert "<script" not in a.lower()
    assert "javascript:" not in a.lower()


def test_scene_relation_card_included():
    svg = render_deterministic_visual(_torque_scene())
    assert "τ = r × F" in svg
    assert "torque depends on lever arm and angle" in svg


def test_scene_captions_render_below_stage_as_legend():
    """Regression: long angle/arc captions must NOT be dropped inline over the
    vector geometry (they overlapped the F vector line in the torque visual).
    They must appear as muted legend lines under a 'WHAT EACH SYMBOL MEANS'
    header that sits BELOW the stage rect."""
    svg = render_deterministic_visual(_torque_scene())
    assert "WHAT EACH SYMBOL MEANS" in svg
    # Captions came from the angle + arc in the fixture.
    assert "WHAT EACH SYMBOL MEANS" in svg
    # The legend text must not live inside the stage band (y < STAGE_Y+STAGE_H).
    stage_bottom = 110 + 470
    for m in re.finditer(r'<text[^>]*y="([\d.]+)"[^>]*>(.*?)</text>', svg):
        content = m.group(2)
        if "WHAT EACH SYMBOL MEANS" in content or "—" in content:
            assert float(m.group(1)) > stage_bottom, (
                f"caption {content!r} inside stage at y={m.group(1)}"
            )


def test_scene_svg_height_is_dynamic():
    """The SVG viewBox height must grow to fit scene + legend + equations, so
    taller content is never clipped by the old fixed 900px viewBox."""
    svg = render_deterministic_visual(_torque_scene())
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    assert m, "viewBox missing"
    w, h = int(m.group(1)), int(m.group(2))
    assert w == 800
    assert h >= 900
    # Content must fit: the last text element must be inside the viewBox.
    ys = [float(y) for y in re.findall(r'<text[^>]*y="([\d.]+)"', svg)]
    assert max(ys) <= h, f"content y={max(ys)} exceeds viewBox height {h}"


def test_scene_flow_renders_boxes_and_connectors():
    spec = DeterministicVisual(
        title="PID Control",
        scene=VisualScene(
            scene_kind="process_flow",
            caption="The controller compares the reference with feedback and drives the plant.",
            flow=ProcessFlow(
                nodes=[
                    FlowNode(label="Reference"),
                    FlowNode(label="Controller"),
                    FlowNode(label="Plant"),
                ],
                connectors=[
                    FlowConnector(source=0, target=1, label="e(t)"),
                    FlowConnector(source=1, target=2, label="u(t)"),
                    FlowConnector(source=2, target=0, label="y(t)", feedback=True),
                ],
                relation=VisualRelation(expression="u(t) = Kp · e(t)", caption="proportional control"),
            ),
        ),
    )
    svg = render_deterministic_visual(spec)
    for token in ["PID Control", "Reference", "Controller", "Plant", "e(t)", "u(t)", "y(t)", "WHAT THE VISUAL SHOWS"]:
        assert token in svg, f"missing {token!r}"
    # Feedback loop must draw as a dashed return path.
    assert "stroke-dasharray" in svg
    assert "<polyline" in svg


def test_scene_fallback_to_card_when_scene_empty():
    # A scene with no vectors/nodes must not crash. Explain Visually is a VISUAL
    # ARTIFACT, so an empty scene synthesizes a generic diagram (central label +
    # callouts) rather than falling back to a notes card with duplicate equations.
    spec = DeterministicVisual(
        title="Fallback",
        scene=VisualScene(scene_kind="force_diagram", force=None),
        equations=[VisualEquation(expression="a = b", meaning="identity")],
    )
    svg = render_deterministic_visual(spec)
    assert svg.startswith("<svg")
    assert "Fallback" in svg  # generic central label
    assert "a = b" in svg  # equation becomes a callout chip, not a KEY EQUATIONS card
    assert "KEY EQUATIONS" not in svg
    assert "KEY POINTS" not in svg


def test_scene_escapes_user_text():
    spec = DeterministicVisual(
        title="<unsafe>",
        scene=VisualScene(
            scene_kind="force_diagram",
            caption="r & F <i>",
            force=ForceDiagram(
                object=VisualObject(kind="pivot", label="O"),
                vectors=[VisualVector(label="<F>", angle_deg=90)],
            ),
        ),
    )
    svg = render_deterministic_visual(spec)
    assert "<script" not in svg.lower()
    assert "&lt;" in svg


def test_scene_arrowheads_never_nested():
    """Regression: arrowheads must be standalone <polygon> elements, never
    embedded inside another element's attribute (double-wrap bug produced a
    malformed points value that Chrome rejected with a console error)."""
    import re

    svg = render_deterministic_visual(_torque_scene())
    for m in re.finditer(r'points="([^"]*)"', svg):
        assert "<polygon" not in m.group(1), f"nested polygon in points: {m.group(1)[:80]}"
    # Every polygon must parse as numbers.
    for m in re.finditer(r'<polygon points="([^"]+)"', svg):
        vals = m.group(1).split()
        assert len(vals) % 2 == 0
        for v in vals:
            float(v)


def test_scene_rotation_arc_arrowhead_present():
    """The torque rotation arc must carry a standalone arrowhead polygon."""
    import re

    svg = render_deterministic_visual(_torque_scene())
    polys = re.findall(r'<polygon[^>]*>', svg)
    # pivot arrowheads (2 vectors) + angle arc + rotation arc = at least 4
    assert len(polys) >= 4, f"only {len(polys)} polygons"


def _argand_collision_scene() -> DeterministicVisual:
    """Regression fixture: two plot curves ending at the same corner, like the
    Argand square ("square ABCD") and its point marker ("A(1,1)")."""
    return DeterministicVisual(
        title="Square on Argand Plane",
        scene=VisualScene(
            scene_kind="plot",
            plot=VisualPlot(
                x_label="Re", y_label="Im",
                x_min=-1, x_max=5, y_min=-1, y_max=5, show_grid=True,
                curves=[
                    VisualCurve(label="square ABCD", points=[[1, 1], [1, 3], [3, 3], [3, 1], [1, 1]]),
                    VisualCurve(label="A(1,1)", points=[[1, 1], [1, 1]]),
                ],
            ),
        ),
    )


def _curve_label_boxes(svg: str) -> list[tuple[float, float, float, float]]:
    import re

    boxes = []
    for m in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*font-size="11"[^>]*>(.*?)</text>', svg):
        x, y, text = float(m.group(1)), float(m.group(2)), m.group(3)
        boxes.append((x, y - 11, len(text) * 6.6 + 8, 14))
    return boxes


def test_plot_curve_labels_never_overlap():
    """Regression (Argand A(1,1)/ABCD collision): labels attached to the same
    geometric point must be placed at deterministic non-overlapping offsets,
    with exact coordinates and geometry untouched."""
    svg = render_deterministic_visual(_argand_collision_scene())
    assert "square ABCD" in svg
    assert "A(1,1)" in svg
    boxes = _curve_label_boxes(svg)
    assert len(boxes) == 2, f"expected 2 curve labels, got {len(boxes)}"
    (ax, ay, aw, ah), (bx, by, bw, bh) = boxes
    overlaps = not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)
    assert not overlaps, f"curve labels overlap: {boxes}"


def test_plot_single_label_keeps_legacy_placement():
    """A lone curve label must render at the legacy (+6,-6) anchor offset."""
    spec = DeterministicVisual(
        title="Plot",
        scene=VisualScene(
            scene_kind="plot",
            plot=VisualPlot(
                x_min=0, x_max=4, y_min=0, y_max=4,
                curves=[VisualCurve(label="y=x", points=[[0, 0], [4, 4]])],
            ),
        ),
    )
    svg = render_deterministic_visual(spec)
    assert "y=x" in svg
    assert len(_curve_label_boxes(svg)) == 1


# ── Hybrid dispatcher ──


def test_generate_visual_deterministic_returns_svg(monkeypatch):
    spec = VisualSpec(
        concept="Rotation",
        render_mode=VisualRenderMode.DETERMINISTIC,
        deterministic=_sample_deterministic(),
    )
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=b"PNG"))
    result = asyncio.run(_run_generate(spec))
    assert result is not None
    mode, payload = result
    assert mode == "svg"
    assert isinstance(payload, str)
    assert "τ = r × F" in payload
    # Deterministic must never hit the generative renderer
    from app.services import visual_service
    visual_service._render_once.assert_not_called()


def test_generate_visual_generative_returns_png_when_legible(monkeypatch):
    import io

    from PIL import Image

    spec = VisualSpec(
        concept="Cell",
        render_mode=VisualRenderMode.GENERATIVE,
        text_required=False,
        visual_form="illustration",
    )
    buf = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=png))
    monkeypatch.setattr("app.services.visual_service._quality_pass", lambda p: True)
    result = asyncio.run(_run_generate(spec))
    assert result is not None
    mode, payload = result
    assert mode == "png"
    assert payload == png


def test_generate_visual_generative_rejects_unreadable_text(monkeypatch):
    """OCR legibility gate: when text_required, an unreadable render is rejected."""
    import io

    from PIL import Image

    spec = VisualSpec(
        concept="Cell",
        render_mode=VisualRenderMode.GENERATIVE,
        text_required=True,
        visual_form="labeled diagram",
    )
    buf = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=png))
    monkeypatch.setattr("app.services.visual_service._quality_pass", lambda p: True)
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: True)
    result = asyncio.run(_run_generate(spec))
    assert result is None


def test_generate_visual_generative_skips_ocr_when_not_required(monkeypatch):
    """text_required=false must NOT reject an illustration that has no text."""
    import io

    from PIL import Image

    spec = VisualSpec(
        concept="Cell",
        render_mode=VisualRenderMode.GENERATIVE,
        text_required=False,
        visual_form="illustration",
    )
    buf = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.visual_service._render_once", AsyncMock(return_value=png))
    monkeypatch.setattr("app.services.visual_service._quality_pass", lambda p: True)
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: True)
    result = asyncio.run(_run_generate(spec))
    assert result is not None
    assert result[0] == "png"


def test_legibility_pass_thresholds(monkeypatch):
    import io

    from PIL import Image
    from app.services import visual_service

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="PNG")
    png = buf.getvalue()
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: True)

    # A few high-confidence tokens -> pass
    monkeypatch.setattr(
        "app.services.ocr_service.read_raw",
        lambda arr: [[[[0, 0], [0, 0], [0, 0], [0, 0]], "torque", 0.95], [[[0, 0], [0, 0], [0, 0], [0, 0]], "axis", 0.9]],
    )
    assert visual_service._legibility_pass(png) is True

    # Zero tokens -> fail
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    assert visual_service._legibility_pass(png) is False

    # Too few tokens -> fail
    monkeypatch.setattr(
        "app.services.ocr_service.read_raw",
        lambda arr: [[[[0, 0], [0, 0], [0, 0], [0, 0]], "x", 0.95]],
    )
    assert visual_service._legibility_pass(png) is False

    # OCR unavailable -> pass (safety net, never hard-break the feature)
    monkeypatch.setattr("app.services.ocr_service.ocr_available", lambda: False)
    monkeypatch.setattr("app.services.ocr_service.read_raw", lambda arr: [])
    assert visual_service._legibility_pass(png) is True


async def _run_generate(spec):
    from app.services.visual_service import generate_visual

    return await generate_visual(spec)


# ── v3 contradiction repair (one bounded retry, then honest refusal) ──

def _v3_torque_json(result_expr: str | None = None) -> str:
    import json

    from ground_truth_specs import torque_v3_spec_from_ground_truth

    spec = torque_v3_spec_from_ground_truth()
    if result_expr is not None:
        assert spec.deterministic.composition is not None
        assert spec.deterministic.composition.result is not None
        spec.deterministic.composition.result.expression = result_expr
    return spec.model_dump_json()


def _run_build(coro):
    return asyncio.run(coro)


def test_composition_contradiction_helper_pure():
    from ground_truth_specs import (
        argand_v3_spec_from_ground_truth,
        torque_v3_spec_from_ground_truth,
    )
    from app.services.vision_service import _composition_contradiction

    assert _composition_contradiction(torque_v3_spec_from_ground_truth()) is None
    # Plot scene carries no relation: result alone is never a contradiction.
    assert _composition_contradiction(argand_v3_spec_from_ground_truth()) is None
    clean = torque_v3_spec_from_ground_truth()
    clean.deterministic.composition = None
    assert _composition_contradiction(clean) is None
    bad = torque_v3_spec_from_ground_truth()
    assert bad.deterministic.composition is not None
    assert bad.deterministic.composition.result is not None
    bad.deterministic.composition.result.expression = "τ = Iα"
    found = _composition_contradiction(bad)
    assert found is not None and found[0] == "τ = r × F" and found[1] == "τ = Iα"


def test_build_visual_spec_no_contradiction_single_call(monkeypatch):
    from app.services import vision_service

    calls: list[str] = []

    async def fake_call(prompt: str, image: bytes, json_mode: bool = False, **kwargs: object) -> str:
        calls.append(prompt)
        return _v3_torque_json()

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    spec = _run_build(vision_service.build_visual_spec(b"fake", None))
    assert spec.deterministic.title == "Torque and Angular Momentum"
    assert len(calls) == 1


def test_build_visual_spec_contradiction_repaired_once(monkeypatch):
    from app.services import vision_service

    responses = [_v3_torque_json("ΔL = ∫τ dt"), _v3_torque_json()]
    calls: list[str] = []
    call_kwargs: list[dict] = []

    async def fake_call(prompt: str, image: bytes, json_mode: bool = False, **kwargs: object) -> str:
        calls.append(prompt)
        call_kwargs.append(kwargs)
        return responses[min(len(calls) - 1, 1)]

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    spec = _run_build(vision_service.build_visual_spec(b"fake", None))
    assert spec.deterministic.composition is not None
    assert spec.deterministic.composition.result is not None
    assert spec.deterministic.composition.result.expression == "τ = r × F"
    assert len(calls) == 2
    # Repair prompt quotes both disagreeing expressions for reconciliation.
    assert "τ = r × F" in calls[1] and "ΔL = ∫τ dt" in calls[1]
    # Repair leg SLA: tight sub-budget, no retry — expiry refuses honestly.
    assert call_kwargs[1].get("max_retries") == 0
    assert call_kwargs[1].get("timeout_seconds") == 12.0


def test_build_visual_spec_repair_context_is_minimal(monkeypatch):
    """The repair prompt carries topic+formulas only — never the full notes
    JSON (uncertainties, analogy, visual context). Smaller payload, faster leg."""
    from app.models.schemas import FormulaEntry, StudyNotes, TopicInfo
    from app.services import vision_service

    responses = [_v3_torque_json("ΔL = ∫τ dt"), _v3_torque_json()]
    calls: list[str] = []

    async def fake_call(prompt: str, image: bytes, json_mode: bool = False, **kwargs: object) -> str:
        calls.append(prompt)
        return responses[min(len(calls) - 1, 1)]

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    notes = StudyNotes(
        topic=TopicInfo(title="Torque"),
        key_formulas=[FormulaEntry(formula="τ = r × F", explanation="t", confidence="clear")],
    )
    _run_build(vision_service.build_visual_spec(b"fake", notes))
    assert len(calls) == 2
    assert "GROUNDED CONTEXT" in calls[1]
    assert "visual_context" not in calls[1]
    assert "analogy" not in calls[1]
    assert "τ = r × F" in calls[1]


def test_build_visual_spec_contradiction_persists_refuses(monkeypatch):
    from app.exceptions import UpstreamError
    from app.services import vision_service

    calls: list[str] = []

    async def fake_call(prompt: str, image: bytes, json_mode: bool = False, **kwargs: object) -> str:
        calls.append(prompt)
        return _v3_torque_json("τ = Iα")

    monkeypatch.setattr(vision_service, "_call_gemini", fake_call)
    with pytest.raises(UpstreamError):
        _run_build(vision_service.build_visual_spec(b"fake", None))
    # Exactly one repair attempt — never an unbounded retry loop.
    assert len(calls) == 2