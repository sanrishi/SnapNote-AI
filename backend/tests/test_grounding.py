"""Regression tests for trust-text grounding (visual context line).

The "Why this visual matters" summary describes the SOURCE screenshot but
renders under OUR rebuilt visual. Sentences claiming structural elements the
emitted scene does not draw must be removed — never shown.
"""
import asyncio
import json
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.schemas import (
    DeterministicVisual,
    FlowConnector,
    FlowNode,
    ProcessFlow,
    VisualCurve,
    VisualObject,
    VisualPlot,
    VisualScene,
    VisualSpec,
    VisualVector,
)
from app.utils.grounding import ground_visual_context, scene_inventory

transport = ASGITransport(app=app)

# Exact overclaim observed in staging (area-between-curves board).
STAGING_OVERCLAIM = (
    "The graph shows two intersecting curves enclosing a shaded region, "
    "with dashed lines dropping down to the x-axis to mark the integration limits."
)


def _plot_spec() -> VisualSpec:
    return VisualSpec(
        concept="Area",
        deterministic=DeterministicVisual(
            scene=VisualScene(
                scene_kind="plot",
                plot=VisualPlot(
                    x_label="x",
                    y_label="y",
                    show_grid=True,
                    curves=[
                        VisualCurve(label="y_upper", expr="1+3x-2x2"),
                        VisualCurve(label="y_lower", expr="1/x"),
                    ],
                ),
            ),
        ),
    )


def _force_spec() -> VisualSpec:
    from app.models.schemas import ForceDiagram

    return VisualSpec(
        concept="Torque",
        deterministic=DeterministicVisual(
            scene=VisualScene(
                scene_kind="force_diagram",
                force=ForceDiagram(
                    object=VisualObject(label="O"),
                    vectors=[VisualVector(label="F"), VisualVector(label="r")],
                ),
            ),
        ),
    )


def test_staging_overclaim_regression():
    out = ground_visual_context(STAGING_OVERCLAIM, _plot_spec())
    assert "shaded" not in out.lower()
    assert "dashed" not in out.lower()


def test_true_sentence_survives():
    summary = (
        "The graph shows two intersecting curves. "
        "The curves enclose a shaded region with dashed drop lines."
    )
    out = ground_visual_context(summary, _plot_spec())
    assert "two intersecting curves" in out
    assert "shaded" not in out.lower()


def test_arrows_kept_when_vectors_present():
    out = ground_visual_context("The arrows show the applied force.", _force_spec())
    assert "arrows" in out


def test_arrows_dropped_when_absent():
    out = ground_visual_context("The arrows show the applied force.", _plot_spec())
    assert out == ""


def test_dashed_curve_claim_kept():
    spec = _plot_spec()
    assert spec.deterministic.scene is not None and spec.deterministic.scene.plot is not None
    spec.deterministic.scene.plot.curves[0].style = "dashed"
    out = ground_visual_context("The dashed curve is the upper function.", spec)
    assert "dashed" in out


def test_feedback_connector_counts_as_dashed():
    spec = VisualSpec(
        concept="Flow",
        deterministic=DeterministicVisual(
            scene=VisualScene(
                scene_kind="process_flow",
                flow=ProcessFlow(
                    nodes=[FlowNode(label="A"), FlowNode(label="B")],
                    connectors=[FlowConnector(source=0, target=1, feedback=True)],
                ),
            ),
        ),
    )
    out = ground_visual_context("The dashed feedback loop returns output.", spec)
    assert "dashed" in out


def test_empty_summary_stays_empty():
    assert ground_visual_context("", _plot_spec()) == ""
    assert ground_visual_context("   ", _plot_spec()) == ""


def test_inventory_labels_collected():
    inv = scene_inventory(_plot_spec())
    assert inv["has_axes"] and inv["has_grid"]
    assert not inv["has_dashed"] and not inv["has_region_fill"]
    assert "y_upper" in inv["labels"]


# ── route: visual response carries the grounded context ──


def _reset_credits(device_id: str, amount: int = 50) -> None:
    from app.utils.credits_store import _get_conn, init_device
    init_device(device_id)
    conn = _get_conn()
    conn.execute(
        "UPDATE device_credits SET credits_remaining = ?, credits_used = 0 WHERE device_id = ?",
        (amount, device_id),
    )
    conn.commit()


def test_visual_route_returns_grounded_context(sample_diagram_image, monkeypatch):
    from app.utils.credits_store import get_credits

    device = "grounding-device-00000000-0000-000000000004"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    payload = {
        "topic": {"title": "Area", "is_probable": False},
        "what_you_should_remember": "Takeaway.",
        "key_formulas": [],
        "understand_it": [],
        "common_mistakes": [],
        "thirty_second_revision": [],
        "visual_context": {"present": True, "summary": STAGING_OVERCLAIM},
        "verify_before_studying": [],
        "uncertainties": [],
        "analogy": "",
    }
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(payload)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/diagram",
                files={"image": ("d.png", sample_diagram_image, "image/png")},
                data={"context": json.dumps({}), "deviceId": device},
            )

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
    render_mock = AsyncMock(return_value=("png", buf.getvalue()))
    monkeypatch.setattr("app.routes.extract.build_visual_spec", AsyncMock(return_value=_plot_spec()))
    monkeypatch.setattr("app.routes.extract.should_use_v3", lambda spec: True)
    monkeypatch.setattr("app.routes.extract.render_v3_visual", render_mock)

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "generated"
    assert body["visualContext"] is None or "dashed" not in body["visualContext"].lower()
    assert body["visualContext"] is None or "shaded" not in body["visualContext"].lower()
    remaining, _ = get_credits(device)
    assert remaining == 45  # grounding touches no credits
