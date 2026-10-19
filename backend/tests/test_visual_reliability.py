"""Reliability-milestone regressions outside the renderer/gate layer.

Covers the bounded admission gate (timeout/credit fairness), confidence-label
normalization (trust layer), the diagram/visual-context contradiction reword,
and exact-claim math canonicalization.
"""
import pytest

from app.models.schemas import DiagramSpec, StudyNotes, VisualContext
from app.services import vision_service
from app.utils.image_window import admit_image_work, release_image_work


def test_admission_allows_one_and_rejects_second():
    from app.utils.image_window import MAX_ADMITTED

    assert MAX_ADMITTED == 1
    assert admit_image_work() is True
    try:
        assert admit_image_work() is False
    finally:
        release_image_work()
    assert admit_image_work() is True
    release_image_work()


def test_admission_release_below_zero_is_safe():
    release_image_work()
    assert admit_image_work() is True
    release_image_work()


def test_server_busy_error_is_503():
    from app.exceptions import ServerBusyError

    err = ServerBusyError()
    assert err.status_code == 503
    assert "try again" in err.message.lower()


def test_confidence_recovers_confusable_subscript():
    notes = StudyNotes(
        key_formulas=[
            {"formula": "τ = Iα", "confidence": "possibleₑxtraction_issue"}
        ]
    )
    vision_service._normalize_confidence(notes)
    assert notes.key_formulas[0].confidence == "possible_extraction_issue"


def test_confidence_canonicalizes_case_drift():
    notes = StudyNotes(
        key_formulas=[
            {"formula": "a = b", "confidence": "Context Needed"},
            {"formula": "c = d", "confidence": "CLEAR"},
        ]
    )
    vision_service._normalize_confidence(notes)
    assert notes.key_formulas[0].confidence == "context_needed"
    assert notes.key_formulas[1].confidence == "clear"


def test_unknown_confidence_is_preserved():
    notes = StudyNotes(
        key_formulas=[{"formula": "a = b", "confidence": "totally_unknown"}]
    )
    vision_service._normalize_confidence(notes)
    assert notes.key_formulas[0].confidence == "totally_unknown"


@pytest.mark.asyncio
async def test_no_diagram_warning_reworded_when_visual_context_present(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        vision_service,
        "validate_diagram_spec",
        lambda spec: SimpleNamespace(
            valid=False,
            canonical=None,
            reasons=["No diagram was detected in the screenshot."],
        ),
    )
    notes = StudyNotes(
        visual_context=VisualContext(present=True, summary="Argand diagram")
    )
    notes.diagram_spec = DiagramSpec(diagram_type="polar_region")
    await vision_service._apply_diagram_spec(notes, b"")
    assert not any("No diagram was detected" in u for u in notes.uncertainties)
    assert any("could not be rebuilt" in u for u in notes.uncertainties)


@pytest.mark.asyncio
async def test_no_diagram_warning_kept_without_visual_context(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        vision_service,
        "validate_diagram_spec",
        lambda spec: SimpleNamespace(
            valid=False,
            canonical=None,
            reasons=["No diagram was detected in the screenshot."],
        ),
    )
    notes = StudyNotes(visual_context=VisualContext(present=False))
    notes.diagram_spec = DiagramSpec(diagram_type="polar_region")
    await vision_service._apply_diagram_spec(notes, b"")
    assert any("No diagram was detected" in u for u in notes.uncertainties)


def test_generic_callouts_normalized_like_claims():
    from app.models.schemas import VisualGeneric, VisualSpec
    from app.utils.latex_clean import normalize_spec_math

    spec = VisualSpec.model_validate(
        {
            "concept": "normal distribution",
            "deterministic": {
                "title": "Normal distribution",
                "scene": {
                    "scene_kind": "generic",
                    "generic": {
                        "central_label": "Distribution",
                        "callouts": ["f(z) = (1/sqrt(2π))exp(-z²/2)"],
                    },
                },
            },
        }
    )
    normalized = normalize_spec_math(spec)
    callouts = normalized.deterministic.scene.generic.callouts
    assert callouts == ["f(z) = (1/√(2π))exp(-z²/2)"]


def test_required_elements_normalized_with_scene_labels():
    """Case 7: the scene label "Summation S" is normalized to "Σ"; the task
    requirement must be normalized identically or the gate false-fails."""
    from app.models.schemas import (
        DeterministicVisual,
        FlowConnector,
        FlowNode,
        ProcessFlow,
        VisualRenderMode,
        VisualScene,
        VisualSpec,
        VisualTask,
    )
    from app.utils.latex_clean import normalize_spec_math

    spec = VisualSpec(
        concept="PID",
        render_mode=VisualRenderMode.DETERMINISTIC,
        text_required=True,
        deterministic=DeterministicVisual(
            title="PID control",
            scene=VisualScene(
                scene_kind="process_flow",
                caption="c",
                flow=ProcessFlow(
                    nodes=[FlowNode(label="Summation S"), FlowNode(label="Plant G(s)")],
                    connectors=[FlowConnector(source=0, target=1)],
                ),
            ),
            task=VisualTask(
                task_type="connected_process",
                required_elements=["Summation S"],
                acceptable_families=["process_flow"],
            ),
        ),
    )
    normalize_spec_math(spec)
    det = spec.deterministic
    assert det.scene.flow.nodes[0].label == "Σ"
    assert det.task is not None and det.task.required_elements == ["Σ"]
    from app.utils.visual_lesson import validate_task_link

    assert validate_task_link(spec) == []


def test_generic_callout_dicts_coerced_to_strings():
    """Observed repair emission (case 8): callouts as {id,label,value}
    objects must coerce instead of hard-failing the whole spec."""
    from app.models.schemas import VisualGeneric, VisualScene

    scene = VisualScene.model_validate(
        {
            "scene_kind": "generic",
            "generic": {
                "central_label": "Cell Theory",
                "callouts": [
                    {"id": "claim1", "label": "Tenet 1", "value": "All living organisms are composed of cells"},
                    {"id": "claim2", "value": "Cells arise from pre-existing cells"},
                    "Plain string callout",
                ],
            },
        }
    )
    generic = scene.generic
    assert generic is not None
    assert generic.callouts == [
        "Tenet 1: All living organisms are composed of cells",
        "Cells arise from pre-existing cells",
        "Plain string callout",
    ]
    assert VisualGeneric.model_validate({"callouts": None}).callouts == []


def test_composition_ids_folded_to_ascii():
    from app.services.vision_service import _normalize_composition_ids

    data = {
        "deterministic": {
            "composition": {
                "callouts": [{"id": "eqₐ", "label": "a", "value": ""}],
                "reasoning": [{"id": "rs₁", "expression": "", "explanation": ""}],
            }
        }
    }
    out = _normalize_composition_ids(data)
    assert out["deterministic"]["composition"]["callouts"][0]["id"] == "eqa"
    assert out["deterministic"]["composition"]["reasoning"][0]["id"] == "rs1"


def test_number_line_points_accept_val_objects():
    from app.models.schemas import VisualNumberLineTrack

    track = VisualNumberLineTrack(
        label="domain",
        points=[{"val": "-1", "open": True}, "0", {"val": "3"}],
    )
    assert track.points == ["-1", "0", "3"]


def test_spec_math_canonicalizes_exact_claims():
    from app.models.schemas import VisualSpec
    from app.utils.latex_clean import normalize_spec_math

    spec = VisualSpec.model_validate(
        {
            "concept": "distance",
            "deterministic": {
                "title": "Distance",
                "task": {
                    "task_type": "graph_relationship",
                    "exact_claims": ["d = sqrt(x**2 + y**2)"],
                    "acceptable_families": ["plot"],
                },
                "scene": {
                    "scene_kind": "plot",
                    "plot": {
                        "x_label": "x",
                        "y_label": "y",
                        "x_min": -2,
                        "x_max": 2,
                        "y_min": -2,
                        "y_max": 2,
                        "curves": [{"label": "y = x", "expr": "x", "x_min": 0, "x_max": 1}],
                    },
                },
            },
        }
    )
    normalized = normalize_spec_math(spec)
    assert normalized.deterministic.task is not None
    claims = normalized.deterministic.task.exact_claims
    assert any("√" in claim for claim in claims)
    assert not any("sqrt(" in claim for claim in claims)
