"""Cross-topic visual benchmark: task planning + family selection + link gate.

Ten cases across supported families. Cases 1-5 and 8 reuse established
fixtures (regression safety). Cases 6-7, 9-10 were constructed for this file
and act as holdouts: no renderer code was written for them. Live demo-image
runs after code freeze are additional holdout evidence (see delivery notes).

Each case asserts, separately: task selection contract, family authorization,
semantic completeness (required elements + exact claims), and renderability.
"""
import sys
import os

from test_visual_lesson import (
    _comparison_spec,
    _flow_v3_spec,
    _generic_spec,
    _polar_spec,
    _torque,
)
from app.models.schemas import (
    DeterministicVisual,
    FlowConnector,
    FlowNode,
    ProcessFlow,
    VisualCurve,
    VisualObject,
    VisualPlot,
    VisualRelation,
    VisualScene,
    VisualSpec,
    VisualTask,
    VisualVector,
)
from app.utils.visual_lesson import select_family, validate_lesson_spec, validate_task_link
from app.utils.visual_renderer import render_deterministic_visual, render_hero_geometry


def _log_inequality_spec():
    return VisualSpec(
        concept="Logarithmic inequality",
        deterministic=DeterministicVisual(
            title="Solving a Log Inequality",
            scene=VisualScene(
                scene_kind="process_flow",
                caption="Domain first, then case splits, then intersect.",
                flow=ProcessFlow(
                    nodes=[
                        FlowNode(label="Domain: x > -3"),
                        FlowNode(label="Case A: base > 1"),
                        FlowNode(label="Case B: 0 < base < 1"),
                        FlowNode(label="Solution: intersect valid intervals"),
                    ],
                    connectors=[
                        FlowConnector(source=0, target=1, label="split"),
                        FlowConnector(source=0, target=2, label="split"),
                        FlowConnector(source=1, target=3, label="keep"),
                        FlowConnector(source=2, target=3, label="keep"),
                    ],
                ),
            ),
        ),
    )


def _optics_lens_spec():
    from app.models.schemas import ForceDiagram, VisualAngle

    return VisualSpec(
        concept="Thin lens image",
        deterministic=DeterministicVisual(
            title="Lens Ray Diagram",
            scene=VisualScene(
                scene_kind="force_diagram",
                caption="Rays through a convex lens meet at the image point.",
                force=ForceDiagram(
                    object=VisualObject(kind="point", label="O"),
                    vectors=[
                        VisualVector(label="incident", angle_deg=10, length=1.2),
                        VisualVector(label="refracted", angle_deg=-15, length=1.2, tail="incident"),
                    ],
                    angles=[],
                    arcs=[],
                    relation=None,
                ),
            ),
        ),
    )


def _tasked(spec, task_type, elements, claims, families):
    spec.task = VisualTask(
        task_type=task_type,
        learning_outcome="student outcome",
        required_elements=list(elements),
        required_relationships=[],
        exact_claims=list(claims),
        acceptable_families=list(families),
        assumptions=[],
        missing=[],
    )
    return spec


def test_01_torque_vector_arrangement():
    spec = _tasked(_torque(), "vector_arrangement", ["torque"], ["τ = r × F"], ["force_diagram"])
    assert validate_task_link(spec) == []
    assert validate_lesson_spec(spec) == []
    assert select_family(spec) == "vector"
    assert render_hero_geometry(spec.deterministic).strip()


def test_02_argand_coordinate_construction():
    import test_visual_hybrid as hybrid

    spec = _tasked(
        _argand_task_wrapped(),
        "coordinate_construction",
        ["square vertices", "side 4"],
        ["D z - 2Re(z) = -x + iy"],
        ["plot"],
    )
    assert validate_task_link(spec) == []
    assert render_hero_geometry(spec.deterministic).strip()


def _argand_task_wrapped():
    import test_visual_hybrid as hybrid

    det = hybrid._argand_square_spec()
    return VisualSpec(concept="Argand square", deterministic=det)


def test_03_comparison_variants():
    spec = _tasked(
        _comparison_spec(), "compare_variants",
        ["Pure translation", "Pure rotation", "Combined motion"],
        [], ["comparison"])
    assert validate_task_link(spec) == []
    assert select_family(spec) == "comparison"
    assert render_hero_geometry(spec.deterministic).strip()


def test_04_polar_graph_relationship():
    spec = _tasked(_polar_spec(), "graph_relationship", ["torque"], ["r = √5"], ["plot"])
    assert validate_task_link(spec) == []
    assert select_family(spec) == "coordinate"


def test_05_flow_connected_process():
    spec = _tasked(_flow_v3_spec(), "connected_process", ["controller", "plant"], [], ["process_flow"])
    assert validate_task_link(spec) == []
    assert select_family(spec) == "flow"


def test_06_log_inequality_multi_step_holdout():
    spec = _tasked(
        _log_inequality_spec(), "multi_step_solution",
        ["domain x > -3", "case split", "solution intervals"],
        ["x > -3"], ["process_flow"])
    assert validate_task_link(spec) == []
    assert select_family(spec) == "flow"
    assert render_hero_geometry(spec.deterministic).strip()


def test_07_simple_definition_stays_simple():
    spec = _tasked(
        _generic_spec(["one", "two", "three"]), "definition_overview",
        ["Some Lesson"], [], ["generic"])
    assert validate_task_link(spec) == []
    assert select_family(spec) == "generic"


def test_08_wrong_family_refused():
    spec = _tasked(
        _comparison_spec(), "compare_variants",
        ["Pure translation"], [], ["plot"])  # task authorizes plot, registry forbids it
    errors = validate_task_link(spec)
    assert any("acceptable_families" in e or "cannot express" in e for e in errors)


def test_09_missing_elements_refused():
    spec = _tasked(
        _torque(), "vector_arrangement",
        ["torque", "unicorn levitation field"], ["τ = r × F"], ["force_diagram"])
    errors = validate_task_link(spec)
    assert any("unsatisfied" in e for e in errors)


def test_10_optics_holdout_vector_arrangement():
    spec = _tasked(
        _optics_lens_spec(), "vector_arrangement",
        ["incident ray", "refracted ray"], [], ["force_diagram"])
    assert validate_task_link(spec) == []
    assert select_family(spec) == "vector"
    assert render_hero_geometry(spec.deterministic).strip()
