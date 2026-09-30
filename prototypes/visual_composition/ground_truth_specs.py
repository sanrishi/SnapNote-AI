"""Build VisualSpec v2 from ground-truth JSON — SINGLE SOURCE for A and B.

Both the current-SVG baseline and the rich Typst composition render from
these specs, which are constructed field-by-field from
ground_truth_argand.json / ground_truth_torque.json. Any drift fails
validation (see validate_ground_truth.py) before rendering.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from app.models.schemas import (
    CompositionCallout,
    CompositionReasoningStep,
    CompositionResult,
    DeterministicVisual,
    FlowConnector,
    FlowNode,
    ForceDiagram,
    LessonComposition,
    ProcessFlow,
    VisualAngle,
    VisualArc,
    VisualCurve,
    VisualEquation,
    VisualGeneric,
    VisualObject,
    VisualPlot,
    VisualRelation,
    VisualRenderMode,
    VisualScene,
    VisualSpec,
    VisualVector,
)

BASE = os.path.dirname(__file__)


def load_ground_truth(name: str) -> dict:
    with open(os.path.join(BASE, f"ground_truth_{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def argand_spec_from_ground_truth() -> VisualSpec:
    g = load_ground_truth("argand")
    pts = g["points"]
    order = g["square"]["vertices"]
    poly = [[pts[k]["re"], pts[k]["im"]] for k in order] + [[pts["A"]["re"], pts["A"]["im"]]]
    a = pts["A"]
    return VisualSpec(
        concept=g["concept"],
        render_mode=VisualRenderMode.DETERMINISTIC,
        text_required=True,
        deterministic=DeterministicVisual(
            title="Square on Argand Plane",
            scene=VisualScene(
                scene_kind="plot",
                caption="A square ABCD on the Argand plane; points map complex numbers to coordinates.",
                plot=VisualPlot(
                    x_label="Re", y_label="Im",
                    x_min=-1, x_max=5, y_min=-1, y_max=5, show_grid=True,
                    curves=[
                        VisualCurve(label="square ABCD", points=poly, style="solid", color="accent"),
                        VisualCurve(label=f"A({a['re']},{a['im']})", points=[[a["re"], a["im"]], [a["re"], a["im"]]], style="solid", color=""),
                    ],
                ),
                generic=VisualGeneric(
                    central_label="Square ABCD",
                    callouts=[
                        f"A: z = {a['z']} → ({a['re']},{a['im']})",
                        f"B: {pts['B']['z']} → ({pts['B']['re']},{pts['B']['im']})",
                        "Side s = |B-A| = 2",
                        "Area = s² = 4",
                    ],
                ),
            ),
            equations=[
                VisualEquation(expression="z = 1 + i", meaning="point A"),
                VisualEquation(expression="Area = 4", meaning="square area"),
            ],
            steps=["Map z to (Re,Im)", "Reflect conjugate", "Compute side s=2"],
            points=["Reflection across Re axis", "Modulus relation"],
        ),
        visual_form="Argand square with coordinates and derivation",
        key_elements=["square ABCD", "A(1,1)", "side s = 2", "Area = 4"],
        key_relationships=["z → (Re,Im)", "conjugate = reflection", "side from distance"],
        must_show=["square", "A/B/C/D", "coordinates", "side length", "result"],
        avoid=["decorative 3D", "photo-realism"],
    )


def torque_spec_from_ground_truth() -> VisualSpec:
    g = load_ground_truth("torque")
    vecs = [
        VisualVector(
            label=v["label"], angle_deg=v["angle_deg"], length=v["length"],
            tail=v.get("tail", ""), color=v.get("color", ""),
        )
        for v in g["vectors"]
    ]
    return VisualSpec(
        concept=g["concept"],
        render_mode=VisualRenderMode.DETERMINISTIC,
        text_required=True,
        deterministic=DeterministicVisual(
            title="Torque and Angular Momentum",
            scene=VisualScene(
                scene_kind="force_diagram",
                caption="Torque magnitude depends on r, F and the angle between them.",
                force=ForceDiagram(
                    object=VisualObject(kind=g["pivot"]["kind"], label=g["pivot"]["label"]),
                    vectors=vecs,
                    angles=[VisualAngle(label=g["angle"]["label"], between=g["angle"]["between"], caption="angle between r and F")],
                    arcs=[VisualArc(label=g["arc"]["label"], around=g["arc"]["around"], direction=g["arc"]["direction"], caption="torque direction")],
                    relation=VisualRelation(expression=g["relationships"][0]["expression"], caption="torque = r cross F"),
                ),
            ),
            equations=[
                VisualEquation(expression="τ = r × F", meaning="torque"),
                VisualEquation(expression="L = Iω", meaning="angular momentum"),
            ],
            steps=["Identify pivot", "Draw r and F", "Angle θ between them"],
            points=["Torque drives angular momentum"],
        ),
        visual_form="force vector diagram",
        key_elements=["pivot O", "r vector", "F vector", "θ", "τ"],
        key_relationships=["τ = r × F", "θ between r and F"],
        must_show=["pivot", "r", "F", "θ", "τ"],
        avoid=["photo-realism"],
    )


def torque_v3_spec_from_ground_truth() -> VisualSpec:
    """v3: same geometry truth as v2 + explicit composition (no inference)."""
    spec = torque_spec_from_ground_truth()
    spec.deterministic.composition = LessonComposition(
        title="Torque and Angular Momentum",
        framing="Pivot → r → F → θ → τ — the turning effect",
        callouts=[
            CompositionCallout(id="obj-pivot", label="O", value="pivot"),
            CompositionCallout(id="vec-r", label="r", value="55° lever"),
            CompositionCallout(id="vec-f", label="F", value="90° force"),
            CompositionCallout(id="ang-theta", label="θ = 35°", value="between r, F"),
        ],
        reasoning=[
            CompositionReasoningStep(id="rs-theta", expression="θ = 90° − 55° = 35°", explanation="angle between r and F"),
            CompositionReasoningStep(id="rs-tau", expression="τ = r × F", explanation="magnitude rF sin(θ)"),
            CompositionReasoningStep(id="rs-dir", expression="right-hand rule", explanation="direction out of page"),
        ],
        result=CompositionResult(expression="τ = r × F", emphasis=True),
        takeaway="A force far from the pivot (large r) with θ near 90° gives maximal torque — like pushing a door at the handle.",
    )
    return spec


def argand_v3_spec_from_ground_truth() -> VisualSpec:
    """v3: same geometry truth as v2 + explicit composition (no inference)."""
    spec = argand_spec_from_ground_truth()
    pts = load_ground_truth("argand")["points"]
    spec.deterministic.composition = LessonComposition(
        title="Square on Argand Plane",
        framing="Each complex number is a point: z = x + iy → (x, y)",
        callouts=[
            CompositionCallout(id="pt-a", label="A", value=f"z = {pts['A']['z']} ({pts['A']['re']},{pts['A']['im']})"),
            CompositionCallout(id="pt-b", label="B", value=f"z = {pts['B']['z']} ({pts['B']['re']},{pts['B']['im']})"),
            CompositionCallout(id="pt-c", label="C", value=f"z = {pts['C']['z']} ({pts['C']['re']},{pts['C']['im']})"),
            CompositionCallout(id="pt-d", label="D", value=f"z = {pts['D']['z']} ({pts['D']['re']},{pts['D']['im']})"),
        ],
        reasoning=[
            CompositionReasoningStep(id="rs-conj", expression="conj(A) = 1 − i", explanation="mirrors across Re → reflection"),
            CompositionReasoningStep(id="rs-side", expression="s = |B − A| = |2i| = 2", explanation="distance formula"),
            CompositionReasoningStep(id="rs-mod", expression="|1 + i| = √2", explanation="modulus"),
        ],
        result=CompositionResult(expression="Area = 4", emphasis=True),
        takeaway="A square's area is side² — here side s = |B − A| = 2, so Area = 4.",
    )
    return spec
