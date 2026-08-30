"""Shared semantic fixtures — one source for all 3 prototype engines."""
from app.models.schemas import (
    DeterministicVisual,
    VisualScene,
    VisualPlot,
    VisualCurve,
    VisualGeneric,
    ForceDiagram,
    VisualObject,
    VisualVector,
    VisualAngle,
    VisualArc,
    VisualRelation,
    VisualSpec,
    VisualRenderMode,
)

# Visual 1: JEE Argand-plane square
# Concept: z, conjugate, square ABCD, side length, modulus, area
from app.models.schemas import VisualEquation

ARGAND_SPEC = VisualSpec(
    concept="Argand plane square",
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
                    VisualCurve(label="square ABCD", points=[[1,1],[1,3],[3,3],[3,1],[1,1]], style="solid", color="accent"),
                    VisualCurve(label="A(1,1)", points=[[1,1],[1,1]], style="solid", color=""),
                ]
            ),
            generic=VisualGeneric(
                central_label="Square ABCD",
                callouts=[
                    "A: z = 1 + i → (1,1)",
                    "B: -1 + 3i → (-1,3) reflected",
                    "Side s = √[(2)²] = 2",
                    "Area = s² = 4"
                ]
            )
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

# Visual 2: Torque
TORQUE_SPEC = VisualSpec(
    concept="Torque",
    render_mode=VisualRenderMode.DETERMINISTIC,
    text_required=True,
    deterministic=DeterministicVisual(
        title="Torque and Angular Momentum",
        scene=VisualScene(
            scene_kind="force_diagram",
            caption="Torque magnitude depends on r, F and the angle between them.",
            force=ForceDiagram(
                object=VisualObject(kind="pivot", label="O"),
                vectors=[
                    VisualVector(label="r", angle_deg=55, length=1.1, color="accent"),
                    VisualVector(label="F", angle_deg=90, length=0.8, color="red", tail="r"),
                ],
                angles=[VisualAngle(label="θ", between=["r", "F"], caption="angle between r and F")],
                arcs=[VisualArc(label="τ", around="O", direction="ccw", caption="torque direction")],
                relation=VisualRelation(expression="τ = r × F", caption="torque = r cross F"),
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
