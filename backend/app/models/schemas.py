from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExtractionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    DIAGRAM = "diagram"


class TopicInfo(BaseModel):
    title: str = ""
    is_probable: bool = False


class FormulaEntry(BaseModel):
    formula: str = ""
    explanation: str = ""
    uncertain_symbols: list[str] = []
    confidence: str = "clear"  # "clear" | "context_needed" | "possible_extraction_issue"


class VisualContext(BaseModel):
    present: bool = False
    summary: str = ""


class DiagramRep(BaseModel):
    present: bool = False
    svg: str = ""
    best_effort: bool = False  # True only for legacy-fallback SVG; presented as unverified, not "clean"


class DiagramType(str, Enum):
    POLAR_REGION = "polar_region"


class PolarBounds(BaseModel):
    inner: str = ""
    outer: str = ""
    theta_min: str = ""
    theta_max: str = ""


class DiagramSpec(BaseModel):
    """Semantic description of a diagram, produced by Gemini. Never pixel/SVG geometry.

    Only polar_region is rendered today; the renderer is deterministic, so the
    same validated spec always produces the same SVG. This is an internal
    intermediate — it never appears on the wire (StudyNotes excludes it).
    """

    present: bool = False
    diagram_type: str = ""
    bounds: PolarBounds = PolarBounds()
    show_axes: bool = True
    labels: list[str] = []
    shade_region: bool = True
    instruction_text: list[str] = []
    uncertain: list[str] = []


class StudyNotes(BaseModel):
    topic: TopicInfo = TopicInfo()
    what_you_should_remember: str = ""
    key_formulas: list[FormulaEntry] = []
    understand_it: list[str] = []
    common_mistakes: list[str] = []
    thirty_second_revision: list[str] = []
    visual_context: VisualContext = VisualContext()
    diagram: DiagramRep = DiagramRep()
    diagram_spec: Optional[DiagramSpec] = Field(default=None, exclude=True)
    verify_before_studying: list[str] = []
    uncertainties: list[str] = []
    analogy: str = ""


class ExtractionResponse(BaseModel):
    type: ExtractionType
    markdown: str
    imageUrl: Optional[str] = None
    tags: list[str] = []
    creditsUsed: int
    studyNotes: Optional[StudyNotes] = None
    diagramId: Optional[str] = None


class ExtractionContext(BaseModel):
    title: str = ""
    url: str = ""
    week: str = ""


class UserCredits(BaseModel):
    creditsRemaining: int
    creditsUsed: int
    plan: str = "free"


class GoogleAuthRequest(BaseModel):
    idToken: str


class AuthResponse(BaseModel):
    accessToken: str
    uid: str
    email: str
    name: str
    creditsRemaining: int


class DeviceAuthRequest(BaseModel):
    deviceId: str


class DeviceAuthResponse(BaseModel):
    deviceId: str
    creditsRemaining: int
    creditsUsed: int
    plan: str = "free"


class RevisionResponse(BaseModel):
    study_notes: StudyNotes
    creditsUsed: int


class DiagramResult(BaseModel):
    markdown: str


class VisualRenderMode(str, Enum):
    DETERMINISTIC = "deterministic"
    GENERATIVE = "generative"


class VisualEquation(BaseModel):
    expression: str = ""
    meaning: str = ""


class VisualObject(BaseModel):
    kind: Literal["pivot", "disk", "point", "block"] = "pivot"
    label: str = ""
    caption: str = ""


class VisualVector(BaseModel):
    label: str = ""
    angle_deg: int = 0
    length: float = 1.0
    tail: str = ""
    color: str = ""
    caption: str = ""


class VisualAngle(BaseModel):
    label: str = "θ"
    between: list[str] = []
    caption: str = ""


class VisualArc(BaseModel):
    label: str = ""
    around: str = ""
    direction: Literal["ccw", "cw"] = "ccw"
    caption: str = ""


class VisualRelation(BaseModel):
    expression: str = ""
    caption: str = ""


class ForceDiagram(BaseModel):
    object: VisualObject = VisualObject()
    vectors: list[VisualVector] = []
    angles: list[VisualAngle] = []
    arcs: list[VisualArc] = []
    relation: VisualRelation | None = None


class FlowNode(BaseModel):
    label: str = ""


class FlowConnector(BaseModel):
    source: int = 0
    target: int = 1
    label: str = ""
    feedback: bool = False


class ProcessFlow(BaseModel):
    nodes: list[FlowNode] = []
    connectors: list[FlowConnector] = []
    relation: VisualRelation | None = None


class VisualCurve(BaseModel):
    label: str = ""
    expr: str = ""  # safe math in x, e.g. "x**2", "sin(x)", "sqrt(x)"
    points: list[list[float]] = Field(default_factory=list)  # explicit [[x,y],...] alternative to expr
    style: Literal["solid", "dashed"] = "solid"
    color: str = ""  # ""|accent|red|green
    x_min: float = 0.0
    x_max: float = 5.0


class VisualPlot(BaseModel):
    x_label: str = "x"
    y_label: str = "y"
    x_min: float = 0.0
    x_max: float = 5.0
    y_min: float = 0.0
    y_max: float = 5.0
    show_grid: bool = True
    curves: list[VisualCurve] = Field(default_factory=list)


class VisualScene(BaseModel):
    scene_kind: Literal["force_diagram", "process_flow", "plot"] = "force_diagram"
    title: str = ""
    caption: str = ""
    force: ForceDiagram | None = None
    flow: ProcessFlow | None = None
    plot: VisualPlot | None = None


class DeterministicVisual(BaseModel):
    """Exact, bounded content payload for the deterministic renderer.

    Carries the information that MUST be rendered exactly. Two supported
    presentations:
      - a `scene` (universal educational primitives: objects, vectors, angles,
        arcs, relations, process boxes, connectors) rendered as a real diagram,
      - or the classic study card (equations + meanings, ordered steps, points).

    Code owns ALL layout/geometry; Gemini only supplies semantics (labels,
    angles in degrees, relative lengths, relationships). Never pixel
    coordinates.
    """

    title: str = ""
    scene: VisualScene | None = None
    equations: list[VisualEquation] = []
    steps: list[str] = []
    points: list[str] = []


class VisualSpec(BaseModel):
    """Structured, grounded description of the educational visual to generate.

    Gemini produces this from the screenshot + StudyNotes. It is a semantic
    spec (concept, form, elements, relationships) — never pixel geometry.

    render_mode picks the rendering path:
      - "deterministic": exact text/symbols matter, so code renders a clean
        SVG (visual_renderer) — no generative model involved.
      - "generative": exact typography is NOT the payload, so Pollinations may
        draw a conceptual illustration, gated by an OCR legibility check when
        text_required is true.
    """

    concept: str = ""
    render_mode: VisualRenderMode = VisualRenderMode.DETERMINISTIC
    text_required: bool = True
    deterministic: DeterministicVisual = DeterministicVisual()
    visual_form: str = ""
    key_elements: list[str] = []
    key_relationships: list[str] = []
    must_show: list[str] = []
    avoid: list[str] = []


class VisualExplanationResponse(BaseModel):
    diagramId: str
    renderMode: str = "deterministic"  # "deterministic" | "generative"
    imageUrl: Optional[str] = None
    imageSvg: Optional[str] = None
    status: str = "generated"  # "generated" | "already_generated"
