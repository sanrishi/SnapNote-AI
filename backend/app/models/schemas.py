from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class JEEPyQRef(BaseModel):
    """One verified PYQ record, exactly as stored. No computed statistics."""

    question_id: str = ""
    exam: str = ""
    year: int = 0
    session: Optional[str] = None
    shift: Optional[str] = None
    paper_question_number: int = 0
    question_type: str = ""
    marks: Optional[int] = None
    question_summary: Optional[str] = None
    source_url: str = ""
    source_document: str = ""
    source_class: str = "THIRD_PARTY_TRANSCRIPTION"
    answer_key: Optional[str] = None
    answer_status: str = "unconfirmed"  # "confirmed" | "unconfirmed"


class JEEConceptRef(BaseModel):
    concept_id: str = ""
    name: str = ""


class JEEPatternRef(BaseModel):
    """One evidence-backed question pattern. Every pattern references only
    verified question IDs that exist in the corpus; the UI must show the
    coverage note instead of implying exhaustive discovery."""

    pattern_id: str = ""
    name: str = ""
    sub_concept: str = ""
    method: str = ""
    question_ids: list[str] = []
    coverage_note: str = ""


class JEECandidateResult(BaseModel):
    """One plausible candidate's own question group inside an ambiguous result.

    Never a confirmed match: pyqs are related records from the candidate's
    chapter/concept, explicitly labelled as such. Empty pyqs with an honest
    note when the candidate has no stored records.
    """

    concept_id: str = ""
    name: str = ""
    subject: str = ""
    chapter: str = ""
    match_granularity: str = ""
    pyqs: list[JEEPyQRef] = []
    pyq_count: int = 0
    pyqs_truncated: bool = False
    coverage_note: str = ""


class JEESyllabusRef(BaseModel):
    node_id: str = ""
    subtopic: str = ""
    source_url: str = ""


class JEEConceptContext(BaseModel):
    """Student-facing JEE context for one lecture concept.

    Every factual field traces to the stored syllabus/taxonomy/PYQ corpora.
    match_status is 'unresolved' (nothing matched) or 'ambiguous' (tie)
    instead of a forced guess — the UI must say less, not invent more.
    """

    concept_id: Optional[str] = None
    canonical_name: Optional[str] = None
    subject: str = ""
    chapter: str = ""
    subtopic: Optional[str] = None
    syllabus: Optional[JEESyllabusRef] = None
    match_status: str = "unresolved"  # "matched" | "ambiguous" | "unresolved"
    evidence: list[str] = []
    candidates: list[JEEConceptRef] = []  # tied concepts when ambiguous
    candidate_results: list[JEECandidateResult] = []  # per-candidate PYQ groups when ambiguous
    prerequisites: list[JEEConceptRef] = []
    related: list[JEEConceptRef] = []
    pyqs: list[JEEPyQRef] = []
    pyq_count: int = 0
    pyqs_truncated: bool = False
    patterns: list[JEEPatternRef] = []
    coverage_note: str = ""
    match_granularity: str = ""  # "sub-concept" | "chapter" | ""


class JEEMapRequest(BaseModel):
    topic_title: str = ""
    key_terms: list[str] = []


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
    jee: Optional[JEEConceptContext] = None


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
    picture: str = ""
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
    jee: Optional[JEEConceptContext] = None


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
    # Comparison panels only: where the vector is anchored on the object.
    # `tail` stays the force-diagram chaining mechanism; `at` is a physical
    # point on the panel body (top = P', bottom = P, center = CM). A zero
    # length is a real value: it renders a point marker + label, never an
    # arrow (pure rotation at the CM, rolling at the contact point).
    at: Literal["center", "top", "bottom", "left", "right"] = "center"
    color: str = ""
    caption: str = ""


class ComparisonPoint(BaseModel):
    """A labeled material point on a comparison panel body."""

    label: str = ""
    at: Literal["center", "top", "bottom", "left", "right"] = "center"


class VisualAngle(BaseModel):
    label: str = "θ"
    between: list[str] = []
    caption: str = ""


class VisualArc(BaseModel):
    label: str = ""
    around: str = ""
    direction: Literal["ccw", "cw"] = "ccw"
    caption: str = ""

    @field_validator("around", mode="before")
    @classmethod
    def _coerce_point_around(cls, value: object) -> object:
        """Repair emissions sometimes give the arc centre as coordinates
        (e.g. [0.0, 0.0]) instead of the wrapped object's label. Coordinates
        for a rotation arc always mean the pivot, which is the renderer's
        default when `around` is empty — so an empty label reproduces the
        intended geometry exactly; only the redundant coordinate is dropped.
        """
        if isinstance(value, str) or value is None:
            return value or ""
        return ""


class VisualRelation(BaseModel):
    expression: str = ""
    caption: str = ""


class ForceDiagram(BaseModel):
    object: VisualObject = VisualObject()
    vectors: list[VisualVector] = []
    angles: list[VisualAngle] = []
    arcs: list[VisualArc] = []
    relation: VisualRelation | None = None
    # True when the object rolls/rests on a surface: the renderer draws the
    # ground/contact line under it. Rolling and friction contexts need the
    # contact line present — without it the diagram reads as floating.
    ground: bool = False


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


class PlotMarker(BaseModel):
    """One labeled point on a plot (focal point, vertex, object base, ...).

    Semantics only: coordinates in data space. The renderer owns pixels,
    halo placement and collision avoidance.
    """

    x: float = 0.0
    y: float = 0.0
    label: str = ""
    color: str = ""  # ""|accent|red|green


class PlotSegment(BaseModel):
    """One straight labeled segment on a plot (lens, object/image arrow, ...).

    With ``arrow`` the renderer draws an arrowhead at (x2, y2) — code-owned
    geometry, so arrowheads are never spec-supplied SVG.
    """

    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    label: str = ""
    color: str = ""  # ""|accent|red|green
    style: Literal["solid", "dashed"] = "solid"
    arrow: bool = False


class VisualPlot(BaseModel):
    x_label: str = "x"
    y_label: str = "y"
    x_min: float = 0.0
    x_max: float = 5.0
    y_min: float = 0.0
    y_max: float = 5.0
    show_grid: bool = True
    curves: list[VisualCurve] = Field(default_factory=list)
    markers: list[PlotMarker] = Field(default_factory=list)
    segments: list[PlotSegment] = Field(default_factory=list)


class VisualGeneric(BaseModel):
    central_label: str = ""
    callouts: list[str] = Field(default_factory=list)  # 2-5 short labels

    @field_validator("callouts", mode="before")
    @classmethod
    def _coerce_callout_dicts(cls, value: object) -> object:
        """Tolerate repair emitting {id,label,value} objects (observed
        case 8: three claim dicts hard-failed pydantic, turning a repairable
        spec into a 502). Structural coercion only — the printed text is the
        prefixed label/value pair, nothing invented or dropped."""
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        coerced: list[object] = []
        for item in value:
            if isinstance(item, dict):
                label = str(item.get("label") or "").strip()
                body = item.get("value") or item.get("text") or item.get("claim") or ""
                body = str(body).strip()
                if label and body:
                    coerced.append(f"{label}: {body}")
                else:
                    coerced.append(body or label)
            else:
                coerced.append(item)
        return coerced


class MapNode(BaseModel):
    """One semantic node of a concept map.

    The planner supplies meaning only: a stable id, a short label, and (for
    tree layouts) the id of the parent node. The renderer owns every pixel,
    spacing decision and connector. ``detail`` is a short teaching note that
    the composition layer may print; it is never used to invent geometry.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", max_length=40)
    label: str = Field(default="", max_length=48)
    detail: str = Field(default="", max_length=90)
    parent: str = Field(default="", max_length=40)
    marker: str = Field(default="", max_length=18)  # timeline date/phase marker
    color: str = ""  # ""|accent|red|green|amber|violet


class MapEdge(BaseModel):
    """One labeled relationship/cross-link between map nodes."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="", max_length=40)
    target: str = Field(default="", max_length=40)
    label: str = Field(default="", max_length=32)


class VisualConceptMap(BaseModel):
    """A real relational grammar for non-geometric subjects.

    ``tree`` renders a parent/child hierarchy (classification, taxonomies,
    tree data structures, whole/part relationships). ``cycle`` renders
    repeating stages in a ring (biological/chemical cycles, feedback loops).
    ``timeline`` renders ordered chronology (history, evolution, phases).
    Layout remains code-owned; the spec only declares semantic relationships.
    """

    model_config = ConfigDict(extra="forbid")

    layout: Literal["tree", "cycle", "timeline"] = "tree"
    nodes: list[MapNode] = Field(default_factory=list, max_length=12)
    edges: list[MapEdge] = Field(default_factory=list, max_length=12)
    relation: VisualRelation | None = None


class ComparisonColumn(BaseModel):
    """One alternative in a feature-comparison table."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", max_length=40)
    values: list[str] = Field(default_factory=list, max_length=6)


class ComparisonRow(BaseModel):
    """One shared feature/dimension compared across columns."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", max_length=40)


class ComparisonTable(BaseModel):
    """Feature matrix for conceptual subjects that are not motion diagrams.

    Used for chemistry (SN1 vs SN2), biology (prokaryote vs eukaryote),
    economics (market structures), CS (array vs linked list), etc. Every cell
    is exact planner text; the renderer only decides geometry and typography.
    """

    model_config = ConfigDict(extra="forbid")

    rows: list[ComparisonRow] = Field(default_factory=list, max_length=6)
    columns: list[ComparisonColumn] = Field(default_factory=list, max_length=3)
    # Top-left header. Rows can be features (default) or alternatives when a
    # wide source is transposed, so the axis name must come from the planner.
    corner: str = Field(default="FEATURE", max_length=20)
    verdict: str = Field(default="", max_length=120)


class ComparisonPanel(BaseModel):
    """One panel of a side-by-side comparison (2-3 panels per scene).

    Same vocabulary as force_diagram, scoped to the panel: an object drawn
    at the panel centre, labeled vectors chained by tail references, and
    rotation arcs around the panel object. Geometry stays code-owned; the
    spec only names which primitives appear.
    """

    label: str = ""
    object: VisualObject = VisualObject()
    points: list[ComparisonPoint] = []
    vectors: list[VisualVector] = []
    arcs: list[VisualArc] = []
    note: str = ""  # one short caption line under the panel


class VisualComparison(BaseModel):
    panels: list[ComparisonPanel] = []  # 2-3 motion panels
    table: ComparisonTable | None = None  # feature matrix alternative
    relation: VisualRelation | None = None


class VisualInterval(BaseModel):
    """One real interval of a number-line track. Endpoints are exact text.

    lo/hi are "-inf"/"inf" or integer/fraction/decimal text ("-3", "1/2",
    "0.30000000000000004"). The bound is wide enough for exact decimal
    endpoints (never truncated to floats).
    """

    model_config = ConfigDict(extra="forbid")

    lo: str = Field(default="-inf", max_length=32)
    hi: str = Field(default="inf", max_length=32)
    lo_closed: bool = False
    hi_closed: bool = False


class VisualNumberLineTrack(BaseModel):
    """One labeled track: a solution/domain set drawn on a number line.

    `points` are values EXCLUDED from the set (e.g. domain exclusions like
    x ≠ -2) — they are removed from the interval union and marked hollow.
    Bounds cover the verifier's worst case (domains split by several
    excluded values), so solver output is never truncated.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", max_length=60)
    intervals: list[VisualInterval] = Field(default_factory=list, max_length=12)
    points: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_point_objects(cls, value: object) -> object:
        """Accept the observed {"val": "-1", "open": true} point shape.

        The repair leg occasionally emits excluded points as objects instead
        of strings, which crashed repair with a pydantic error and turned an
        otherwise recoverable number_line spec into a refusal. "open" carries
        no extra information here (every listed point is an exclusion drawn
        hollow), so only the exact value text is kept — never rounded, never
        re-derived.
        """
        if not isinstance(value, list):
            return value
        out: list[object] = []
        for item in value:
            if isinstance(item, dict) and "val" in item:
                out.append(str(item["val"]))
            else:
                out.append(item)
        return out


class VisualMathCase(BaseModel):
    """One case branch of a worked inequality solution (e.g. base > 1)."""

    model_config = ConfigDict(extra="forbid")

    condition: str = Field(default="", max_length=80)
    transformed: str = Field(default="", max_length=160)
    solution: str = Field(default="", max_length=160)


class VisualNumberLine(BaseModel):
    """Worked interval mathematics: problem, domain, case split, final answer.

    The math verifier (app.utils.math_verify) re-solves `problem` exactly and
    refuses any spec whose claimed domain, case solutions, or final solution
    interval set disagree with the computed ones. The renderer draws the
    verified structure; it never re-derives anything.
    """

    model_config = ConfigDict(extra="forbid")

    problem: str = Field(default="", max_length=200)
    domain: VisualNumberLineTrack | None = None
    cases: list[VisualMathCase] = Field(default_factory=list, max_length=3)
    solution: VisualNumberLineTrack = Field(default_factory=VisualNumberLineTrack)


class VisualScene(BaseModel):
    scene_kind: Literal["force_diagram", "process_flow", "plot", "generic", "comparison", "concept_map", "number_line"] = "force_diagram"
    title: str = ""
    caption: str = ""
    force: ForceDiagram | None = None
    flow: ProcessFlow | None = None
    plot: VisualPlot | None = None
    generic: VisualGeneric | None = None
    comparison: VisualComparison | None = None
    concept_map: VisualConceptMap | None = None
    number_line: VisualNumberLine | None = None


class CompositionCallout(BaseModel):
    """One lesson chip: stable anchor id + short label + value. No pixels."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", pattern=r"^[a-z0-9_-]{1,32}$")
    label: str = Field(default="", max_length=24)
    value: str = Field(default="", max_length=80)


class CompositionReasoningStep(BaseModel):
    """One reasoning row: exact expression + short explanation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", pattern=r"^[a-z0-9_-]{1,32}$")
    expression: str = Field(default="", max_length=120)
    explanation: str = Field(default="", max_length=200)


class CompositionResult(BaseModel):
    """Highlighted final result. Expression passes through byte-identical."""

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(default="", max_length=120)
    emphasis: bool = True


class LessonComposition(BaseModel):
    """Explicit educational composition (VisualSpec v3).

    Optional block: absent (None) = v2 derived behavior exactly. Present =
    explicit lesson preferred over derivation. Values originate from validated
    semantic data and pass through byte-identical; composition never invents
    geometry or facts.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=80)
    framing: str = Field(default="", max_length=200)
    callouts: list[CompositionCallout] = Field(default_factory=list, max_length=6)
    reasoning: list[CompositionReasoningStep] = Field(default_factory=list, max_length=4)
    result: CompositionResult | None = None
    takeaway: str = Field(default="", max_length=200)


class VisualTask(BaseModel):
    """Educational task plan: WHAT the visual must teach, before HOW.

    The planner (LLM, untrusted) declares the teaching job; the validator
    enforces that the emitted scene can actually do it. Task determines
    representation: `acceptable_families` must contain the scene's kind, and
    every `required_elements` entry must be satisfiable by that family's
    primitives. `exact_claims` are strings that must appear verbatim in the
    rendered output (equations, values, labels) — the renderer's text is
    checked against them, never the reverse.
    """

    task_type: Literal[
        "compare_variants",
        "multi_step_solution",
        "graph_relationship",
        "vector_arrangement",
        "connected_process",
        "coordinate_construction",
        "definition_overview",
        "interval_solution",
        "hierarchy_structure",
        "cyclic_process",
        "chronology",
    ] = "definition_overview"
    learning_outcome: str = ""
    required_elements: list[str] = []
    required_relationships: list[str] = []
    exact_claims: list[str] = []
    acceptable_families: list[str] = []
    assumptions: list[str] = []
    missing: list[str] = []

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

    v3: optional `composition` carries the explicit educational composition.
    Absent = v2 derived behavior (unchanged).
    """

    title: str = ""
    task: VisualTask | None = None
    scene: VisualScene | None = None
    equations: list[VisualEquation] = []
    steps: list[str] = []
    points: list[str] = []
    composition: LessonComposition | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_card_items(cls, data: object) -> object:
        """Tolerate near-miss shapes for the card lists before strict parse.

        Observed LLM emission (thin-scene repair): "equations": ["1/v - 1/u
        = 1/f"] — bare strings instead of {"expression", "meaning"} objects,
        which hard-failed pydantic and turned a repairable spec into a 502.
        Same class of shape drift for steps/points items. Coercion is
        structural only: no value is altered, invented, or dropped.
        """
        if not isinstance(data, dict):
            return data
        equations = data.get("equations")
        if isinstance(equations, list):
            data["equations"] = [
                {"expression": item, "meaning": ""} if isinstance(item, str) else item
                for item in equations
            ]
        for key in ("steps", "points"):
            values = data.get(key)
            if isinstance(values, list):
                data[key] = [
                    item if isinstance(item, str) else ("" if item is None else str(item))
                    for item in values
                ]
        return data



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
    # Trust layer: study-notes context summary re-grounded against the emitted
    # scene (None = keep the diagram-step text / hide when empty).
    visualContext: Optional[str] = None
