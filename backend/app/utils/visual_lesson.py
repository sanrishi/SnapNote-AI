"""Production composition contract: VisualSpec -> educational visual lesson.

Pipeline (additive — the old path `visual_service.generate_visual` is untouched
and remains the fallback until this path is validated):

    VisualSpec
      -> validate_lesson_spec      (semantic validity, exact values preserved)
      -> select_family             (vector | coordinate | flow | generic)
      -> render_hero_geometry      (geometry + geometry labels ONLY, no prose)
      -> extract_lesson_content    (structured prose: title/callouts/reasoning/result)
      -> build_lesson_typst        (Typst owns 100% of layout + prose)
      -> compile_typst             (official CLI; deterministic output)

Contract: ``render_visual_lesson(spec) -> RenderedLesson``.

Rules enforced here, not by convention:
- Gemini/LLM output is NEVER executed and NEVER trusted with pixels.
- The hero SVG carries zero prose (no title/caption/relation cards).
- All numeric values pass through byte-identical (no rounding, no rewriting).
- If Typst is unavailable, fall back to the bare hero SVG (honest, never fake).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.models.schemas import (
    ComparisonTable,
    CompositionCallout,
    DeterministicVisual,
    MapNode,
    StudyNotes,
    VisualComparison,
    VisualConceptMap,
    VisualRenderMode,
    VisualScene,
    VisualSpec,
    VisualTask,
)
from app.utils.visual_renderer import (
    GENERIC_SATELLITE_CAP,
    concept_map_is_horizontal,
    render_hero_geometry,
    rendered_scene_text,
)
from app.utils.table_quality import (
    normalize_visual_spec,
    table_quality_errors,
)

logger = logging.getLogger(__name__)

# scene_kind -> visual family. New families plug in here; no per-subject code.
SUPPORTED_FAMILIES: dict[str, str] = {
    "force_diagram": "vector",
    "plot": "coordinate",
    "process_flow": "flow",
    "generic": "generic",
    "comparison": "comparison",
    "concept_map": "concept_map",
    "number_line": "worked_math",
}

# Capability registry: what each family can faithfully represent, and what it
# must refuse. Family selection is checked against this AND the task plan —
# both must agree, so neither prompt gaming nor a stale default can silently
# degrade a complex lesson into a drawable-but-empty figure.
FAMILY_CAPABILITIES: dict[str, dict[str, object]] = {
    "force_diagram": {
        "represents": ["bodies", "pivots", "anchored vectors", "angles", "rotation arcs", "dimensions"],
        "requires": "at least 1 labeled vector",
        "cannot": ["plotted functions", "process steps", "multi-case comparisons", "circuits"],
    },
    "plot": {
        "represents": ["axes", "curves", "labeled points", "polygons", "ticks", "dimensions"],
        "requires": "at least 1 curve",
        "cannot": ["force vectors", "process steps", "circuits", "chemical structures"],
    },
    "process_flow": {
        "represents": ["ordered steps", "labeled nodes", "connectors", "feedback loops"],
        "requires": "at least 2 labeled nodes",
        "cannot": ["geometry", "vectors", "plotted functions", "circuits"],
    },
    "generic": {
        "represents": ["central idea", "short callouts"],
        "requires": "central label + at least 3 callouts",
        "cannot": ["vectors", "curves", "points", "processes", "comparisons", "dimensions"],
    },
    "comparison": {
        "represents": [
            "2-3 labeled variant panels with own objects/vectors/arcs",
            "2-3 alternative feature columns compared row by row",
        ],
        "requires": "at least 2 labeled panels with geometry, or a 2-3 column feature table",
        "cannot": ["plotted functions", "process steps", "circuits"],
    },
    "concept_map": {
        "represents": [
            "hierarchies and classifications (parent/child)",
            "cyclic processes (ordered repeating stages)",
            "chronologies and timelines (ordered events)",
            "labeled cross-relationships between concepts",
        ],
        "requires": "3-12 labeled nodes with valid relationships",
        "cannot": [
            "quantitative curves", "force vectors", "worked interval mathematics",
            "precise molecular geometry",
        ],
    },
    "number_line": {
        "represents": ["interval sets on number lines", "open/closed endpoints", "excluded values", "domain constraints", "case branches", "final solution sets"],
        "requires": "a problem statement whose solution the verifier can compute exactly",
        "cannot": ["vectors", "curves as geometry", "process steps", "unverifiable mathematics (irrational critical points, non-polynomial operands)"],
    },
}

# Educational task -> families capable of expressing it. The task plan names
# acceptable_families; the scene kind must ALSO appear here. Both agree, or
# the spec is refused — this is what stops a multi-case solution silently
# degrading into generic text cards.
TASK_FAMILIES: dict[str, list[str]] = {
    "compare_variants": ["comparison"],
    "multi_step_solution": ["process_flow"],
    "graph_relationship": ["plot"],
    "vector_arrangement": ["force_diagram"],
    "connected_process": ["process_flow"],
    "coordinate_construction": ["plot"],
    "definition_overview": ["force_diagram", "plot", "process_flow", "generic", "comparison", "concept_map"],
    "interval_solution": ["number_line"],
    "hierarchy_structure": ["concept_map"],
    "cyclic_process": ["concept_map"],
    "chronology": ["concept_map"],
}

_TASK_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "under", "between", "such", "then", "than", "also", "must", "show",
    "each", "both", "will",
})


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[^\W_]+", text.casefold()))


def _claim_key(text: str) -> str:
    """Presentation-insensitive key for exact-claim containment.

    Claims must still appear with the same symbols and values — this only
    ignores whitespace, label punctuation (``:``) and the multiplication glyph
    (×, ·, *) which the planner writes inconsistently around the same printed
    expression (observed: claim "f(z) = (1/√(2π)) × exp(-z²/2)" vs rendered
    "f(z) = (1/√(2π))exp(-z²/2)"; claim "Worst-case: O(n)" vs the visible
    "Worst-case" label plus "O(n)" detail). Different numbers, variables,
    powers or operators still fail.
    """
    import re

    return re.sub(r"[\s×·*:]+", "", text).casefold()


def _distinctive(text: str) -> set[str]:
    return {w for w in _tokens(text) if len(w) > 2 and w not in _TASK_STOPWORDS}


def _scene_text(scene: VisualScene, generic_cap: int | None = None) -> str:
    """All human-visible scene strings, for task-link grounding checks.

    ``generic_cap`` truncates generic callouts to the hero's visible
    satellite capacity (default: no truncation, for non-gate callers).
    """
    bits: list[str] = []
    kind = scene.scene_kind
    if kind == "force_diagram" and scene.force is not None:
        force = scene.force
        bits.append((force.object.label or ""))
        for v in force.vectors:
            bits.append(v.label or "")
        for a in force.angles:
            bits.append(a.label or "")
        for a in force.arcs:
            bits.append((a.label or "") + " " + (a.caption or ""))
        if force.relation is not None:
            bits.append((force.relation.expression or "") + " " + (force.relation.caption or ""))
    elif kind == "plot" and scene.plot is not None:
        plot = scene.plot
        bits += [plot.x_label or "", plot.y_label or ""]
        for c in plot.curves:
            bits.append(c.label or "")
        for m in plot.markers:
            bits.append(m.label or "")
        for s in plot.segments:
            bits.append(s.label or "")
    elif kind == "process_flow" and scene.flow is not None:
        for n in scene.flow.nodes:
            bits.append((n.label or "") + " " + (getattr(n, "caption", "") or ""))
        for c in scene.flow.connectors:
            bits.append(c.label or "")
        if scene.flow.relation is not None:
            bits.append(scene.flow.relation.expression or "")
    elif kind == "generic" and scene.generic is not None:
        bits.append(scene.generic.central_label or "")
        callouts = [c for c in (scene.generic.callouts or []) if (c or "").strip()]
        if generic_cap is not None:
            callouts = callouts[:generic_cap]
        bits += callouts
    elif kind == "comparison" and scene.comparison is not None:
        for p in scene.comparison.panels or []:
            bits.append((p.label or "") + " " + (p.note or ""))
            bits.append((p.object.label or ""))
            for point in p.points:
                bits.append(point.label or "")
            for v in p.vectors:
                bits.append(v.label or "")
            for a in p.arcs:
                bits.append((a.label or "") + " " + (a.caption or ""))
        table = scene.comparison.table
        if table is not None:
            bits.append(table.verdict or "")
            for row in table.rows:
                bits.append(row.label or "")
            for column in table.columns:
                bits.append(column.label or "")
                bits += [value or "" for value in column.values]
        if scene.comparison.relation is not None:
            bits.append((scene.comparison.relation.expression or "") + " "
                        + (scene.comparison.relation.caption or ""))
    elif kind == "concept_map" and scene.concept_map is not None:
        cmap = scene.concept_map
        visible_detail_ids = _visible_map_detail_ids(cmap)
        for node in cmap.nodes:
            node_id = (node.id or "").strip()
            detail = node.detail if node_id in visible_detail_ids else ""
            bits.append((node.label or "") + " " + (detail or "") + " " + (node.marker or ""))
        for edge in cmap.edges:
            bits.append(edge.label or "")
        if cmap.relation is not None:
            bits.append((cmap.relation.expression or "") + " " + (cmap.relation.caption or ""))
    elif kind == "number_line" and scene.number_line is not None:
        nl = scene.number_line
        bits.append(nl.problem or "")
        for track in (nl.domain, nl.solution):
            if track is None:
                continue
            bits.append(track.label or "")
            for iv in track.intervals:
                bits.append(f"{iv.lo} {iv.hi}")
            bits += [p or "" for p in track.points]
        for case in nl.cases:
            bits.append(case.condition or "")
            bits.append(case.transformed or "")
            bits.append(case.solution or "")
    bits.append(scene.title or "")
    bits.append(scene.caption or "")
    return " ".join(bits)


def _scene_structural_tokens(scene: VisualScene) -> set[str]:
    """Structural vocabulary: what the geometry IS, independent of labels.

    Lets task requirements name structural facts ("pivot", "vector",
    "axes", "node") without demanding those exact words appear as printed
    labels. Labels stay the authority for names/values; structure covers
    kinds. Keeps the satisfiability check honest without forcing label
    duplication.
    """
    tokens: set[str] = set()
    kind = scene.scene_kind
    if kind == "force_diagram" and scene.force is not None:
        force = scene.force
        tokens.add(str(force.object.kind or "").casefold())
        if force.vectors:
            tokens.add("vector")
            tokens.add("vectors")
        if force.angles:
            tokens.add("angle")
            tokens.add("angles")
        if force.arcs:
            tokens.add("arc")
            tokens.add("arcs")
        if force.relation is not None:
            tokens.add("relation")
            tokens.add("relations")
        if force.ground:
            # The renderer draws the contact baseline itself ("surface"), so
            # "ground"/"surface" requirements are structurally satisfied even
            # though no spec label prints those words.
            tokens.update({"ground", "surface", "floor", "table"})
    elif kind == "plot" and scene.plot is not None:
        tokens.update({"axes", "axis", "curve", "curves", "tick", "ticks"})
        if any(len(c.points or []) == 1 for c in scene.plot.curves):
            tokens.add("point")
            tokens.add("points")
        if scene.plot.markers:
            tokens.update({"marker", "markers", "point", "points"})
        if scene.plot.segments:
            tokens.update({"segment", "segments", "line", "lines"})
    elif kind == "process_flow" and scene.flow is not None:
        tokens.add("node")
        tokens.add("nodes")
        if scene.flow.connectors:
            tokens.add("connector")
            tokens.add("connectors")
        if any(getattr(c, "feedback", False) for c in scene.flow.connectors):
            tokens.add("feedback")
    elif kind == "generic":
        tokens.add("callout")
        tokens.add("callouts")
    elif kind == "comparison" and scene.comparison is not None:
        if scene.comparison.table is not None:
            tokens.update({"table", "tables", "column", "columns", "row", "rows", "comparison"})
        tokens.add("panel")
        tokens.add("panels")
        for p in scene.comparison.panels or []:
            tokens.add(str(p.object.kind or "").casefold())
            if p.points:
                tokens.add("point")
                tokens.add("points")
            if p.vectors:
                tokens.add("vector")
                tokens.add("vectors")
            if p.arcs:
                tokens.add("arc")
                tokens.add("arcs")
    elif kind == "concept_map" and scene.concept_map is not None:
        cmap = scene.concept_map
        tokens.update({"node", "nodes", "concept", "concepts", "map", "relationship", "relationships"})
        tokens.add(cmap.layout)
        if cmap.layout == "tree":
            tokens.update({"tree", "hierarchy", "parent", "child", "children", "branch", "branches"})
        elif cmap.layout == "cycle":
            tokens.update({"cycle", "cyclic", "stage", "stages", "loop", "feedback"})
        elif cmap.layout == "timeline":
            tokens.update({"timeline", "chronology", "event", "events", "date", "dates", "phase", "phases"})
        if cmap.edges:
            tokens.update({"edge", "edges", "link", "links", "connector", "connectors"})
    elif kind == "number_line" and scene.number_line is not None:
        tokens.update(
            {
                "number", "line", "number line", "interval", "intervals",
                "endpoint", "endpoints", "domain", "solution", "case", "cases",
                "excluded", "constraint", "constraints", "set", "union",
            }
        )
    return {t for t in tokens if t}


def _map_detail_chip_pairs(cmap: VisualConceptMap) -> list[tuple[str, str]]:
    """(heading, body) pairs the composer prints as map detail cards.

    Mirrors ``extract_lesson_content`` exactly: horizontal trees and timelines
    print details inside the hero (no chips), tall trees print leaf details as
    cards (deepest first), cycles print every stage detail. Validation must use
    this same list; a detail that never reaches a card or the hero is invisible.
    """
    if cmap.layout == "timeline" or concept_map_is_horizontal(cmap):
        return []
    ordered = _map_nodes_deepest_first(cmap)
    parent_ids = {
        (n.parent or "").strip()
        for n in cmap.nodes
        if (n.parent or "").strip()
    }
    visible_detail_ids = _visible_map_detail_ids(cmap)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in ordered:
        node_id = (node.id or "").strip()
        if node_id not in visible_detail_ids:
            continue
        if cmap.layout == "tree" and node_id in parent_ids:
            continue
        detail = _clean(node.detail)
        label = _clean(node.label)
        if not detail:
            continue
        pair = (label, detail)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
        if len(pairs) >= MAP_DETAIL_CARD_CAP:
            break
    return pairs


def derive_task_from_source(study_notes: StudyNotes | None, scene_kind: str) -> VisualTask | None:
    """Build a task contract from SOURCE content, not from the proposed scene.

    Required elements come from the StudyNotes topic title and formula
    explanations; exact claims come from the stored formula expressions.
    The emitted scene kind only scopes acceptable_families (the representation
    choice stands; content coverage is what gets enforced). Because
    requirements originate outside the scene, a sparse-but-consistent scene
    can no longer satisfy its own contract — missing source requirements
    fail validation honestly. Returns None when the source has no usable
    topic/formula content; the caller then repairs or refuses, never renders
    unvalidated. No LLM calls, no invented content.
    """
    if study_notes is None:
        return None
    topic = (getattr(getattr(study_notes, "topic", None), "title", "") or "").strip()
    formulas = getattr(study_notes, "key_formulas", []) or []
    elements: list[str] = []
    if topic:
        elements.append(topic)
    for formula in formulas[:4]:
        explanation = (getattr(formula, "explanation", "") or "").strip()
        if explanation:
            elements.append(explanation)
    claims: list[str] = []
    for formula in formulas[:4]:
        expression = (getattr(formula, "formula", "") or "").strip()
        if expression:
            claims.append(" ".join(expression.split()))
    if not elements and not claims:
        return None
    kind_tasks = {
        "force_diagram": "vector_arrangement",
        "process_flow": "connected_process",
        "comparison": "compare_variants",
        "concept_map": "hierarchy_structure",
        "generic": "definition_overview",
        "number_line": "interval_solution",
    }
    task_type = kind_tasks.get(scene_kind, "definition_overview")
    if task_type == "definition_overview" and scene_kind == "plot":
        task_type = "graph_relationship"
    return VisualTask(
        task_type=task_type,  # type: ignore[arg-type]
        learning_outcome="",
        required_elements=elements,
        required_relationships=[],
        exact_claims=claims,
        acceptable_families=[scene_kind] if scene_kind else [],
        assumptions=["task derived from StudyNotes source content (model omitted task plan)"],
        missing=[],
    )


def derive_task_from_scene(spec: VisualSpec) -> VisualTask | None:
    """Weak fallback: derive a contract from the scene itself.

    Only for when no StudyNotes exist to ground requirements (corrupt
    entitlement). Documented as weak: it cannot detect source omissions.
    Prefer derive_task_from_source whenever notes are available.
    """
    """Deterministically build a task contract from an emitted scene.

    Used when the model omits the task plan: every successful result must
    still carry an enforceable contract. Derivation is documented inside the
    record (assumptions field) so a derived task is never mistaken for a
    planned one. Returns None when the scene has no drawable content — the
    caller then falls through to repair-or-unavailable, never to silent
    acceptance. No LLM calls, no invented content: elements come from scene
    labels, claims from the scene relation and composition result.
    """
    det = spec.deterministic
    scene = det.scene
    if scene is None:
        return None
    kind = scene.scene_kind
    text = _scene_text(scene)
    labels = [w for w in text.split() if len(w) > 2][:12]
    claims: list[str] = []
    relation = _scene_relation_expression(scene)
    if relation.strip():
        claims.append(" ".join(relation.split()))
    comp = det.composition
    if comp is not None and comp.result is not None:
        result = (comp.result.expression or "").strip()
        if result:
            claims.append(" ".join(result.split()))
    if kind == "force_diagram":
        task_type, families = "vector_arrangement", ["force_diagram"]
    elif kind == "plot":
        has_points = scene.plot is not None and any(
            len(c.points or []) == 1 for c in scene.plot.curves)
        task_type = "coordinate_construction" if has_points else "graph_relationship"
        families = ["plot"]
    elif kind == "process_flow":
        task_type, families = "connected_process", ["process_flow"]
    elif kind == "comparison":
        task_type, families = "compare_variants", ["comparison"]
    elif kind == "concept_map":
        layout = scene.concept_map.layout if scene.concept_map is not None else "tree"
        task_type = {
            "tree": "hierarchy_structure",
            "cycle": "cyclic_process",
            "timeline": "chronology",
        }.get(layout, "hierarchy_structure")
        families = ["concept_map"]
    elif kind == "number_line":
        task_type, families = "interval_solution", ["number_line"]
    else:
        task_type, families = "definition_overview", ["generic"]
    elements = labels if labels else []
    if not elements and not claims:
        return None
    return VisualTask(
        task_type=task_type,  # type: ignore[arg-type]
        learning_outcome="",
        required_elements=elements,
        required_relationships=[],
        exact_claims=claims,
        acceptable_families=families,
        assumptions=["task derived deterministically from emitted scene (model omitted task plan)"],
        missing=[],
    )


def _number_line_claim_matches(claim: str, scene: VisualScene, haystack: str) -> bool:
    """Semantic normalization for number_line exact claims (never weakens math).

    A claim that is an interval set matches when the scene shows the SAME set
    anywhere (canonicalized through the exact Fraction algebra). A claim that
    is text/formula appears when it is identical after the same symbol
    normalization the verifier applies (unicode minus/union/exponents and
    spacing). The mathematics itself is independently re-solved by
    app.utils.math_verify before any render; this only stops honest
    display-form differences from failing the task contract.
    """
    from app.utils.math_intervals import (
        IntervalSet,
        format_interval_set,
        parse_interval_set,
    )
    from app.utils.math_verify import _LOG_SUBSCRIPT_RE, _prep, claimed_track_set

    def normalize(text: str) -> str:
        normalized = _LOG_SUBSCRIPT_RE.sub(r"log_{\1}(", _prep(text))
        return " ".join(normalized.split())

    nl = scene.number_line
    if nl is None:
        return False
    claim_set = parse_interval_set(claim)
    if not isinstance(claim_set, str):
        candidates: list[IntervalSet] = []
        for track in (nl.domain, nl.solution):
            if track is None:
                continue
            track_set = claimed_track_set(track)
            if not isinstance(track_set, str):
                candidates.append(track_set)
        for case in nl.cases:
            case_set = parse_interval_set(case.solution)
            if not isinstance(case_set, str):
                candidates.append(case_set)
        if any(candidate.equals(claim_set) for candidate in candidates):
            return True
        return format_interval_set(claim_set) in haystack
    normalized_claim = normalize(claim)
    if not normalized_claim:
        return False
    return normalized_claim in normalize(haystack)


def _expand_acceptable_families(task: VisualTask) -> set[str]:
    """Normalize the model-authored family list against our registry (pure).

    The planner sometimes writes task_type names ("definition_overview",
    "graph_relationship") into acceptable_families instead of scene kinds —
    near-miss vocabulary that would false-refuse a scene our own capability
    registry authorizes. Each entry that IS a scene kind is kept; each entry
    that matches a known task_type expands to that task type's registry
    families; unrecognized entries authorize nothing and are dropped. The
    hard capability check against TASK_FAMILIES still runs separately, so
    this can never admit a scene the registry forbids. An EMPTY list after
    expansion means the planner expressed no family constraint at all — the
    registry is then the only (and stricter) authority.
    """
    known_kinds = set(SUPPORTED_FAMILIES)
    expanded: set[str] = set()
    for family in task.acceptable_families or []:
        token = str(family).strip()
        if token in known_kinds:
            expanded.add(token)
        elif token in TASK_FAMILIES:
            expanded.update(TASK_FAMILIES[token])
    return expanded


def validate_task_link(spec: VisualSpec) -> list[str]:
    """Enforce the task -> representation -> scene contract.

    No task plan: legacy behavior, no errors (old specs keep working).
    With a task plan: the scene kind must be authorized by BOTH the task's
    acceptable_families (alias-normalized) and the capability registry; every
    required element must be satisfiable (at least one of its distinctive
    tokens present in the rendered lesson text: scene, equations, steps,
    points, composition); every exact claim must appear verbatim.
    Failures mean the scene cannot teach the planned task — refuse, don't
    render.
    """
    errors: list[str] = []
    task = spec.deterministic.task
    if task is None:
        return errors
    det = spec.deterministic
    scene = det.scene
    if scene is None:
        errors.append("task plan present but scene is missing")
        return errors
    kind = scene.scene_kind
    authorized = _expand_acceptable_families(task)
    if authorized:
        if kind not in authorized:
            errors.append(f"scene kind {kind!r} not in task acceptable_families")
    elif task.acceptable_families:
        # A non-empty list matched neither a scene kind nor a known task type:
        # the planner expressed a constraint we cannot honor. Silently
        # treating it as "no constraint" would neutralize the contract.
        errors.append(
            "task acceptable_families contains no recognized scene kind or task type"
        )
    registry_ok = TASK_FAMILIES.get(task.task_type, [])
    if kind not in registry_ok:
        errors.append(f"scene kind {kind!r} cannot express task {task.task_type!r}")
    # The haystack is STRICTLY what the rendered artifact shows: with a scene
    # present (always true when a task plan exists) both render paths center
    # on the scene hero — the card content (equations/steps/points) renders
    # only when no scene exists, so it can never satisfy a claim. A claim
    # living only in deterministic.equations is invisible in the product and
    # is promoted verbatim into a visible callout by
    # promote_equation_claims() before this gate runs. Generic satellites are
    # capped by the hero, so callouts beyond the cap are not visible either.
    hero_text = ""
    composed_text = ""
    if scene.scene_kind == "number_line":
        # number_line renders through its own deterministic path; its scene
        # fields (problem/domain/cases/solution) are the rendered material.
        hero_text = _scene_text(scene, generic_cap=GENERIC_SATELLITE_CAP)
    else:
        # Visibility is exact: read the text nodes that the SAME hero renderer
        # actually paints. Spec strings that were wrapped into an ellipsis,
        # sliced away by a card cap, or never drawn cannot satisfy a claim.
        # Invalid geometry may make the renderer raise; validation reports the
        # structural error separately and the claim fails honestly.
        try:
            hero_text = rendered_scene_text(det)
        except Exception:
            hero_text = ""
        # The composition layer (Typst) owns every non-hero word. Ask the SAME
        # content extractor the renderer uses, so a detail capped into a card,
        # a suppressed duplicate chip, or an echo-filtered edge label is
        # counted only when the composed page truly prints it.
        try:
            content = extract_lesson_content(spec)
            composed_parts = [
                content.title,
                content.subtitle,
                content.reasoning,
                content.result_text,
                content.takeaway,
            ]
            for callout in content.callouts:
                composed_parts.extend([callout.heading, callout.body])
            composed_text = " ".join(part for part in composed_parts if part)
        except Exception:
            composed_text = ""
    haystack = " ".join([hero_text, composed_text]).casefold()
    hay_tokens = _tokens(haystack) | _scene_structural_tokens(scene)
    # A comparison's motion-panel claims describe what each variant's body
    # actually does (v = 0 at the rolling contact, v = 2v_CM at the top). They
    # must be attached to a panel/vector, not merely printed in a composition
    # callout: otherwise a two-panel scene can pass the gate while missing the
    # combined variant entirely. A TABLE is different: its hero prints the
    # matrix and its composition result IS the table verdict, so the whole
    # composed page is an honest place for a printed claim (the verdict is
    # separately evidence-gated by table_quality).
    claim_haystack = haystack
    if scene.scene_kind == "comparison":
        comparison = scene.comparison
        table_only = comparison is not None and comparison.table is not None
        if not table_only:
            claim_haystack = hero_text.casefold()
    claim_errors: list[str] = []
    for claim in task.exact_claims:
        norm = " ".join(str(claim).split())
        if not norm:
            continue
        if norm.casefold() in claim_haystack or _claim_key(norm) in _claim_key(claim_haystack):
            continue
        if scene.scene_kind == "number_line" and _number_line_claim_matches(
            norm, scene, haystack
        ):
            continue
        claim_errors.append(f"exact claim missing from scene: {claim!r}")
    # A taxonomy may name a category "Series Sum" while the planner's contract
    # calls it "Summation Notation". A zero-token heading synonym is not a
    # missing concept when every exact formula/claim is visibly printed (the
    # stronger, verbatim contract). Restrict the relaxation to concept maps so
    # geometric families still fail when a required entity is absent.
    claims_complete = bool(task.exact_claims) and not claim_errors
    for element in task.required_elements:
        need = _distinctive(element)
        if not need:
            continue
        # Any-token match: free-text explanations paraphrase freely
        # ("torque equals r cross F" vs symbols τ/r/F), so demanding half
        # the tokens would refuse honest visuals. Zero shared tokens still
        # fails ("unicorn levitation field"). Exact formulas/values stay
        # verbatim-strict via exact_claims below — math fidelity lives there.
        if not any(w in hay_tokens for w in need):
            if kind == "concept_map" and claims_complete:
                continue
            errors.append(f"required element unsatisfied by scene: {element!r}")
    errors.extend(claim_errors)
    return errors


_CLAIM_MISSING_RE = re.compile(r"^exact claim missing from scene: '(?P<claim>.*)'$")


def promote_equation_claims(spec: VisualSpec, gate_errors: list[str]) -> int:
    """Make equations-only claims visible WITHOUT a model call.

    Neither the composed lesson nor the scene hero renders
    ``deterministic.equations`` (that card content exists for scene-less
    specs only), so a claim satisfied solely by an equation entry passes the
    naive text check while being invisible in the product. This promotes such
    claims — verbatim, nothing invented — into a channel that actually
    renders: generic satellite slots (while capacity remains) or composition
    callout chips. Returns the number of claims promoted. Callers recompute
    the gate afterwards; claims that fit nowhere still go to model repair.

    ``gate_errors`` must come from ``validate_legacy_visual``/``validate_task_link``
    so the promoted text provably closes the reported gap.
    """
    det = spec.deterministic
    scene = det.scene
    if scene is None or scene.scene_kind == "number_line":
        return 0
    missing: list[str] = []
    for err in gate_errors:
        m = _CLAIM_MISSING_RE.match(err)
        if m:
            missing.append(m.group("claim"))
    if not missing:
        return 0
    equations = [
        (e.expression.strip(), e.meaning.strip())
        for e in det.equations
        if e.expression.strip()
    ]
    generic = scene.generic if scene.scene_kind == "generic" else None
    comp = det.composition
    visible_satellites = (
        len([c for c in (generic.callouts or []) if (c or "").strip()])
        if generic is not None
        else 0
    )
    promoted = 0
    for claim in missing:
        ck = _claim_key(claim)
        match: tuple[str, str] | None = None
        for expr, meaning in equations:
            if ck == _claim_key(expr) or ck in _claim_key(f"{expr} {meaning}"):
                match = (expr, meaning)
                break
        if match is None:
            continue
        _, meaning = match
        if generic is not None:
            if visible_satellites >= GENERIC_SATELLITE_CAP:
                continue
            generic.callouts.append(claim)
            visible_satellites += 1
            promoted += 1
        elif comp is not None:
            # Chip bounds come from the CompositionCallout schema (label 24,
            # value 80) and LessonComposition (6 callouts); refuse to exceed
            # them — a claim that cannot fit visibly goes to model repair.
            if len(comp.callouts) >= 6 or len(claim) > 24 or len(meaning) > 80:
                continue
            existing_ids = {c.id for c in comp.callouts}
            idx = 1
            while f"promoted-claim-{idx}" in existing_ids:
                idx += 1
            comp.callouts.append(
                CompositionCallout(
                    id=f"promoted-claim-{idx}", label=claim, value=meaning
                )
            )
            promoted += 1
    return promoted


TYMST_TIMEOUT_SECONDS = 25
MAX_CALLOUTS = 4


@dataclass
class LessonCallout:
    heading: str
    body: str = ""


@dataclass
class LessonContent:
    title: str
    subtitle: str = ""
    hero_svg: str = ""
    hero_kind: str = ""
    layout: str = ""
    callouts: list[LessonCallout] = field(default_factory=list)
    reasoning: str = ""
    reasoning_steps: list[tuple[str, str]] = field(default_factory=list)
    result_text: str = ""
    takeaway: str = ""


@dataclass
class RenderedLesson:
    out_format: str  # "pdf" | "png" | "svg"
    data: bytes
    hero_kind: str
    ms_total: float
    ms_hero: float = 0.0
    ms_compose: float = 0.0
    warnings: list[str] = field(default_factory=list)
    fallback_used: bool = False


class LessonValidationError(ValueError):
    """Semantic spec is invalid — refuse to render rather than guess."""


class TypstUnavailable(RuntimeError):
    """Typst CLI binary not found — caller must fall back honestly."""


class TypstCompileError(RuntimeError):
    """Typst source failed to compile — never return a partial visual."""


def _finite(value: object, what: str, errors: list[str]) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        errors.append(f"{what} is not a number: {value!r}")
        return None
    if not math.isfinite(number):
        errors.append(f"{what} is not finite: {value!r}")
        return None
    return number


_RELATION_OPERATORS = ("=", "<", ">", "\u2264", "\u2265", "\u2260", "\u2248", "\u221d", "\u2192", "\u27f6")


def _relation_errors(relation: object) -> list[str]:
    """The scene relation renders as the visual's highlighted answer chip.

    Observed defect (case 7): a bounded reconcile repair chose the node label
    "Plant G(s)" as the relation, so the hero showed a block name where the
    PID equation belongs. A relation must be a complete mathematical
    statement (equality/comparison), not a label; empty is allowed (the chip
    is simply omitted). The message doubles as the repair instruction.
    """
    expr = (getattr(relation, "expression", "") or "").strip()
    if not expr or any(op in expr for op in _RELATION_OPERATORS):
        return []
    return [
        "scene relation is not an equation (it must contain = or a "
        f"comparison operator): {expr!r} — replace it with the actual "
        "relation equation or leave the relation empty"
    ]


def validate_legacy_visual(spec: VisualSpec) -> list[str]:
    """Enforceable contract for the legacy (non-v3) deterministic path.

    Same bar as the v3 gate when a scene exists (full semantic validation,
    including task links). Sceneless specs are allowed only as explicit
    equation/step/point cards with a title — the renderer's only other
    supported shape; anything emptier is refused. Generative specs are
    validated separately by their own contract.
    """
    errors = validate_lesson_spec(spec)
    det = spec.deterministic
    if det.scene is not None:
        return errors
    if (det.title or "").strip() and any([det.equations, det.steps, det.points]):
        return []
    return ["legacy card has no title or card content"]


def _validate_comparison_table(table: ComparisonTable | None) -> list[str]:
    """Validate a feature comparison matrix (2-3 columns x 2-6 rows)."""
    errors: list[str] = []
    if table is None:
        return ["comparison table payload is missing"]
    matrix = table
    rows = [r for r in matrix.rows if (r.label or "").strip()]
    columns = [c for c in matrix.columns if (c.label or "").strip()]
    if len(rows) < 2:
        errors.append("comparison table needs at least 2 labeled rows")
    if len(columns) < 2:
        errors.append("comparison table needs at least 2 labeled columns")
    for column in columns:
        if len(column.values) != len(rows):
            errors.append(
                f"comparison column {column.label!r} has {len(column.values)} values "
                f"for {len(rows)} rows"
            )
            continue
        for index, value in enumerate(column.values):
            if not (value or "").strip():
                errors.append(
                    f"comparison cell {column.label!r}/{rows[index].label!r} is empty"
                )
            elif len(value) > 80:
                errors.append(
                    f"comparison cell {column.label!r}/{rows[index].label!r} exceeds 80 chars"
                )
    return errors


def _validate_concept_map(cmap: VisualConceptMap) -> list[str]:
    """Validate a hierarchy/cycle/timeline map without inventing structure."""
    errors: list[str] = []
    nodes = [n for n in cmap.nodes if (n.label or "").strip()]
    if len(nodes) < 3:
        errors.append("concept_map needs at least 3 labeled nodes")
        return errors
    limit = 8 if cmap.layout == "cycle" else 6
    if cmap.layout in ("cycle", "timeline") and len(nodes) > limit:
        errors.append(
            f"concept_map {cmap.layout} supports at most {limit} stages/events"
        )
    ids = [(n.id or "").strip() for n in nodes]
    if any(not node_id for node_id in ids):
        errors.append("concept_map nodes need non-empty ids")
    duplicates = {node_id for node_id in ids if node_id and ids.count(node_id) > 1}
    if duplicates:
        errors.append(f"concept_map node ids are duplicated: {sorted(duplicates)}")
    known = {node_id for node_id in ids if node_id}
    for edge in cmap.edges:
        source = (edge.source or "").strip()
        target = (edge.target or "").strip()
        if source not in known or target not in known:
            errors.append(f"concept_map edge references unknown node: {source!r}->{target!r}")
        if source and source == target:
            errors.append(f"concept_map edge is self-referential: {source!r}")
    if cmap.layout == "tree":
        roots = [n for n in nodes if not (n.parent or "").strip()]
        if len(roots) != 1:
            errors.append(f"concept_map tree needs exactly one root, found {len(roots)}")
            return errors
        parents: dict[str, str] = {}
        for node in nodes:
            node_id = (node.id or "").strip()
            parent = (node.parent or "").strip()
            if parent:
                if parent not in known:
                    errors.append(f"concept_map node {node_id!r} has unknown parent {parent!r}")
                else:
                    parents[node_id] = parent
        # Cycle and depth checks: every chain must terminate at the root.
        root_id = (roots[0].id or "").strip()
        for node in nodes:
            node_id = (node.id or "").strip()
            seen = {node_id}
            current = parents.get(node_id, "")
            depth = 0
            while current:
                depth += 1
                if depth > 6:
                    errors.append(f"concept_map parent chain for {node_id!r} is too deep")
                    break
                if current in seen:
                    errors.append(f"concept_map parent chain for {node_id!r} contains a cycle")
                    break
                seen.add(current)
                current = parents.get(current, "")
            if current == "" and node_id != root_id and node_id not in parents:
                errors.append(
                    f"concept_map node {node_id!r} is disconnected (no parent and not the root)"
                )
            if depth > 4:
                errors.append(f"concept_map node {node_id!r} exceeds depth 4")
    return errors


def validate_lesson_spec(spec: VisualSpec) -> list[str]:
    """Return a list of semantic errors (empty == valid). Pure, no I/O."""
    errors: list[str] = []
    # Quality-gate the comparison table on a copy first: redundant axes and
    # untraceable verdicts must never satisfy a claim that the student cannot
    # actually see. The render path applies the identical normalization.
    gated, _ = normalize_visual_spec(spec)
    spec = gated
    det: DeterministicVisual = spec.deterministic
    if not (det.title or "").strip():
        errors.append("deterministic.title is empty")
    scene: VisualScene | None = det.scene
    if scene is None:
        errors.append("deterministic.scene is missing")
        return errors
    if scene.scene_kind not in SUPPORTED_FAMILIES:
        errors.append(f"unsupported scene_kind: {scene.scene_kind!r}")
        return errors
    kind = scene.scene_kind
    if kind == "force_diagram":
        force = scene.force
        if force is None:
            errors.append("force_diagram has no force payload")
            return errors
        labeled = [v for v in force.vectors if (v.label or "").strip()]
        if not labeled:
            errors.append("force_diagram has no labeled vectors")
        known = {(force.object.label or "").strip(), *(v.label.strip() for v in labeled)}
        for v in labeled:
            _finite(v.angle_deg, f"vector {v.label!r} angle_deg", errors)
            _finite(v.length, f"vector {v.label!r} length", errors)
            tail = (v.tail or "").strip()
            if tail and tail not in known:
                errors.append(f"vector {v.label!r} tails on unknown {tail!r}")
        for a in force.angles:
            for lab in a.between:
                if lab.strip() not in {v.label.strip() for v in labeled}:
                    errors.append(f"angle {a.label!r} references unknown vector {lab!r}")
        for arc in force.arcs:
            if (arc.around or "").strip() and (arc.around or "").strip() not in known:
                errors.append(f"arc {arc.label!r} wraps unknown {arc.around!r}")
        errors.extend(_relation_errors(force.relation))
    elif kind == "plot":
        plot = scene.plot
        if plot is None or not (plot.curves or plot.segments or plot.markers):
            errors.append("plot has no curves, segments, or markers")
            return errors
        _finite(plot.x_min, "plot.x_min", errors)
        _finite(plot.x_max, "plot.x_max", errors)
        _finite(plot.y_min, "plot.y_min", errors)
        _finite(plot.y_max, "plot.y_max", errors)
        try:
            if float(plot.x_max) <= float(plot.x_min) or float(plot.y_max) <= float(plot.y_min):
                errors.append("plot ranges are degenerate")
        except (TypeError, ValueError):
            pass
        for c in plot.curves:
            if not (c.expr or "").strip() and not c.points:
                errors.append(f"curve {c.label!r} has neither expr nor points")
            for p in c.points:
                if len(p) != 2 or _finite(p[0], "point x", errors) is None or _finite(p[1], "point y", errors) is None:
                    errors.append(f"curve {c.label!r} has a malformed point")
                    break
        for m in plot.markers:
            _finite(m.x, f"marker {m.label!r} x", errors)
            _finite(m.y, f"marker {m.label!r} y", errors)
        for s in plot.segments:
            for value, what in (
                (s.x1, "segment x1"), (s.y1, "segment y1"),
                (s.x2, "segment x2"), (s.y2, "segment y2"),
            ):
                _finite(value, f"{what} ({s.label!r})", errors)
    elif kind == "process_flow":
        flow = scene.flow
        if flow is None or len([n for n in flow.nodes if (n.label or "").strip()]) < 2:
            errors.append("process_flow needs at least 2 labeled nodes")
            return errors
        n = len(flow.nodes)
        for c in flow.connectors:
            if not (0 <= c.source < n and 0 <= c.target < n):
                errors.append(f"connector {c.source}->{c.target} out of range for {n} nodes")
        errors.extend(_relation_errors(flow.relation))
    elif kind == "generic":
        if scene.generic is None or not (scene.generic.central_label or "").strip():
            errors.append("generic scene needs a central_label")
        elif len([c for c in (scene.generic.callouts or []) if (c or "").strip()]) < 3:
            # A central box with 1-2 chips composes into a near-empty page
            # (observed failure: title box + two formula callouts + blank
            # space presented as a successful visual). Refuse honestly so the
            # route reports unavailable instead of shipping sparse output.
            errors.append("generic scene needs at least 3 callouts to fill a lesson page")
    elif kind == "comparison":
        comp = scene.comparison
        if comp is None:
            errors.append("comparison has no comparison payload")
            return errors
        table_used = comp.table is not None
        if table_used:
            errors.extend(_validate_comparison_table(comp.table))
        errors.extend(table_quality_errors(comp.table))
        if not table_used:
            panels = [p for p in (comp.panels or []) if (p.label or "").strip()][:3]
            if len(panels) < 2:
                errors.append("comparison needs at least 2 labeled panels or a feature table")
                return errors
            for p in panels:
                labeled = [v for v in p.vectors if (v.label or "").strip()]
                if not labeled and not p.arcs:
                    errors.append(f"comparison panel {p.label!r} has no vectors or arcs")
                known = {
                    (p.object.label or "").strip(),
                    *((point.label or "").strip() for point in p.points),
                    *(v.label.strip() for v in labeled),
                }
                for v in labeled:
                    _finite(v.angle_deg, f"vector {v.label!r} angle_deg", errors)
                    _finite(v.length, f"vector {v.label!r} length", errors)
                    tail = (v.tail or "").strip()
                    if tail and tail not in known:
                        errors.append(f"vector {v.label!r} tails on unknown {tail!r}")
        errors.extend(_relation_errors(comp.relation))
    elif kind == "concept_map":
        cmap = scene.concept_map
        if cmap is None:
            errors.append("concept_map has no concept_map payload")
            return errors
        errors.extend(_validate_concept_map(cmap))
        errors.extend(_relation_errors(cmap.relation))
    elif kind == "number_line":
        nl = scene.number_line
        if nl is None:
            errors.append("number_line has no number_line payload")
            return errors
        # Deterministic mathematics verification: the scene may render ONLY
        # when the verifier can re-solve the problem and reproduce every
        # claimed interval set exactly (see app.utils.math_verify). Anything
        # unsupported or mismatched is refused — never displayed unverified.
        from app.utils.math_verify import verify_number_line

        errors.extend(verify_number_line(nl))
    comp = det.composition
    if comp is not None:
        errors.extend(_validate_composition(comp))
        # Req 6 — contradiction gate: when the scene carries its own relation
        # expression AND the composition highlights a result, both render as
        # "the answer" and must agree (whitespace-insensitive; math is
        # case-significant so no case folding). Deeper numeric cross-checking
        # (e.g. requiring every number in reasoning to appear in scene labels)
        # is deliberately NOT done: derived arithmetic such as θ = 90° − 55°
        # legitimately combines emitted values, so such a check would reject
        # honest reasoning. Anti-invention for callouts/reasoning is enforced
        # at the prompt layer (composition grounding rules) instead.
        relation_expr = _scene_relation_expression(scene)
        result_expr = (comp.result.expression if comp.result is not None else "") or ""
        if relation_expr.strip() and result_expr.strip():
            if " ".join(relation_expr.split()) != " ".join(result_expr.split()):
                errors.append(
                    "composition result contradicts scene relation: "
                    f"{result_expr!r} != {relation_expr!r}"
                )
    # Task contract: the planned teaching job must be satisfiable by the
    # emitted scene. No task plan = legacy behavior, unchanged.
    errors.extend(validate_task_link(spec))
    return errors


def spec_needs_repair(spec: VisualSpec) -> str | None:
    """Decide whether a spec deserves one bounded source-grounded repair.

    Pure, no I/O. Returns a repair reason, or None when the spec should
    render as-is. Triggers ONLY for:
    - generic family (no geometry by construction; the source may support
      a real scene the first pass missed), and
    - structural emptiness in geometric families (zero labeled
      vectors/curves/nodes/panels — validation would refuse these; a repair
      attempt may salvage a rich source before honest refusal).
    Single-primitive geometric scenes are NOT flagged: a simple concept
    legitimately needs a simple visual, and forcing richness would invent
    content. Returns None for generative specs (different pipeline).
    """
    if spec.render_mode != VisualRenderMode.DETERMINISTIC:
        return None
    det = spec.deterministic
    scene = det.scene if det is not None else None
    if scene is None:
        return "missing scene"
    kind = scene.scene_kind
    if kind == "generic":
        return "generic family carries no geometry"
    if kind == "force_diagram":
        force = scene.force
        if force is None or not [v for v in force.vectors if (v.label or "").strip()]:
            return "force_diagram has no labeled vectors"
    elif kind == "plot":
        plot = scene.plot
        if plot is None or not (plot.curves or plot.segments or plot.markers):
            return "plot has no curves, segments, or markers"
    elif kind == "process_flow":
        flow = scene.flow
        if flow is None or len([n for n in flow.nodes if (n.label or "").strip()]) < 2:
            return "process_flow has fewer than 2 labeled nodes"
    elif kind == "comparison":
        comp = scene.comparison
        has_table = (
            comp is not None
            and comp.table is not None
            and len([c for c in comp.table.columns if (c.label or "").strip()]) >= 2
            and len([r for r in comp.table.rows if (r.label or "").strip()]) >= 2
        )
        if not has_table and (
            comp is None
            or len([p for p in (comp.panels or []) if (p.label or "").strip()]) < 2
        ):
            return "comparison has fewer than 2 labeled panels or a usable feature table"
    elif kind == "concept_map":
        cmap = scene.concept_map
        if cmap is None or len([n for n in cmap.nodes if (n.label or "").strip()]) < 3:
            return "concept_map has fewer than 3 labeled nodes"
    elif kind == "number_line":
        # Worked math is never repaired by adding geometry: correctness is
        # judged by the math verifier, and a thin-looking solution is a
        # legitimate solution. No repair trigger.
        return None
    else:
        return f"unsupported scene_kind: {kind!r}"
    return None


def _scene_relation_expression(scene) -> str:
    """Extract the scene's own relation expression, if any (force/flow/comparison)."""
    try:
        if scene.scene_kind == "force_diagram" and scene.force is not None:
            rel = scene.force.relation
        elif scene.scene_kind == "process_flow" and scene.flow is not None:
            rel = scene.flow.relation
        elif scene.scene_kind == "comparison" and scene.comparison is not None:
            rel = scene.comparison.relation
        elif scene.scene_kind == "concept_map" and scene.concept_map is not None:
            rel = scene.concept_map.relation
        else:
            return ""
        return (rel.expression if rel is not None else "") or ""
    except AttributeError:
        return ""


def _validate_composition(comp) -> list[str]:
    """Validate the explicit v3 composition block (shape already enforced by
    pydantic: id pattern, lengths, counts, extra=forbid). Checks semantic
    rules only; values pass through byte-identical."""
    from app.models.schemas import LessonComposition

    assert isinstance(comp, LessonComposition)
    errors: list[str] = []
    seen_callouts: set[str] = set()
    for c in comp.callouts:
        if c.id in seen_callouts:
            errors.append(f"composition callout id duplicated: {c.id!r}")
        seen_callouts.add(c.id)
        if not c.label.strip():
            errors.append(f"composition callout {c.id!r} has empty label")
    seen_steps: set[str] = set()
    for s in comp.reasoning:
        if s.id in seen_steps:
            errors.append(f"composition reasoning id duplicated: {s.id!r}")
        seen_steps.add(s.id)
        if not s.expression.strip():
            errors.append(f"composition reasoning {s.id!r} has empty expression")
    if comp.result is not None and not (comp.result.expression or "").strip():
        errors.append("composition result has empty expression")
    if (
        not comp.callouts
        and not comp.reasoning
        and comp.result is None
        and not (comp.takeaway or "").strip()
    ):
        errors.append("composition is empty: needs callouts, reasoning, result or takeaway")
    return errors


def select_family(spec: VisualSpec) -> str:
    """Map scene_kind to a visual family. Raises LessonValidationError if unknown."""
    kind = (spec.deterministic.scene.scene_kind if spec.deterministic.scene else "")
    try:
        return SUPPORTED_FAMILIES[kind]
    except KeyError:
        raise LessonValidationError(f"unsupported scene_kind: {kind!r}") from None


def should_use_v3(spec: VisualSpec) -> bool:
    """Route decision for Explain Visually: v3 composition boundary or legacy.

    True only when the flag is on AND the spec is deterministic. Generative
    specs always use the legacy path (v3 composition is deterministic-only).
    number_line (worked mathematics) also stays on the legacy deterministic
    path: its layout is content-fitted and exactness-critical, so it must not
    go through the generic Typst page composition. Pure function of
    (flag, spec) — no I/O, so the route stays thin and the decision is
    unit-testable without FastAPI.
    """
    from app.config import settings

    if not settings.EXPLAIN_VISUALLY_V3 or spec.render_mode != VisualRenderMode.DETERMINISTIC:
        return False
    scene = spec.deterministic.scene
    if scene is not None and scene.scene_kind == "number_line":
        return False
    return True


def _clean(text: str) -> str:
    return " ".join((text or "").split())


MAP_DETAIL_CARD_CAP = 8


def _visible_map_detail_ids(cmap: VisualConceptMap) -> set[str]:
    """Ids whose node detail is actually printed somewhere on the page.

    Horizontal/timeline heroes print details inside cards; a tall tree prints
    header details in the hero and leaf details in the callout grid; a cycle
    prints every stage detail in the grid. Validation must use this exact set,
    otherwise a hidden detail can satisfy an exact claim while the student
    never sees it.
    """
    nodes = [n for n in cmap.nodes if (n.label or "").strip()]
    all_ids = {(n.id or "").strip() for n in nodes}
    if cmap.layout == "timeline" or concept_map_is_horizontal(cmap):
        return all_ids
    if cmap.layout == "cycle":
        return all_ids
    parent_ids = {
        (n.parent or "").strip()
        for n in nodes
        if (n.parent or "").strip()
    }
    leaves = [
        (n.id or "").strip()
        for n in _map_nodes_deepest_first(cmap)
        if (n.id or "").strip() not in parent_ids
    ]
    return parent_ids | set(leaves[:MAP_DETAIL_CARD_CAP])


def _map_nodes_deepest_first(cmap: VisualConceptMap) -> list[MapNode]:
    """Order map nodes so leaf teaching details are printed before headers.

    Used by the composer when the hero draws labels only (tall tree/cycle):
    a formula on a leaf matters more than a category subtitle, and the callout
    grid is capped, so depth-first order keeps the strongest content visible.
    """
    nodes = [n for n in cmap.nodes if (n.label or "").strip()]
    by_id = {(n.id or "").strip(): n for n in nodes}
    depth: dict[str, int] = {}

    def resolve(node: MapNode, guard: int = 0) -> int:
        node_id = (node.id or "").strip()
        if node_id in depth:
            return depth[node_id]
        parent = (node.parent or "").strip()
        if not parent or parent not in by_id or guard > len(by_id):
            depth[node_id] = 0
            return 0
        depth[node_id] = resolve(by_id[parent], guard + 1) + 1
        return depth[node_id]

    for node in nodes:
        resolve(node)
    indexed = list(enumerate(nodes))
    indexed.sort(key=lambda pair: (-depth.get((pair[1].id or "").strip(), 0), pair[0]))
    return [node for _, node in indexed]


def _layout_for(family: str, scene: VisualScene | None) -> str:
    """Choose a composition layout from the visual structure, never the subject.

    The hero is the same validated geometry; only the page architecture changes:
    annotated scenes put notes beside the visual, process bands put the sequence
    under it, comparisons give the matrix the full width, and maps lead with the
    relational structure. This is what stops every topic collapsing into one
    universal card stack.
    """
    if family in ("vector", "coordinate"):
        return "annotated"
    if family == "flow":
        return "band"
    if family == "comparison":
        return "compare"
    if family == "concept_map":
        if scene is not None and scene.concept_map is not None:
            return f"map_{scene.concept_map.layout}"
        return "map_tree"
    return "focus"


def extract_lesson_content(spec: VisualSpec, hero_svg: str = "") -> LessonContent:
    """Pull structured lesson content from the spec WITHOUT rewriting values.

    Numbers, labels and equations pass through byte-identical. No LLM call.
    """
    det = spec.deterministic
    scene = det.scene
    assert scene is not None
    family = select_family(spec)
    comp = det.composition
    if comp is not None:
        # v3 explicit composition: prefer it over derivation. Values pass
        # through byte-identical; order kept; no markdown parsing, no LLM.
        # The generic scene IS a callout box set (central label + satellite
        # boxes), and a comparison hero already carries every panel label and
        # note. Rendering those same labels again as a chip grid duplicates
        # the visual verbatim (observed: case-10 formula sheet showed every
        # formula twice; rolling motion repeated its three panels as cards).
        # The hero carries the callout set for these families; the composition
        # still contributes framing, reasoning, result and takeaway.
        # EXCEPTION: an equation promoted by promote_equation_claims() is
        # intentionally NOT in the hero (that is why it was promoted). Hiding
        # it here would pass validation while making the claim invisible.
        suppress_duplicate_chips = scene.scene_kind in ("generic", "comparison")
        map_detail_chips: list[LessonCallout] = []
        if scene.scene_kind == "concept_map" and scene.concept_map is not None:
            cmap = scene.concept_map
            details_in_hero = cmap.layout == "timeline" or concept_map_is_horizontal(cmap)
            if details_in_hero:
                # These heroes print each node's detail inside the node card, so
                # repeating the same text as a callout grid is pure duplication.
                suppress_duplicate_chips = True
            else:
                # A tall tree and a cycle draw labels only. Their details are
                # real teaching content (formulas, dates, roles) and would
                # otherwise be invisible; print them as cards, deepest first.
                # This is the exact list validation treats as visible.
                map_detail_chips = [
                    LessonCallout(heading=label, body=detail)
                    for label, detail in _map_detail_chip_pairs(cmap)
                ]
        v3_callouts = map_detail_chips + [
            LessonCallout(heading=c.label.strip(), body=_clean(c.value))
            for c in comp.callouts
            if not suppress_duplicate_chips or c.id.startswith("promoted-claim-")
        ]
        # CompositionCallout caps a single chip at 24/80 chars; the composer's
        # own card grid is bounded per page, so cap here too.
        v3_callouts = v3_callouts[:MAP_DETAIL_CARD_CAP]
        v3_reasoning = "; ".join(
            (f"{s.expression.strip()} - {_clean(s.explanation)}" if _clean(s.explanation) else s.expression.strip())
            for s in comp.reasoning
        )
        v3_steps = [
            (s.expression.strip(), _clean(s.explanation))
            for s in comp.reasoning
            if s.expression.strip()
        ]
        if (
            scene.scene_kind == "comparison"
            and scene.comparison is not None
            and scene.comparison.table is not None
        ):
            # The matrix already states every condition/outcome pair; a
            # reasoning block below it repeats the same two rows verbatim.
            v3_reasoning = ""
            v3_steps = []
        result_text = comp.result.expression.strip() if comp.result is not None else ""
        if (
            scene.scene_kind == "comparison"
            and scene.comparison is not None
            and scene.comparison.table is not None
            and (scene.comparison.table.verdict or "").strip()
        ):
            # The table's verdict is the synthesis of the comparison; prefer
            # it over a terse equilibrium condition as the key result.
            result_text = scene.comparison.table.verdict.strip()
        return LessonContent(
            title=_clean(comp.title) or _clean(det.title) or "Key concept",
            subtitle=_clean(comp.framing) or _clean(spec.concept),
            hero_svg=hero_svg,
            hero_kind=family,
            layout=_layout_for(family, scene),
            callouts=v3_callouts,
            reasoning=v3_reasoning,
            reasoning_steps=v3_steps,
            result_text=result_text,
            takeaway=_clean(comp.takeaway) or _clean(scene.caption),
        )
    callouts: list[LessonCallout] = []
    reasoning_bits: list[str] = []
    result_text = ""
    if family == "vector":
        force = scene.force
        assert force is not None
        for v in force.vectors:
            if (v.label or "").strip():
                callouts.append(LessonCallout(heading=v.label.strip(), body=_clean(v.caption)))
        for a in force.angles:
            reasoning_bits.append(_clean(a.caption) or f"Angle {a.label.strip()} between {', '.join(a.between)}")
        for arc in force.arcs:
            reasoning_bits.append(_clean(arc.caption) or f"{arc.label.strip()} {arc.direction}")
        if force.relation is not None and (force.relation.expression or "").strip():
            result_text = force.relation.expression.strip()
            if (force.relation.caption or "").strip():
                reasoning_bits.append(force.relation.caption.strip())
    elif family == "coordinate":
        plot = scene.plot
        assert plot is not None
        for c in plot.curves:
            if (c.label or "").strip():
                callouts.append(LessonCallout(heading=c.label.strip(), body=""))
        for m in plot.markers:
            if (m.label or "").strip():
                callouts.append(LessonCallout(heading=m.label.strip(), body=""))
        for s in plot.segments:
            if (s.label or "").strip():
                callouts.append(LessonCallout(heading=s.label.strip(), body=""))
        if plot.x_label or plot.y_label:
            reasoning_bits.append(f"Axes: {plot.x_label} × {plot.y_label}".strip())
    elif family == "flow":
        flow = scene.flow
        assert flow is not None
        for nd in flow.nodes:
            if (nd.label or "").strip():
                callouts.append(LessonCallout(heading=nd.label.strip(), body=""))
        for c in flow.connectors:
            label = (c.label or "").strip()
            # Echo filter (R9): bare-word edge labels ("allocates", "Forward")
            # merely restate what the arrows already show — keep only labels
            # carrying symbols/digits ("e(t)", "Kp × e(t)"), which teach.
            if label and any(not (ch.isalpha() or ch.isspace() or ch == "-") for ch in label):
                reasoning_bits.append(label)
        if flow.relation is not None and (flow.relation.expression or "").strip():
            result_text = flow.relation.expression.strip()
    elif family == "generic":
        # The hero already IS the callout box set (central label + satellites).
        # Emitting the same callouts again as Typst chips duplicates every
        # formula verbatim; the hero carries this family's content.
        pass
    elif family == "comparison":
        compa = scene.comparison
        assert compa is not None
        # The comparison hero already renders each panel label, table cell and
        # note, so repeating them as callout cards below the visual is
        # duplication, not explanation. Arc labels are visual marks, not prose.
        if compa.relation is not None and (compa.relation.expression or "").strip():
            result_text = compa.relation.expression.strip()
            if (compa.relation.caption or "").strip():
                reasoning_bits.append(compa.relation.caption.strip())
        elif compa.table is not None and (compa.table.verdict or "").strip():
            result_text = compa.table.verdict.strip()
    elif family == "concept_map":
        cmap = scene.concept_map
        assert cmap is not None
        # The hero carries node labels and the relational layout. Only use
        # details as supporting notes (never repeat the label verbatim).
        # Timeline cards already print their detail inside the hero; repeating
        # them as a callout grid would duplicate every line.
        if cmap.layout != "timeline" and not concept_map_is_horizontal(cmap):
            parent_ids = {
                (n.parent or "").strip()
                for n in cmap.nodes
                if (n.parent or "").strip()
            }
            visible_detail_ids = _visible_map_detail_ids(cmap)
            for node in cmap.nodes:
                node_id = (node.id or "").strip()
                # Tall-tree headers print their detail inside the hero box;
                # only leaf details still need a card. Cycle stages print
                # labels only, so every detail is card-worthy.
                if node_id not in visible_detail_ids:
                    continue
                if cmap.layout == "tree" and node_id in parent_ids:
                    continue
                if (node.detail or "").strip():
                    callouts.append(LessonCallout(heading=node.label.strip(), body=_clean(node.detail)))
        for edge in cmap.edges:
            label = (edge.label or "").strip()
            # Same echo filter as flow: bare words ("type", "includes") only
            # restate the connector, while symbolic labels ("ΔH < 0") teach.
            if label and any(not (ch.isalpha() or ch.isspace() or ch == "-") for ch in label):
                reasoning_bits.append(label)
        if cmap.relation is not None and (cmap.relation.expression or "").strip():
            result_text = cmap.relation.expression.strip()
            if (cmap.relation.caption or "").strip():
                reasoning_bits.append(cmap.relation.caption.strip())
    elif family == "worked_math":
        raise LessonValidationError(
            "worked math renders through the deterministic number-line path; "
            "the composition engine is not used for it"
        )
    else:
        raise LessonValidationError(f"unsupported family for content extraction: {family!r}")
    callout_cap = MAP_DETAIL_CARD_CAP if family == "concept_map" else MAX_CALLOUTS
    callouts = callouts[:callout_cap]
    reasoning = "; ".join(b for b in reasoning_bits if b)
    return LessonContent(
        title=_clean(det.title) or "Key concept",
        subtitle=_clean(spec.concept),
        hero_svg=hero_svg,
        hero_kind=family,
        layout=_layout_for(family, scene),
        callouts=callouts,
        reasoning=reasoning,
        result_text=result_text,
        takeaway=_clean(scene.caption),
    )


_ESCAPE_CHARS = frozenset("\\#$%*_{}@`<>[]")

_ESCAPE_DOC = "Escape Typst markup chars; parens/commas/dots pass through raw."


def _typst_esc(text: str) -> str:
    return "".join(("\\" + ch) if ch in _ESCAPE_CHARS else ch for ch in text)


def build_lesson_typst(content: LessonContent, hero_filename: str = "hero.svg") -> str:
    """Render a content-selected lesson architecture as Typst source.

    Deterministic: same content in, same source out. The scene family picks
    the page architecture (annotated scene, process band, comparison matrix,
    hierarchy, cycle, timeline or focus sheet) and its accent palette; the
    planner's content decides which sections exist. No pixels, no subject-name
    branches: two topics with the same information structure share a layout.
    """
    nl = chr(10)
    subtitle = content.subtitle
    if subtitle and " ".join(subtitle.split()).casefold() == " ".join(content.title.split()).casefold():
        subtitle = ""

    layout = content.layout or {
        "vector": "annotated",
        "coordinate": "annotated",
        "flow": "band",
        "comparison": "compare",
        "concept_map": "map_tree",
        "generic": "focus",
    }.get(content.hero_kind, "focus")

    style = {
        "annotated": ("#4f46e5", "#eef2ff", "ANNOTATED SCENE"),
        "band": ("#7c3aed", "#f5f3ff", "PROCESS BAND"),
        "compare": ("#b45309", "#fff7ed", "COMPARISON MATRIX"),
        "map_tree": ("#15803d", "#f0fdf4", "CONCEPT HIERARCHY"),
        "map_cycle": ("#0e7490", "#ecfeff", "CYCLE MAP"),
        "map_timeline": ("#be123c", "#fff1f2", "TIMELINE"),
        "focus": ("#4f46e5", "#eef7ff", "CONCEPT FOCUS"),
    }
    accent, header_fill, layout_label = style.get(layout, style["focus"])

    number = 0

    def section(label: str) -> str:
        nonlocal number
        number += 1
        return nl.join([f'#sec("{number:02d}", "{label}")', '#v(6pt)'])

    accents = [accent, "#b45309", "#0f766e", "#047857", "#7c3aed", "#be123c"]
    callout_cards = [(c, accents[i % len(accents)]) for i, c in enumerate(content.callouts)]

    def fact(heading: str, body: str, accent_value: str) -> str:
        body_arg = f'[{_typst_esc(body)}]' if body else "none"
        return f'#fact([{_typst_esc(heading)}], {body_arg}, rgb("{accent_value}"))'

    side_by_side = (
        2 <= len(callout_cards) <= 4
        and (content.hero_kind in ("vector", "coordinate") or layout == "map_cycle")
    )

    # The hero SVG already draws its own framed stage on white; wrapping it in
    # a second bordered card produced a double frame around every visual.
    hero = f'#image("{hero_filename}", width: 100%)'
    if side_by_side:
        side = (nl + "#v(7pt)" + nl).join(
            fact(c.heading, c.body, accent) for c, accent in callout_cards
        )
        hero_block = nl.join([
            '#sec("01", "THE VISUAL")',
            '#v(6pt)',
            '#grid(columns: (3fr, 2fr), column-gutter: 14pt,',
            f'  [{hero}],',
            '  [',
            '    #text(size: 8pt, weight: "bold", fill: rgb("#64748b"), tracking: 1pt)[WHAT TO NOTICE]',
            '    #v(5pt)',
            f'    {side}',
            '  ],',
            ')',
        ])
        number = 1
        callouts_block = ""
    else:
        hero_block = nl.join(['#sec("01", "THE VISUAL")', '#v(6pt)', hero])
        number = 1
        callouts_block = ""
        if callout_cards:
            cols = 3 if len(callout_cards) >= 5 else (2 if len(callout_cards) >= 4 else len(callout_cards))
            cells = ("," + nl).join(
                f'  [{fact(c.heading, c.body, accent)}]'
                for c, accent in callout_cards
            ) + ","
            callouts_block = nl.join([
                section("WHAT TO NOTICE"),
                f'#grid(columns: ({", ".join(["1fr"] * cols)}), gutter: 8pt,',
                cells,
                ')',
                '#v(4pt)',
            ])

    reasoning_block = ""
    if content.reasoning_steps:
        cells: list[str] = []
        for i, (eq, explanation) in enumerate(content.reasoning_steps, 1):
            cells.append(
                f'  [#rect(width: 22pt, height: 22pt, radius: 11pt, fill: rgb("{accent}"), '
                f'inset: 0pt)[#align(center + horizon)[#text(size: 8pt, weight: "bold", fill: white)[{i}]]]],'
            )
            body_lines = [f'#text(size: 13pt, weight: "bold")[{_typst_esc(eq)}]']
            if explanation:
                body_lines.extend([
                    '    #v(2pt)',
                    f'    #text(size: 9pt, fill: rgb("#64748b"))[{_typst_esc(explanation)}]',
                ])
            cells.append('  [' + nl.join(body_lines) + '],')
        reasoning_block = nl.join([
            section("HOW IT WORKS"),
            '#grid(columns: (22pt, 1fr), column-gutter: 9pt, row-gutter: 10pt,',
            *cells,
            ')',
            '#v(4pt)',
        ])
    elif content.reasoning:
        reasoning_block = nl.join([
            section("HOW IT WORKS"),
            f'#rect(width: 100%, fill: rgb("{header_fill}"), stroke: 0.7pt + rgb("{accent}"), '
            'radius: 8pt, inset: 12pt)[',
            f'  #text(size: 8pt, weight: "bold", fill: rgb("{accent}"), tracking: 0.8pt)[REASONING]',
            '  #v(4pt)',
            f'  #text(size: 10pt, fill: rgb("#334155"))[{_typst_esc(content.reasoning)}]',
            ']',
        ])

    result_block = ""
    if content.result_text:
        result_block = nl.join([
            section("THE RESULT"),
            '#rect(width: 100%, fill: rgb("#ecfdf5"), stroke: 0.7pt + rgb("#a7f3d0"), '
            'radius: 10pt, inset: (x: 20pt, y: 16pt))[',
            '  #grid(columns: (1fr, auto), align: horizon,',
            '    [',
            '      #text(size: 8pt, fill: rgb("#047857"), weight: "bold", tracking: 1pt)[KEY RESULT]',
            '      #v(4pt)',
            f'      #text(size: 20pt, weight: "bold", fill: rgb("#065f46"))[{_typst_esc(content.result_text)}]',
            '    ],',
            '    [#text(size: 8pt, fill: rgb("#047857"), weight: "bold", tracking: 0.6pt)[DETERMINISTIC]],',
            '  )',
            ']',
        ])

    takeaway_block = ""
    if content.takeaway:
        takeaway_block = nl.join([
            section("REMEMBER THIS"),
            '#rect(width: 100%, fill: rgb("#ecfdf5"), stroke: 0.7pt + rgb("#a7f3d0"), '
            'radius: 8pt, inset: 12pt)[',
            f'  #text(size: 10.5pt, fill: rgb("#065f46"))[{_typst_esc(content.takeaway)}]',
            ']',
        ])

    subtitle_block = ""
    if subtitle:
        subtitle_block = nl.join([
            '  #v(5pt)',
            f'  #text(size: 11pt, fill: rgb("#475569"))[{_typst_esc(subtitle)}]',
        ])

    header = nl.join([
        f'#rect(width: 100%, fill: rgb("{header_fill}"))[',
        f'  #rect(width: 100%, height: 3pt, fill: rgb("{accent}"))',
        '  #pad(x: 44pt, top: 26pt, bottom: 28pt)[',
        '    #grid(columns: (1fr, auto), align: horizon,',
        f'      [#text(size: 8pt, fill: rgb("{accent}"), weight: "bold", tracking: 1.4pt)[DETERMINISTIC VISUAL LESSON]],',
        f'      [#text(size: 8pt, fill: rgb("#64748b"), weight: "bold", tracking: 1pt)[{layout_label}]],',
        '    )',
        '    #v(8pt)',
        f'    #text(size: 28pt, weight: "bold", fill: rgb("#0f172a"))[{_typst_esc(content.title)}]',
        subtitle_block,
        '  ]',
        ']',
    ])

    footer = nl.join([
        '#v(8pt)',
        '#text(size: 7.5pt, fill: rgb("#94a3b8"))[Deterministic composition · source values unchanged]',
    ])

    prelude = nl.join([
        '#set page(width: 800pt, height: auto, margin: 0pt, fill: white)',
        '#set text(font: "DejaVu Sans", size: 10pt, fill: rgb("#0f172a"))',
        '#set par(leading: 0.7em)',
        '',
        '#let indigo = rgb("' + accent + '")',
        '#let rule = rgb("#e2e8f0")',
        '#let sec(num, label) = block(width: 100%)[',
        '  #grid(columns: (3pt, auto, 1fr), column-gutter: 8pt, align: horizon,',
        '    rect(width: 3pt, height: 22pt, fill: indigo),',
        '    text(size: 8.5pt, weight: "bold", fill: indigo, tracking: 1.2pt)[#num],',
        '    text(size: 10pt, weight: "bold", fill: indigo, tracking: 1pt)[#label],',
        '  )',
        '  #v(4pt)',
        '  #line(length: 100%, stroke: 0.6pt + rule)',
        ']',
        '#let fact(heading, body, accent) = rect(width: 100%, fill: white, '
        'stroke: 0.7pt + rule, radius: 8pt, inset: 11pt)[',
        '  #text(size: 16pt, weight: "bold", fill: accent)[#heading]',
        '  #if body != none [',
        '    #v(3pt)',
        '    #text(size: 10.5pt, fill: rgb("#475569"))[#body]',
        '  ]',
        ']',
        '',
    ])

    if layout == "compare":
        blocks = [hero_block, result_block, callouts_block, reasoning_block, takeaway_block]
    elif layout == "band":
        blocks = [hero_block, reasoning_block, callouts_block, result_block, takeaway_block]
    elif layout in ("map_tree", "map_timeline"):
        blocks = [hero_block, callouts_block, reasoning_block, result_block, takeaway_block]
    else:
        blocks = [hero_block, callouts_block, reasoning_block, result_block, takeaway_block]
    body = nl.join([
        header,
        '#pad(x: 44pt, top: 22pt, bottom: 28pt)[',
        *[block for block in blocks if block],
        '#v(14pt)',
        footer,
        ']',
    ])
    # Section numbers are allocated while blocks are BUILT, but layouts emit
    # them in a different order (compare puts the result before reasoning).
    # Renumber in final reading order so the page never says 03 before 02.
    section_seq = 0

    def _next_section(match: re.Match[str]) -> str:
        nonlocal section_seq
        section_seq += 1
        return f'#sec("{section_seq:02d}", "'

    body = re.sub(r'#sec\("\d{2}", "', _next_section, body)
    return prelude + body + nl


@functools.lru_cache(maxsize=1)
def typst_status() -> dict:
    """Deployment check: official Typst CLI present, and which version.

    Cached for the process lifetime (the binary cannot change at runtime).
    Used by /health so a missing Typst is visible in deployment, never silent.
    """
    binary = shutil.which("typst")
    if binary is None:
        return {"available": False, "version": ""}
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return {"available": False, "version": ""}
        version = (proc.stdout or proc.stderr or "").strip().splitlines()[0]
        return {"available": bool(version), "version": version}
    except Exception:
        return {"available": False, "version": ""}


def compile_typst(
    source: str,
    out_format: str = "pdf",
    timeout: int = TYMST_TIMEOUT_SECONDS,
    extra_files: dict[str, str] | None = None,
) -> bytes:
    """Compile Typst source with the official CLI. Raises TypstUnavailable /
    TypstCompileError — never returns a partial visual.

    `extra_files` maps filenames (e.g. ``{"hero.svg": "<svg/>"}``) to contents;
    they are written next to ``in.typ`` so ``image("hero.svg")`` resolves.
    """
    binary = shutil.which("typst")
    if binary is None:
        raise TypstUnavailable("typst CLI not found on PATH")
    with tempfile.TemporaryDirectory() as td:
        inp = str(Path(td) / "in.typ")
        out = str(Path(td) / f"out.{out_format}")
        Path(inp).write_text(source, encoding="utf-8")
        for filename, contents in (extra_files or {}).items():
            target = Path(td) / filename
            if target.parent != Path(td):
                raise TypstCompileError(f"refusing to write outside build dir: {filename!r}")
            target.write_text(contents, encoding="utf-8")
        try:
            proc = subprocess.run(
                [binary, "compile", inp, out, "--format", out_format],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise TypstCompileError(f"typst compile timed out after {timeout}s") from e
        if proc.returncode != 0:
            raise TypstCompileError(f"typst failed: {(proc.stderr or '')[:400]}")
        data = Path(out).read_bytes() if Path(out).exists() else b""
        if not data:
            raise TypstCompileError("typst produced no output")
        return data


def render_visual_lesson(spec: VisualSpec, out_format: str = "pdf") -> RenderedLesson:
    """Production composition contract.

    Validates semantics -> selects family -> renders bare hero geometry ->
    extracts structured content -> composes with Typst. Falls back to the bare
    hero SVG (honest, never fake) when Typst is unavailable or fails.
    """
    t0 = time.perf_counter()
    spec, table_report = normalize_visual_spec(spec)
    if table_report.dropped_columns:
        logger.info(
            "visual_lesson removed redundant table column(s): %s",
            ", ".join(table_report.dropped_columns),
        )
    if table_report.dropped_verdict:
        logger.info(
            "visual_lesson dropped untraceable table verdict (%s): %r",
            table_report.verdict_reason,
            table_report.dropped_verdict,
        )
    errors = validate_lesson_spec(spec)
    if errors:
        raise LessonValidationError("; ".join(errors))
    family = select_family(spec)
    t_hero = time.perf_counter()
    hero_svg = render_hero_geometry(spec.deterministic)
    ms_hero = (time.perf_counter() - t_hero) * 1000
    if not hero_svg.strip():
        raise LessonValidationError("hero geometry is empty")
    content = extract_lesson_content(spec, hero_svg=hero_svg)
    t_compose = time.perf_counter()
    source = build_lesson_typst(content)
    warnings: list[str] = []
    try:
        data = compile_typst(source, out_format, extra_files={"hero.svg": hero_svg})
        fallback = False
    except (TypstUnavailable, TypstCompileError) as e:
        warnings.append(str(e))
        logger.warning("visual_lesson falling back to bare hero: %s", e)
        data = hero_svg.encode("utf-8")
        out_format = "svg"
        fallback = True
    ms_compose = (time.perf_counter() - t_compose) * 1000
    return RenderedLesson(
        out_format=out_format,
        data=data,
        hero_kind=family,
        ms_total=(time.perf_counter() - t0) * 1000,
        ms_hero=ms_hero,
        ms_compose=ms_compose,
        warnings=warnings,
        fallback_used=fallback,
    )


def v3_store_mode(v3_used: bool, mode: str) -> str:
    """Render-mode label for a stored visual.

    A v3-composed PNG is deterministic content (Typst from a validated spec),
    not a generative illustration — label it honestly. Everything else keeps
    the legacy labels ("deterministic" for svg, "generative" for png).
    """
    if v3_used and mode == "png":
        return "deterministic"
    return "deterministic" if mode == "svg" else "generative"


async def render_v3_visual(spec: VisualSpec) -> tuple[str, str | bytes] | None:
    """v3 composition path for the Explain Visually route.

    Same return shape as the legacy ``generate_visual``: ``("svg", svg)`` for
    the honest bare-hero fallback, ``("png", bytes)`` for a composed lesson.
    Returns None (like the legacy dispatcher) when the spec fails semantic
    validation or nothing usable was produced, so the caller reports the same
    honest unavailable state. Logs the reason. Touches no credits.
    """
    errors = validate_lesson_spec(spec)
    if errors:
        logger.warning("v3 composition refused (semantic validation): %s", "; ".join(errors))
        return None
    try:
        family = select_family(spec)
    except LessonValidationError as e:
        logger.warning("v3 composition refused (family selection): %s", e)
        return None
    t_start = time.perf_counter()
    try:
        lesson = await asyncio.to_thread(render_visual_lesson, spec, "png")
    except LessonValidationError as e:
        logger.warning("v3 composition refused at render: %s", e)
        return None
    total_ms = (time.perf_counter() - t_start) * 1000
    if lesson.fallback_used:
        logger.info(
            "v3 composition fallback: bare hero svg (family=%s, total=%.1fms hero=%.1f, warnings=%s)",
            family, total_ms, lesson.ms_hero, lesson.warnings,
        )
        return ("svg", lesson.data.decode("utf-8"))
    logger.info(
        "v3 composition rendered (family=%s, %s, %d bytes, total=%.1fms hero=%.1f compose=%.1f)",
        family, lesson.out_format, len(lesson.data), total_ms, lesson.ms_hero, lesson.ms_compose,
    )
    return ("png", lesson.data)
