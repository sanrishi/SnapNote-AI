"""Trust-text grounding for the Explain Visually context line.

The "Why this visual matters" summary is generated from the SOURCE screenshot,
but it is displayed underneath OUR rebuilt visual. Any sentence that describes
a structural element our scene does not actually draw (shaded regions, dashed
drop-lines, arrows that do not exist) is an overclaim and must be removed.

Source of truth: the emitted VisualSpec deterministic scene inventory.
This module never touches geometry, prompts, or rendering.
"""
import re

from app.models.schemas import VisualSpec


def scene_inventory(spec: VisualSpec) -> dict:
    """Extract the set of structural elements actually present in the scene."""
    inv: dict = {
        "scene_kind": "",
        "labels": set(),
        "has_arrows": False,
        "has_dashed": False,
        "has_region_fill": False,  # renderer draws no fills (verified)
        "has_grid": False,
        "has_axes": False,
    }
    scene = spec.deterministic.scene
    if scene is None:
        return inv
    inv["scene_kind"] = scene.scene_kind
    if scene.force is not None:
        force = scene.force
        inv["labels"].add(force.object.label)
        for vector in force.vectors:
            inv["labels"].add(vector.label)
        for angle in force.angles:
            inv["labels"].add(angle.label)
        for arc in force.arcs:
            inv["labels"].add(arc.label)
        if force.vectors or force.arcs:
            inv["has_arrows"] = True
    if scene.flow is not None:
        for node in scene.flow.nodes:
            inv["labels"].add(node.label)
        if scene.flow.connectors:
            inv["has_arrows"] = True
        if any(connector.feedback for connector in scene.flow.connectors):
            inv["has_dashed"] = True  # feedback connectors render dashed
    if scene.plot is not None:
        plot = scene.plot
        inv["has_axes"] = True
        inv["has_grid"] = bool(plot.show_grid)
        inv["labels"].add(plot.x_label)
        inv["labels"].add(plot.y_label)
        for curve in plot.curves:
            inv["labels"].add(curve.label)
            inv["labels"].add(curve.expr)
            if curve.style == "dashed":
                inv["has_dashed"] = True
    if scene.generic is not None:
        inv["labels"].add(scene.generic.central_label)
        inv["labels"].update(scene.generic.callouts)
    inv["labels"] = {label for label in inv["labels"] if label}
    return inv


# sentence -> required inventory flag when the pattern matches
_STRUCTURE_RULES: tuple[tuple[str, str], ...] = (
    (r"\bshad\w*|\bfill\w*\s+(region|area)\b|\b(region|area)\b.{0,20}\bfill\w*", "has_region_fill"),
    (r"\bdashed?\b", "has_dashed"),
    (r"\barrows?\b", "has_arrows"),
    (r"\bx-axis\b|\by-axis\b|\baxes\b", "has_axes"),
)


def _sentence_grounded(sentence: str, inv: dict) -> bool:
    lowered = sentence.lower()
    for pattern, flag in _STRUCTURE_RULES:
        if re.search(pattern, lowered) and not inv.get(flag):
            return False
    # "dotted" is only true as plot gridlines — never as data lines.
    if re.search(r"\bdotted\b", lowered) and not (inv.get("has_grid") or inv.get("has_dashed")):
        return False
    return True


def ground_visual_context(summary: str, spec: VisualSpec) -> str:
    """Remove sentences that claim structural elements absent from the scene.

    Returns the grounded summary (possibly empty). Empty means: hide the
    context line rather than show an ungrounded claim.
    """
    if not summary or not summary.strip():
        return ""
    inv = scene_inventory(spec)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary.strip()) if s.strip()]
    kept = [s for s in sentences if _sentence_grounded(s, inv)]
    return " ".join(kept)
