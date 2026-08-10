"""Validate a semantic DiagramSpec BEFORE any rendering.

Purpose: never draw a potentially misleading diagram. If the required geometry
is missing or contradictory, validation fails and the caller reports an honest
"couldn't confidently reconstruct" state instead of a beautiful-but-wrong SVG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.models.schemas import DiagramSpec
from app.utils.math_normalize import math_value, normalize_math

SUPPORTED_DIAGRAM_TYPES = ("polar_region",)
TWO_PI = 2.0 * math.pi


@dataclass
class CanonicalPolarRegion:
    """Fully validated, canonical polar spec — the only input the renderer accepts.

    Pure geometry. Axes and annular shading are ALWAYS drawn for a polar region
    (the spec's present=true already implies a shaded region between the two
    boundaries), and labels/captions are DERIVED from these values at render
    time — never taken from Gemini's free text or presentational toggles. So
    identical geometry always produces byte-identical SVG.
    """

    inner: str
    outer: str
    theta_min: str
    theta_max: str
    inner_value: float
    outer_value: float
    theta_min_value: float
    theta_max_value: float


@dataclass
class ValidationResult:
    valid: bool
    canonical: CanonicalPolarRegion | None
    reasons: list[str]


def _normalize_bounds(spec: DiagramSpec) -> tuple[dict[str, tuple[str, float]], list[str]]:
    reasons: list[str] = []
    out: dict[str, tuple[str, float]] = {}
    for field in ("inner", "outer", "theta_min", "theta_max"):
        raw = getattr(spec.bounds, field)
        canonical = normalize_math(raw)
        if canonical is None:
            reasons.append(f"The {field} bound could not be read confidently.")
            continue
        value = math_value(canonical)
        if value is None:
            reasons.append(f"The {field} bound could not be read confidently.")
            continue
        out[field] = (canonical, value)
    return out, reasons


def validate_diagram_spec(spec: DiagramSpec) -> ValidationResult:
    reasons: list[str] = []

    if not spec.present:
        return ValidationResult(valid=False, canonical=None, reasons=["No diagram was detected in the screenshot."])

    if spec.diagram_type not in SUPPORTED_DIAGRAM_TYPES:
        what = spec.diagram_type or "an unknown"
        reasons.append(f"This diagram type ({what}) is not supported yet, so it could not be reconstructed.")
        return ValidationResult(valid=False, canonical=None, reasons=reasons)

    bounds, bound_reasons = _normalize_bounds(spec)
    reasons.extend(bound_reasons)

    if set(bounds) != {"inner", "outer", "theta_min", "theta_max"}:
        return ValidationResult(valid=False, canonical=None, reasons=reasons)

    inner_c, inner_v = bounds["inner"]
    outer_c, outer_v = bounds["outer"]
    tmin_c, tmin_v = bounds["theta_min"]
    tmax_c, tmax_v = bounds["theta_max"]

    if inner_v < 0:
        reasons.append("The inner radius cannot be negative.")
    if outer_v <= inner_v:
        reasons.append("The outer radius must be greater than the inner radius.")
    if tmin_v < 0:
        reasons.append("The start angle cannot be negative.")
    if tmax_v <= tmin_v:
        reasons.append("The end angle must be greater than the start angle.")
    if tmax_v > TWO_PI + 1e-6:
        reasons.append("The end angle exceeds one full revolution (2π).")

    if reasons:
        return ValidationResult(valid=False, canonical=None, reasons=reasons)

    canonical = CanonicalPolarRegion(
        inner=inner_c,
        outer=outer_c,
        theta_min=tmin_c,
        theta_max=tmax_c,
        inner_value=inner_v,
        outer_value=outer_v,
        theta_min_value=tmin_v,
        theta_max_value=tmax_v,
    )
    return ValidationResult(valid=True, canonical=canonical, reasons=[])
