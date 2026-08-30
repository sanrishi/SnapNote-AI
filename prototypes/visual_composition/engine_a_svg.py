"""Engine A: Current SnapNote SVG renderer baseline (no changes)."""
import time
from app.utils.visual_renderer import render_deterministic_visual

def render_a(spec, label: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    svg = render_deterministic_visual(spec.deterministic)
    dt = (time.perf_counter() - t0) * 1000
    return svg, dt
