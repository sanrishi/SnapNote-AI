"""Rich composition benchmark — one coherent infographic per lesson, from ground truth.

SAME semantic JSON (ground_truth_*.json -> ground_truth_specs.py) feeds both:
  A: current SVG baseline
  B: rich Typst (hero SVG + callouts + derivation + result)

Includes semantic validation (ground truth math) + layout validation
(required tokens present, no overlap placeholders). No production touch.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from ground_truth_specs import argand_spec_from_ground_truth, torque_spec_from_ground_truth
from rich_typst import render_rich_typst, TEMPLATE_ARGAND, TEMPLATE_TORQUE
from engine_a_svg import render_a
from validate_ground_truth import main as validate_main

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# A-baseline tokens: what the CURRENT renderer must show (geometry + relation only;
# "Area"/"L" live in study notes, not duplicated inside the visual per product rule).
REQUIRED_TOKENS_A = {
    "torque": ["Torque and Angular Momentum", "O", "r", "F", "θ", "τ", "τ = r × F"],
    "argand": ["Square on Argand Plane", "square ABCD", "A(1,1)", "WHAT THE VISUAL SHOWS"],
}
# B-Typst composition must additionally carry the result + derivation (checked in template source).
REQUIRED_TOKENS_B_TEMPLATE = {
    "torque": ["τ = r × F", "Takeaway"],
    "argand": ["Area = 4", "How it connects"],
}


def layout_validate(svg: str, name: str) -> list[str]:
    """Check the SVG for required tokens and gross overlap markers."""
    problems: list[str] = []
    for tok in REQUIRED_TOKENS_A[name]:
        if tok not in svg:
            problems.append(f"missing token {tok!r}")
    # malformed nested-polygon marker from the old double-wrap bug
    if '<polygon points="<polygon' in svg:
        problems.append("malformed nested polygon")
    texts = re.findall(r"<text[^>]*>(.*?)</text>", svg)
    if len(texts) < 5:
        problems.append(f"only {len(texts)} text elements (expected labels)")
    return problems


def bench():
    print("=== Rich Composition — Ground-Truth-Driven (one coherent visual) ===")
    rc = validate_main()
    if rc != 0:
        print("ABORT: ground truth invalid — refusing to render")
        sys.exit(1)
    for name, builder, tmpl in [
        ("torque", torque_spec_from_ground_truth, TEMPLATE_TORQUE),
        ("argand", argand_spec_from_ground_truth, TEMPLATE_ARGAND),
    ]:
        spec = builder()
        # A: current SVG baseline (same spec)
        svg_a, ms_a = render_a(spec, name)
        open(os.path.join(OUT, f"rich_a_{name}.svg"), "w", encoding="utf-8").write(svg_a)
        probs_a = layout_validate(svg_a, name)
        print(f"A current SVG {name}: {len(svg_a)} chars, {ms_a:.1f}ms, layout={'OK' if not probs_a else probs_a}")
        # B: composition must carry result + derivation in template source (semantic check)
        missing = [t for t in REQUIRED_TOKENS_B_TEMPLATE[name] if t not in tmpl]
        if missing:
            print(f"B template {name} MISSING: {missing}")
        # B: rich Typst PDF (same spec hero + composition)
        data, ms_b, log = render_rich_typst(spec, tmpl, name, "pdf")
        if data:
            open(os.path.join(OUT, f"rich_b_typst_{name}.pdf"), "wb").write(data)
            print(f"B rich Typst {name}: {len(data)} bytes, {ms_b:.1f}ms — {log[:80]}")
        else:
            print(f"B rich Typst {name} FAILED: {log[:300]}")
        # B PNG for direct viewing (Typst rasterizes the same composition)
        png, ms_p, log_p = render_rich_typst(spec, tmpl, name, "png")
        if png:
            open(os.path.join(OUT, f"rich_b_typst_{name}.png"), "wb").write(png)
            print(f"B rich Typst PNG {name}: {len(png)} bytes, {ms_p:.1f}ms")
        else:
            print(f"B rich Typst PNG {name} FAILED: {log_p[:200]}")


if __name__ == "__main__":
    bench()
