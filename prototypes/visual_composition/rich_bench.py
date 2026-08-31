"""Rich composition benchmark — one coherent infographic per lesson, not placeholders."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
from fixtures import ARGAND_SPEC, TORQUE_SPEC
from rich_typst import render_rich_typst, TEMPLATE_ARGAND, TEMPLATE_TORQUE
from engine_a_svg import render_a

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

def bench():
    print("=== Rich Composition — One Coherent Visual per Lesson ===")
    for name, spec, tmpl in [("torque", TORQUE_SPEC, TEMPLATE_TORQUE), ("argand", ARGAND_SPEC, TEMPLATE_ARGAND)]:
        # A: current SVG baseline (same VisualSpec, but without composition)
        svg_a, ms_a = render_a(spec, name)
        open(os.path.join(OUT, f"rich_a_{name}.svg"), "w", encoding="utf-8").write(svg_a)
        print(f"A current SVG {name}: {len(svg_a)} chars, {ms_a:.1f}ms — has <svg { '<svg' in svg_a }")
        # B: rich Typst (hero + callouts + derivation + result) — use PDF for multi-page infographic
        data, ms_b, log = render_rich_typst(spec, tmpl, name, "pdf")
        if data:
            open(os.path.join(OUT, f"rich_b_typst_{name}.pdf"), "wb").write(data)
            print(f"B rich Typst {name}: {len(data)} bytes, {ms_b:.1f}ms — {log[:80]}")
        else:
            print(f"B rich Typst {name} FAILED: {log[:300]}")

if __name__ == "__main__":
    bench()
