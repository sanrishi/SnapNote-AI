"""Rich Typst infographic prototype — one coherent visual per lesson, not placeholders."""
import subprocess
import tempfile
import time
import os

TEMPLATE_ARGAND = r'''
#set page(width: 800pt, height: 1050pt, margin: 18pt)
#set text(size: 10pt)

#align(center)[#text(size: 18pt, weight: "bold")[Square on Argand Plane]]
#v(4pt)
#align(center)[#text(size: 10pt, fill: rgb("#64748b"))[Each complex number is a point: $z = x + i y arrow.r (x, y)$]]
#v(10pt)

// Hero: Argand axes + square (embedded SVG from current renderer)
#figure(
  image("hero_argand.svg", width: 88%),
  caption: [Square ABCD with A(1,1), B(1,3), C(3,3), D(3,1) on Re/Im axes.],
) <argand-hero>

#v(6pt)
// Points strip: all four corners in ONE row, spatially ordered A→B→C→D
#grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 8pt,
  rect(width: 100%, fill: rgb("#ede9fe"), stroke: rgb("#6366f1"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[A] \ $z=1+i$ \ (1,1)]],
  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[B] \ $z=1+3i$ \ (1,3)]],
  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[C] \ $z=3+3i$ \ (3,3)]],
  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[D] \ $z=3+i$ \ (3,1)]],
)

#v(8pt)
// Reasoning chain: conjugate → side → modulus → area, flowing with arrows
#rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 8pt)[
  #text(weight: "bold")[How it connects:] conj(A) $= 1-i$ mirrors across Re $arrow.r$ reflection; side $s = |B-A| = |2i| = 2$; modulus $|1+i| = sqrt(2)$; therefore $s^2 = 4$.
]

#v(6pt)
#align(center)[#rect(fill: rgb("#facc15"), stroke: none, radius: 6pt, inset: 8pt)[#text(weight: "bold", size: 13pt)[Area = 4]]]

#align(left)[#text(size: 9pt, fill: rgb("#64748b"))[Takeaway: a square’s area is side² — here side s = |B − A| = 2, so Area = 4.]]
'''

TEMPLATE_TORQUE = r'''
#set page(width: 800pt, height: 900pt, margin: 18pt)
#set text(size: 10pt)

#align(center)[#text(size: 18pt, weight: "bold")[Torque and Angular Momentum]]
#v(6pt)
#align(center)[#text(size: 10pt, fill: rgb("#64748b"))[Pivot → r → F → θ → τ — the turning effect]]
#v(12pt)

// Hero: force diagram (pivot O, r, F, θ, τ)
#figure(
  image("hero_torque.svg", width: 85%),
  caption: [Pivot O, position vector r, force F, angle θ between them, torque τ out of page.],
) <torque-hero>

#grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 8pt,
  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[O] \ pivot]],
  rect(width: 100%, fill: rgb("#ede9fe"), stroke: rgb("#6366f1"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[r] \ 55° lever]],
  rect(width: 100%, fill: rgb("#fef2f2"), stroke: rgb("#dc2626"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[F] \ 90° force]],
  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 6pt)[#align(center)[#text(weight: "bold")[θ = 35°] \ between r, F]],
)

#v(8pt)
#rect(width: 100%, fill: rgb("#fef9c3"), stroke: rgb("#facc15"), radius: 6pt, inset: 8pt)[
  #text(weight: "bold")[Reasoning:] $theta = 90° - 55° = 35°$; torque $tau = r times F$, magnitude $r F sin(theta)$; direction by right-hand rule (out of page).
]

#v(6pt)
#align(center)[#rect(fill: rgb("#facc15"), stroke: none, radius: 6pt, inset: 8pt)[#text(weight: "bold", size: 13pt)[τ = r × F]]]

#align(left)[#text(size: 9pt, fill: rgb("#64748b"))[Takeaway: a force far from the pivot (large r) with θ near 90° gives maximal torque — like pushing a door at the handle.]]

#align(left)[#text(size: 9pt, fill: rgb("#64748b"))[Tip: If θ = 0° (push along r), sinθ = 0 → no turning.]]
'''

def _hero_svg_for_spec(spec) -> str:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
    from app.utils.visual_renderer import render_deterministic_visual
    svg = render_deterministic_visual(spec.deterministic)
    # Extract just the hero stage SVG (the first <svg> is the whole visual; for hero we want the stage part
    # For the prototype, we embed the whole visual as hero — the composition will add callouts around it.
    return svg

def render_rich_typst(spec, template: str, name: str, out_fmt: str = "pdf") -> tuple[bytes | None, float, str]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        hero_svg = _hero_svg_for_spec(spec)
        hero_path = os.path.join(td, f"hero_{name}.svg")
        open(hero_path, "w", encoding="utf-8").write(hero_svg)
        src = template
        inp = os.path.join(td, "in.typ")
        out = os.path.join(td, f"out.{out_fmt}")
        open(inp, "w", encoding="utf-8").write(src)
        try:
            r = subprocess.run(["typst", "compile", inp, out, "--format", out_fmt], capture_output=True, text=True, timeout=20)
            dt = (time.perf_counter() - t0) * 1000
            if r.returncode != 0:
                return None, dt, f"typst failed: {r.stderr[:500]}"
            data = open(out, "rb").read() if os.path.exists(out) else None
            return data, dt, f"typst {out_fmt} {dt:.1f}ms"
        except Exception as e:
            return None, (time.perf_counter() - t0) * 1000, str(e)


def render_typst_source(source: str, out_fmt: str = "svg") -> tuple[bytes | None, float, str]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.typ")
        out = os.path.join(td, f"out.{out_fmt}")
        open(inp, "w", encoding="utf-8").write(source)
        try:
            r = subprocess.run(["typst", "compile", inp, out, "--format", out_fmt], capture_output=True, text=True, timeout=20)
            dt = (time.perf_counter() - t0) * 1000
            if r.returncode != 0:
                return None, dt, f"typst failed: {r.stderr[:300]}"
            data = open(out, "rb").read() if os.path.exists(out) else None
            return data, dt, f"typst {out_fmt} {dt:.1f}ms"
        except Exception as e:
            return None, (time.perf_counter() - t0) * 1000, str(e)

if __name__ == "__main__":
    for name, src in [("torque", TEMPLATE_TORQUE), ("argand", TEMPLATE_ARGAND)]:
        data, ms, log = render_typst_source(src, "svg")
        print(f"Rich Typst {name}: {len(data) if data else 0} bytes, {ms:.1f}ms — {log[:80]}")
