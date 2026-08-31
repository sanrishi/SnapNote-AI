"""Rich Typst infographic prototype — one coherent visual per lesson, not placeholders."""
import subprocess
import tempfile
import time
import os

TEMPLATE_ARGAND = r'''
#set page(width: 800pt, height: 1000pt, margin: 18pt)
#set text(font: "Inter", size: 10pt)

#align(center)[#text(size: 18pt, weight: "bold")[Square on Argand Plane]]
#v(6pt)
#align(center)[#text(size: 10pt, fill: rgb("#64748b"))[JEE Complex Numbers — modulus, conjugate, area]]
#v(12pt)

// Hero: Argand axes + square (embedded SVG from current renderer)
#figure(
  image("hero_argand.svg", width: 90%),
  caption: [Argand plane: Re (x) horizontal, Im (y) vertical. Square ABCD with A(1,1), B(1,3), C(3,3), D(3,1).],
) <argand-hero>

#grid(columns: (1fr, 1fr), gutter: 10pt,
  rect(width: 100%, height: 60pt, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt)[
    #text(weight: "bold")[A: z = 1 + i] \ (1, 1) \ Conjugate → reflection across Re
  ],
  rect(width: 100%, height: 60pt, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt)[
    #text(weight: "bold")[B: z = -1 + 3i] \ (-1, 3) \ Mirror of A
  ],
)

#v(8pt)
#rect(width: 100%, fill: rgb("#fef9c3"), stroke: rgb("#facc15"), radius: 6pt, inset: 8pt)[
  #text(weight: "bold")[Derivation:] Side $s = sqrt((3-1)^2 + (1-1)^2) = 2$; Modulus $|z| = sqrt(x^2+y^2)$; $s^2 = 4$.
]

#v(6pt)
#align(center)[#rect(fill: rgb("#facc15"), stroke: none, radius: 6pt, inset: 8pt)[#text(weight: "bold", size: 13pt)[Area = 4]]]

#align(left)[#text(size: 9pt, fill: rgb("#64748b"))[Why this matters: A square’s area is side² — here the side comes from the distance between conjugate-related points.]]
'''

TEMPLATE_TORQUE = r'''
#set page(width: 800pt, height: 900pt, margin: 18pt)
#set text(font: "Inter", size: 10pt)

#align(center)[#text(size: 18pt, weight: "bold")[Torque and Angular Momentum]]
#v(6pt)
#align(center)[#text(size: 10pt, fill: rgb("#64748b"))[Pivot → r → F → θ → τ — the turning effect]]
#v(12pt)

// Hero: force diagram (pivot O, r, F, θ, τ)
#figure(
  image("hero_torque.svg", width: 85%),
  caption: [Pivot O, position vector r, force F, angle θ between them, torque τ out of page.],
) <torque-hero>

#grid(columns: (1fr, 1fr), gutter: 10pt,
  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 8pt)[
    #text(weight: "bold")[r] \ position vector \ from pivot to point of application
  ],
  rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 6pt, inset: 8pt)[
    #text(weight: "bold")[τ = r × F] \ magnitude $r F sin(θ)$
  ],
)

#v(8pt)
#rect(width: 100%, fill: rgb("#fef9c3"), stroke: rgb("#facc15"), radius: 6pt, inset: 8pt)[
  #text(weight: "bold")[Takeaway:] A force far from the pivot (large r) with θ near 90° gives maximal torque — like pushing a door at the handle.
]

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

def render_rich_typst(spec, template: str, name: str, out_fmt: str = "svg") -> tuple[bytes | None, float, str]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        # Generate hero SVG and write it for Typst to embed
        hero_svg = _hero_svg_for_spec(spec)
        hero_path = os.path.join(td, f"hero_{name}.svg")
        open(hero_path, "w", encoding="utf-8").write(hero_svg)
        # Rewrite template to point to the hero file (single replace for this lesson)
        src = template.replace(f"hero_{name}.svg", hero_path)
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
