"""Engine B: Typst (CLI official first, typst-py binding separately if available)."""
import subprocess
import tempfile
import time
import os

TYPST_VERSION = "0.13 (assumed, check via `typst --version`; official docs 0.15.1 as of 2026-08)"
# Official binary: https://github.com/typst/typst/releases

def _typst_available() -> tuple[bool, str]:
    try:
        r = subprocess.run(["typst", "--version"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0, r.stdout.strip() or r.stderr.strip()
    except FileNotFoundError:
        return False, "not found"
    except Exception as e:
        return False, str(e)

def _typst_binding_available() -> tuple[bool, str]:
    try:
        import typst  # typst-py
        return True, getattr(typst, "__version__", "unknown")
    except ImportError:
        return False, "not installed"
    except Exception as e:
        return False, str(e)

def render_typst_cli(typst_source: str, fmt: str = "svg") -> tuple[bytes | None, float, str]:
    """Compile Typst source via CLI to SVG/PNG bytes. Returns (bytes, ms, log)."""
    ok, ver = _typst_available()
    if not ok:
        return None, 0, f"typst CLI not available: {ver}"
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.typ")
        out = os.path.join(td, f"out.{fmt}")
        open(inp, "w", encoding="utf-8").write(typst_source)
        try:
            r = subprocess.run(["typst", "compile", inp, out, "--format", fmt], capture_output=True, text=True, timeout=20)
            dt = (time.perf_counter() - t0) * 1000
            if r.returncode != 0:
                return None, dt, f"typst compile failed: {r.stderr[:300]}"
            data = open(out, "rb").read() if os.path.exists(out) else None
            return data, dt, f"typst {ver} compile {fmt} {dt:.1f}ms"
        except Exception as e:
            return None, (time.perf_counter() - t0) * 1000, str(e)

def render_typst_binding(typst_source: str) -> tuple[bytes | None, float, str]:
    ok, ver = _typst_binding_available()
    if not ok:
        return None, 0, f"typst-py not available: {ver}"
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        try:
            import typst
            inp = os.path.join(td, "in.typ")
            open(inp, "w", encoding="utf-8").write(typst_source)
            data = typst.compile(inp, format="svg")
            dt = (time.perf_counter() - t0) * 1000
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data, dt, f"typst-py {ver} {dt:.1f}ms"
        except Exception as e:
            return None, (time.perf_counter() - t0) * 1000, f"typst-py failed: {e}"

# Minimal Typst templates for the two fixtures (deterministic, no LLM)
TORQUE_TYPST = r'''
#set page(width: 800pt, height: 900pt, margin: 20pt)
#set text(font: "Inter", size: 11pt)
#align(center)[#text(weight: "bold", size: 16pt)[Torque and Angular Momentum]]
#v(12pt)
#rect(width: 100%, height: 420pt, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 8pt)[
  #place(center)[*Force diagram placeholder — pivot O, vector r, vector F, angle θ, arc τ*]
]
#v(8pt)
#align(left)[*WHAT THE VISUAL SHOWS:* Torque magnitude depends on r, F and the angle between them.]
#v(6pt)
#table(columns: (1fr, 1fr), [θ — angle between r and F], [τ — torque direction])
#v(8pt)
#rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 8pt)[ $ tau = r times F $ — torque = r cross F ]
'''

ARGAND_TYPST = r'''
#set page(width: 800pt, height: 900pt, margin: 20pt)
#set text(font: "Inter", size: 11pt)
#align(center)[#text(weight: "bold", size: 16pt)[Square on Argand Plane]]
#v(12pt)
#rect(width: 100%, height: 380pt, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 8pt)[
  #place(center)[*Argand plot placeholder — axes Re/Im, square ABCD, points (1,1) etc.*]
]
#v(8pt)
#align(left)[*Callouts:* A: z=1+i → (1,1), Side s=2, Area=4]
#v(6pt)
#rect(width: 100%, fill: rgb("#f8fafc"), stroke: rgb("#e2e8f0"), radius: 8pt)[ $ "Area" = s^2 = 4 $ ]
'''

if __name__ == "__main__":
    print("Typst CLI:", _typst_available())
    print("typst-py:", _typst_binding_available())
    for name, src in [("torque", TORQUE_TYPST), ("argand", ARGAND_TYPST)]:
        data, ms, log = render_typst_cli(src, "svg")
        print(f"B CLI {name}: {len(data) if data else 0} bytes, {ms:.1f}ms — {log[:80]}")
        data2, ms2, log2 = render_typst_binding(src)
        print(f"B binding {name}: {len(data2) if data2 else 0} bytes, {ms2:.1f}ms — {log2[:80]}")
