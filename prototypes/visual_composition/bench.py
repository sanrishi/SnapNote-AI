"""Isolated benchmark — 6 outputs, timings cold/warm, deps, quality."""
import time
import os
import sys

# Ensure backend on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

import importlib.util
import pathlib
def _load(name):
    p = pathlib.Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
_fixtures = _load("fixtures")
TORQUE_SPEC = _fixtures.TORQUE_SPEC
ARGAND_SPEC = _fixtures.ARGAND_SPEC
_engine_a = _load("engine_a_svg")
render_a = _engine_a.render_a
_engine_b = _load("engine_b_typst")
_typst_available = _engine_b._typst_available
_typst_binding_available = _engine_b._typst_binding_available
render_typst_cli = _engine_b.render_typst_cli
render_typst_binding = _engine_b.render_typst_binding
TORQUE_TYPST = _engine_b.TORQUE_TYPST
ARGAND_TYPST = _engine_b.ARGAND_TYPST
_engine_c = _load("engine_c_weasyprint")
_weasy_available = _engine_c._weasy_available
render_weasyprint = _engine_c.render_weasyprint

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

def bench():
    print("=== Prototype Battle — Isolated (no master touch) ===")
    print(f"Typst CLI: {_typst_available()}")
    print(f"typst-py: {_typst_binding_available()}")
    print(f"WeasyPrint: {_weasy_available()}")
    print()

    # A: Current SVG
    for name, spec in [("torque", TORQUE_SPEC), ("argand", ARGAND_SPEC)]:
        svg, ms = render_a(spec, name)
        open(os.path.join(OUT, f"a_{name}.svg"), "w", encoding="utf-8").write(svg)
        print(f"A {name}: {len(svg)} chars, {ms:.1f}ms, has <svg { '<svg' in svg }")

    # B: Typst CLI
    for name, src in [("torque", TORQUE_TYPST), ("argand", ARGAND_TYPST)]:
        data, ms, log = render_typst_cli(src, "svg")
        if data:
            open(os.path.join(OUT, f"b_typst_{name}.svg"), "wb").write(data)
        print(f"B CLI {name}: {len(data) if data else 0} bytes, {ms:.1f}ms — {log[:100]}")
        data2, ms2, log2 = render_typst_binding(src)
        if data2:
            open(os.path.join(OUT, f"b_binding_{name}.svg"), "wb").write(data2)
        print(f"B binding {name}: {len(data2) if data2 else 0} bytes, {ms2:.1f}ms — {log2[:100]}")

    # C: WeasyPrint (use hero SVG from A as hero)
    import pathlib
    torque_hero = open(os.path.join(OUT, "a_torque.svg"), encoding="utf-8").read()[:2000] if os.path.exists(os.path.join(OUT, "a_torque.svg")) else "<svg></svg>"
    argand_hero = open(os.path.join(OUT, "a_argand.svg"), encoding="utf-8").read()[:2000] if os.path.exists(os.path.join(OUT, "a_argand.svg")) else "<svg></svg>"
    for name, hero in [("torque", torque_hero), ("argand", argand_hero)]:
        data, ms, log = render_weasyprint(name.title(), hero, ["callout 1", "callout 2"], "Result = 4")
        if data:
            open(os.path.join(OUT, f"c_{name}.png"), "wb").write(data)
        print(f"C weasy {name}: {len(data) if data else 0} bytes, {ms:.1f}ms — {log[:100]}")

    # Cold vs warm for A (run 5 times)
    print("\n--- Cold vs warm (A torque, 5 runs) ---")
    for i in range(5):
        _, ms = render_a(TORQUE_SPEC, "torque")
        print(f"  run {i+1}: {ms:.1f}ms")

    # Deps
    print("\n--- Deps ---")
    try:
        import weasyprint, typst
        print("weasyprint", weasyprint.__version__, "typst-py", getattr(typst, "__version__", "?"))
    except Exception as e:
        print("deps check:", e)

if __name__ == "__main__":
    bench()
