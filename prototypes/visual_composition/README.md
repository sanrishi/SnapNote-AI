# Visual Composition Prototype — Isolated Battle
**Branch: `feat/prototype-visual-composition` — No master touch, no deploy.**

This directory is a throwaway prototype to compare 3 rendering stacks on the SAME 2 semantic inputs.

- `fixtures.py` — VisualSpec v3 for Argand square + Torque (one source for all 3 engines)
- `engine_a_svg.py` — Current SnapNote SVG renderer baseline
- `engine_b_typst.py` — Typst CLI (official) + typst-py binding benchmark (third-party)
- `engine_c_weasyprint.py` — HTML/CSS + WeasyPrint
- `bench.py` — runs all 6, measures cold/warm, memory, Docker impact, writes `outputs/`

Run: `python prototypes/visual_composition/bench.py`
Outputs: `outputs/a_torque.svg`, `outputs/a_argand.svg`, `outputs/b_typst_torque.svg`, etc.

**Hard constraints:** Do not import production renderer code for B/C except via the shared fixtures (copy-paste allowed). Do not modify `backend/` or `website/`.
