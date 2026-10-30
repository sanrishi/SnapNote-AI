# Real-material visual acceptance fixtures

These JSON files are saved VisualSpec outputs from the real lecture screenshots
in `stress_test_fixtures/` and `public/`. They are committed so the acceptance
test can be rerun without another paid model call, and so layout regressions are
reproducible from the exact planner output that produced each page.

Provenance (Gemini `gemini-flash-lite-latest`, October 2026):

| File | Source screenshot | Planner outcome |
|------|-------------------|-----------------|
| `chemistry_equilibrium.json` | `stress_test_fixtures/8_chemistry_equilibrium.png` | first pass: `comparison` feature matrix |
| `biology_cell.json` | `stress_test_fixtures/7_biology_prose.png` | first pass: shallow `concept_map` tree |
| `cs_bst.json` | `stress_test_fixtures/9_dark_bst.png` | first pass chose a map; **repaired** spec after claim normalization |
| `engineering_math.json` | `stress_test_fixtures/3_math_formulas.png` | first pass: deep formula tree |
| `stats_distributions.json` | `stress_test_fixtures/1_stats_table.png` | empty comparison payload repaired by the focused table call |
| `physics_torque.json` | `public/rotational one.jpg` | first pass: force diagram + deterministic claim promotion |
| `pid_flow.json` | `stress_test_fixtures/2_flowchart.png` | first pass: process flow |

Claims, symbols and layout choices are the model's exact output (including its
transliterations, e.g. `l` for λ) — the point is to test the renderer and gates
against real material, not to present these as perfect transcriptions. The
expected family/layout assertions are in `tests/test_visual_acceptance.py`.
