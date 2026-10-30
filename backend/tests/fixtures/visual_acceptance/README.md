# Real-material visual acceptance fixtures

These JSON files are saved VisualSpec outputs from real lecture screenshots
(`stress_test_fixtures/`, `public/`) and two openly licensed educational
source images (`sources/`). They are committed so the acceptance test can be
rerun without another paid model call, and so layout regressions are
reproducible from the exact planner output that produced each page.

Provenance (Gemini `gemini-flash-lite-latest`, October 2026):

| File | Source screenshot | Planner outcome |
|------|-------------------|-----------------|
| `chemistry_equilibrium.json` | `stress_test_fixtures/8_chemistry_equilibrium.png` | first pass: `comparison` feature matrix |
| `chemistry_equilibrium_repaired.json` | same screenshot, second pass | **repaired** spec: planner's prose claims moved into visible table cells |
| `biology_cell.json` | `stress_test_fixtures/7_biology_prose.png` | first pass: shallow `concept_map` tree |
| `cs_bst.json` | `stress_test_fixtures/9_dark_bst.png` | first pass chose a map; **repaired** spec after claim normalization |
| `engineering_math.json` | `stress_test_fixtures/3_math_formulas.png` | first pass: deep formula tree |
| `stats_distributions.json` | `stress_test_fixtures/1_stats_table.png` | empty comparison payload repaired by the focused table call |
| `physics_torque.json` | `public/rotational one.jpg` | first pass: force diagram + deterministic claim promotion |
| `pid_flow.json` | `stress_test_fixtures/2_flowchart.png` | first pass: process flow |
| `cycle_krebs.json` | `sources/krebs_cycle.jpg` | first pass: 8-stage `concept_map` cycle |
| `timeline_vaccines.json` | `sources/vaccine_timeline.png` | first pass: 6-event `concept_map` timeline |

Claims, symbols and layout choices are the model's exact output (including its
transliterations, e.g. `l` for λ) — the point is to test the renderer and gates
against real material, not to present these as perfect transcriptions. The
expected family/layout assertions are in `tests/test_visual_acceptance.py`.

The two older comparison fixtures predate the verdict-evidence contract. Their
verdicts are generic syntheses that cannot be traced to two cell outcomes, so
the deterministic gate drops them (the matrix and takeaway still render). The
acceptance test asserts that drop rather than displaying an unverified claim.

## Source licensing

- `sources/krebs_cycle.jpg` — “2507 The Krebs Cycle.jpg”, OpenStax College,
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2507_The_Krebs_Cycle.jpg),
  CC BY 3.0. Downloaded at 1400 px width.
- `sources/vaccine_timeline.png` — “Vaccine Evolution Timeline.png”,
  HourMinSec, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vaccine_Evolution_Timeline.png),
  CC0. Downloaded at original size.
