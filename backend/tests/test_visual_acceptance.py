"""Acceptance benchmark over real lecture screenshots.

Each fixture is the planner's saved output for a real material (see the
directory README). The test runs the same deterministic gate the route runs,
then composes the lesson and renders it. It asserts the family/layout choice,
not pixel identity, so legitimate typography changes do not fail the suite.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.models.schemas import VisualSpec
from app.utils.table_quality import normalize_visual_spec
from app.utils.visual_lesson import (
    concept_map_is_horizontal,
    extract_lesson_content,
    promote_equation_claims,
    render_visual_lesson,
    select_family,
    validate_legacy_visual,
)
from app.utils.visual_renderer import render_hero_geometry

FIXTURES = Path(__file__).parent / "fixtures" / "visual_acceptance"

CASES = [
    ("chemistry_equilibrium.json", "comparison", "compare"),
    ("chemistry_equilibrium_repaired.json", "comparison", "compare"),
    ("biology_cell.json", "concept_map", "map_tree"),
    ("cs_bst.json", "concept_map", "map_tree"),
    ("engineering_math.json", "concept_map", "map_tree"),
    ("stats_distributions.json", "comparison", "compare"),
    ("physics_torque.json", "vector", "annotated"),
    ("pid_flow.json", "flow", "band"),
    ("cycle_krebs.json", "concept_map", "map_cycle"),
    ("timeline_vaccines.json", "concept_map", "map_timeline"),
]


def _load(name: str) -> VisualSpec:
    return VisualSpec.model_validate_json(
        (FIXTURES / name).read_text(encoding="utf-8")
    )


def _accepted(name: str) -> tuple[VisualSpec, object]:
    spec = _load(name)
    # Apply the same deterministic quality gate the route applies before
    # validation/composition. A legacy fixture with an unverifiable verdict
    # must be tested as the product would render it (verdict dropped).
    spec, _ = normalize_visual_spec(spec)
    errors = validate_legacy_visual(spec)
    promote_equation_claims(spec, errors)
    errors = validate_legacy_visual(spec)
    assert errors == [], f"{name}: {errors}"
    hero = render_hero_geometry(spec.deterministic)
    content = extract_lesson_content(spec, hero_svg=hero)
    return spec, content


@pytest.mark.parametrize("name,family,layout", CASES)
def test_real_material_is_accepted_with_the_expected_layout(
    name: str, family: str, layout: str
) -> None:
    spec, content = _accepted(name)
    assert select_family(spec) == family
    assert content.layout == layout
    assert content.title.strip()
    assert content.hero_svg.strip()


@pytest.mark.parametrize("name,family,layout", CASES)
def test_real_material_composes_to_a_png(
    name: str, family: str, layout: str
) -> None:
    if shutil.which("typst") is None:
        pytest.skip("typst CLI not installed")
    spec, _ = _accepted(name)
    lesson = render_visual_lesson(spec, out_format="png")
    assert lesson.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert lesson.fallback_used is False
    assert len(lesson.data) > 50_000


def test_chemistry_matrix_is_a_real_feature_table() -> None:
    spec, content = _accepted("chemistry_equilibrium.json")
    table = spec.deterministic.scene.comparison.table  # type: ignore[union-attr]
    assert table is not None
    assert len(table.columns) == 2
    assert {column.label for column in table.columns} == {"Qc < Kc", "Qc > Kc"}
    assert [row.label for row in table.rows] == ["Condition", "Reaction Direction"]
    # The saved spec predates verdict_basis and its generic synthesis cannot
    # be traced to two cell outcomes, so the gate drops it. The correct
    # equilibrium condition from the composition remains as the key result.
    assert table.verdict == ""
    assert content.result_text == "Qc = Kc"
    assert "equilibrium" in content.takeaway.casefold()


def test_legacy_unverifiable_verdict_is_dropped_not_displayed() -> None:
    spec = _load("chemistry_equilibrium.json")
    gated, report = normalize_visual_spec(spec)
    assert report.dropped_verdict
    assert "does not state outcomes" in report.verdict_reason
    table = gated.deterministic.scene.comparison.table  # type: ignore[union-attr]
    assert table is not None and table.verdict == ""


def test_repaired_chemistry_spec_puts_each_claim_into_a_visible_cell() -> None:
    # Second planner pass + one bounded repair (real Gemini output): the
    # prose claims that failed the exact gate are now table cells, and the
    # table renders them verbatim. corner="STATE" is inferred as
    # feature-rows (columns are the alternatives).
    spec, _ = _accepted("chemistry_equilibrium_repaired.json")
    table = spec.deterministic.scene.comparison.table  # type: ignore[union-attr]
    assert table is not None
    # The planner wrote corner="STATE" over rows Condition/Direction; the
    # deterministic gate corrects it to the truthful "FEATURE".
    assert table.corner == "FEATURE"
    assert table.verdict == ""
    values = [value for column in table.columns for value in column.values]
    assert "reaction proceeds forward" in values
    assert "reaction proceeds backward" in values


def test_stats_matrix_is_transposed_for_four_alternatives() -> None:
    spec, _ = _accepted("stats_distributions.json")
    table = spec.deterministic.scene.comparison.table  # type: ignore[union-attr]
    assert table is not None
    assert table.corner == "DISTRIBUTION"
    assert len(table.rows) == 4
    assert len(table.columns) == 3
    assert [row.label for row in table.rows][0].startswith("Binomial")
    assert [column.label for column in table.columns] == [
        "Mean (u)", "Variance (s²)", "MGF"
    ]


def test_krebs_cycle_renders_every_stage_and_its_teaching_detail() -> None:
    spec, content = _accepted("cycle_krebs.json")
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    assert cmap is not None and cmap.layout == "cycle"
    labels = [node.label for node in cmap.nodes]
    assert len(labels) == 8
    for label in labels:
        assert label in content.hero_svg
    # Cycle heroes draw labels only; the details must survive as composed cards.
    assert content.callouts
    assert any("acetyl CoA" in callout.body for callout in content.callouts)
    assert any("FADH2" in callout.body for callout in content.callouts)


def test_vaccine_timeline_renders_six_dated_events_in_order() -> None:
    spec, content = _accepted("timeline_vaccines.json")
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    assert cmap is not None and cmap.layout == "timeline"
    assert [node.marker for node in cmap.nodes] == [
        "1796", "1974", "1995", "1998", "2006", "2020"
    ]
    for marker in ("1796", "2020"):
        assert marker in content.hero_svg
    for label in ("Smallpox Vaccine", "COVID-19 Vaccine"):
        assert label in content.hero_svg
    # Timeline heroes print details inside the cards; repeating them as chips
    # would duplicate the page verbatim.
    assert content.callouts == []


def test_biology_tree_stays_horizontal_and_does_not_repeat_details() -> None:
    spec, content = _accepted("biology_cell.json")
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    assert cmap is not None
    assert concept_map_is_horizontal(cmap) is True
    assert content.callouts == []
    assert "Cell Theory" in content.hero_svg
    assert "All organisms are composed of cells" in content.hero_svg


def test_cs_tree_keeps_the_hierarchy_and_avoids_duplicate_callouts() -> None:
    spec, content = _accepted("cs_bst.json")
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    assert cmap is not None
    assert concept_map_is_horizontal(cmap) is True
    assert content.callouts == []
    assert "Worst-case" in content.hero_svg
    assert "O(n) for skewed tree" in content.hero_svg


def test_deep_formula_tree_prints_leaf_details_as_cards() -> None:
    spec, content = _accepted("engineering_math.json")
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    assert cmap is not None
    assert concept_map_is_horizontal(cmap) is False
    bodies = {callout.body for callout in content.callouts}
    assert "x = (-b ± √(b² - 4ac)) / (2a)" in bodies
    assert "Σ i = n(n+1)/2" in bodies
    assert "F(ω) = ∫ f(t) e^(-jωt) dt" in bodies
    assert "Core Analytical Tools" in content.hero_svg


def test_physics_scene_keeps_grounded_vectors_and_promoted_claim() -> None:
    spec, content = _accepted("physics_torque.json")
    force = spec.deterministic.scene.force  # type: ignore[union-attr]
    assert force is not None
    assert {"r", "F"} <= {vector.label for vector in force.vectors}
    assert force.relation is not None
    assert force.relation.expression == "τ = r × F"
    assert any("ΔL" in callout.heading for callout in content.callouts)


def test_pid_flow_keeps_all_parallel_branch_connectors() -> None:
    spec, _ = _accepted("pid_flow.json")
    flow = spec.deterministic.scene.flow  # type: ignore[union-attr]
    assert flow is not None
    assert len(flow.nodes) == 8
    pairs = {(c.source, c.target) for c in flow.connectors}
    assert {(1, 2), (1, 3), (1, 4)} <= pairs
    assert {(2, 5), (3, 5), (4, 5)} <= pairs


def test_fixture_specs_are_json_objects() -> None:
    for name, _family, _layout in CASES:
        raw = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert raw["render_mode"] == "deterministic"
