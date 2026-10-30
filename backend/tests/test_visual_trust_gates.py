"""Trust gates: redundant table axes, verdict evidence, exact visibility.

These tests protect the three quality defects found in the first adaptive
release: a live chemistry matrix spent a column repeating its row labels, a
table verdict was accepted merely because it was grounded somewhere, and the
validator counted node details that the rendered hero had truncated.
"""

from __future__ import annotations

from app.models.schemas import (
    ComparisonColumn,
    ComparisonRow,
    ComparisonTable,
    ComparisonVerdictBasis,
    CompositionResult,
    DeterministicVisual,
    LessonComposition,
    MapEdge,
    MapNode,
    VisualComparison,
    VisualConceptMap,
    VisualScene,
    VisualSpec,
    VisualTask,
)
from app.utils.table_quality import (
    normalize_comparison_table,
    normalize_visual_spec,
    redundant_column_indices,
    verdict_issue,
)
from app.utils.visual_lesson import validate_lesson_spec
from app.utils.visual_renderer import (
    render_hero_geometry,
    rendered_scene_text,
    svg_text_content,
)


def _task(kind: str, families: list[str], claims: list[str]) -> VisualTask:
    return VisualTask(
        task_type=kind,
        required_elements=[],
        exact_claims=claims,
        acceptable_families=families,
    )


def _table_spec(table: ComparisonTable, claims: list[str] | None = None) -> VisualSpec:
    return VisualSpec(
        concept="Quality gate",
        deterministic=DeterministicVisual(
            title="Comparison",
            task=_task("compare_variants", ["comparison"], claims or []),
            scene=VisualScene(
                scene_kind="comparison",
                comparison=VisualComparison(table=table),
            ),
        ),
    )


def _table(
    rows: list[str],
    columns: list[tuple[str, list[str]]],
    verdict: str = "",
    basis: list[tuple[str, str]] | None = None,
) -> ComparisonTable:
    return ComparisonTable(
        rows=[ComparisonRow(label=row) for row in rows],
        columns=[ComparisonColumn(label=label, values=values) for label, values in columns],
        verdict=verdict,
        verdict_basis=[
            ComparisonVerdictBasis(column=column, row=row) for column, row in (basis or [])
        ],
    )


# ── redundant dimensions ─────────────────────────────────────────────────────


def test_redundant_axis_is_dropped_when_real_dimensions_survive() -> None:
    table = _table(
        ["Qc < Kc", "Qc > Kc"],
        [
            ("Condition", ["Qc < Kc", "Qc > Kc"]),  # repeats the row labels
            ("Shift", ["forward", "backward"]),
            ("Products", ["more products", "more reactants"]),
        ],
    )
    normalized, report = normalize_comparison_table(table)
    assert report.dropped_columns == ["Condition"]
    assert [c.label for c in normalized.columns] == ["Shift", "Products"]
    assert redundant_column_indices(table) == {0}
    spec = _table_spec(table)
    assert validate_lesson_spec(spec) == []  # the gate repairs before it renders


def test_redundant_axis_is_refused_when_only_one_real_dimension_remains() -> None:
    table = _table(
        ["Qc < Kc", "Qc > Kc"],
        [
            ("Condition", ["Qc < Kc", "Qc > Kc"]),
            ("Shift", ["forward", "backward"]),
        ],
    )
    normalized, report = normalize_comparison_table(table)
    # The gate does not invent a second dimension: dropping the duplicate
    # would leave a one-dimensional list. Validation must refuse it instead.
    assert report.dropped_columns == []
    assert [c.label for c in normalized.columns] == ["Condition", "Shift"]
    spec = _table_spec(table)
    errors = validate_lesson_spec(spec)
    assert any("repeat the row labels" in error for error in errors)


def test_duplicate_columns_are_dropped_once() -> None:
    table = _table(
        ["Mean", "Variance"],
        [
            ("Binomial", ["np", "np(1-p)"]),
            ("Binomial", ["np", "np(1-p)"]),
            ("Poisson", ["l", "l"]),
        ],
    )
    normalized, report = normalize_comparison_table(table)
    assert report.dropped_columns == ["Binomial"]
    assert [c.label for c in normalized.columns] == ["Binomial", "Poisson"]


# ── verdict evidence ─────────────────────────────────────────────────────────


def test_verdict_with_evidence_cells_is_kept() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="Qc < Kc shifts forward; Qc > Kc shifts backward.",
        basis=[("Qc < Kc", "Direction"), ("Qc > Kc", "Direction")],
    )
    assert verdict_issue(table) is None


def test_verdict_without_evidence_must_name_two_alternatives() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="The system always shifts toward products.",
    )
    assert verdict_issue(table) is not None
    normalized, report = normalize_comparison_table(table)
    assert normalized.verdict == ""
    assert report.dropped_verdict


def test_verdict_cannot_introduce_a_number_absent_from_the_table() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="At 500 K, Qc < Kc shifts forward and Qc > Kc shifts backward.",
        basis=[("Qc < Kc", "Direction"), ("Qc > Kc", "Direction")],
    )
    issue = verdict_issue(table)
    assert issue is not None and "500" in issue
    normalized, _ = normalize_comparison_table(table)
    assert normalized.verdict == ""


def test_verdict_cannot_restate_one_cell() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="excess reactants",
    )
    assert verdict_issue(table) is not None


def test_verdict_evidence_must_reference_real_nonempty_cells() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="Qc < Kc shifts forward; Qc > Kc shifts backward.",
        basis=[("Qc < Kc", "Direction"), ("Qc < Kc", "Missing row")],
    )
    issue = verdict_issue(table)
    assert issue is not None and "unknown cell" in issue


def test_verdict_cannot_assert_the_opposite_of_its_cited_cell() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="Qc < Kc shifts backward; Qc > Kc shifts forward.",
        basis=[("Qc < Kc", "Direction"), ("Qc > Kc", "Direction")],
    )
    issue = verdict_issue(table)
    assert issue is not None and "does not state the cited outcome" in issue
    normalized, report = normalize_comparison_table(table)
    assert normalized.verdict == ""
    assert "does not state the cited outcome" in report.verdict_reason


def test_verdict_evidence_cannot_span_only_one_alternative() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="Qc < Kc shifts forward and also has excess reactants.",
        basis=[("Qc < Kc", "Direction"), ("Qc < Kc", "Condition")],
    )
    issue = verdict_issue(table)
    assert issue is not None and "two alternatives (columns)" in issue


def test_verdict_without_basis_must_state_real_outcomes_not_just_labels() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="Qc < Kc and Qc > Kc are the two conditions.",
    )
    issue = verdict_issue(table)
    assert issue is not None


def test_verdict_without_basis_is_kept_when_it_states_cell_outcomes() -> None:
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="Qc < Kc shifts forward, while Qc > Kc shifts backward.",
    )
    assert verdict_issue(table) is None


def test_row_oriented_verdict_is_verified_against_row_alternatives() -> None:
    table = _table(
        ["Binomial", "Poisson"],
        [("Mean", ["np", "l"]), ("Variance", ["np(1-p)", "l"])],
        verdict="Binomial has mean np, while Poisson has mean l.",
    )
    table.corner = "DISTRIBUTION"
    assert verdict_issue(table) is None

    table.verdict = "Mean is np and Variance is np(1-p)."
    issue = verdict_issue(table)
    assert issue is not None and "rows axis" in issue


def test_mislabelled_corner_is_inferred_from_feature_row_labels() -> None:
    # Live planner output (Oct 2026) used corner="STATE" over rows
    # Condition/Direction: the columns are the alternatives, not the rows.
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
        verdict="Qc < Kc shifts forward, while Qc > Kc shifts backward.",
    )
    table.corner = "STATE"
    assert verdict_issue(table) is None

    normalized, report = normalize_comparison_table(table)
    assert normalized.corner == "FEATURE"
    assert report.corrected_corner == "STATE"


def test_one_feature_like_alternative_does_not_flip_the_axis() -> None:
    # Newton/Secant are the alternatives (rows); Convergence/Derivative needed
    # are shared features. The word "rate" in one feature label must not flip
    # the axis on its own.
    table = _table(
        ["Newton method", "Secant method"],
        [
            ("Rate of convergence", ["quadratic", "superlinear"]),
            ("Derivative needed", ["yes", "no"]),
        ],
        verdict="Newton method converges quadratically, while Secant method converges superlinearly.",
    )
    table.corner = "METHOD"
    assert verdict_issue(table) is None


def test_table_claim_printed_as_composition_result_is_visible() -> None:
    """A table's KEY RESULT is the table verdict; a claim printed there is
    visible. Motion-panel scenes stay strict (claims must be on the panels)."""
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", ["excess reactants", "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
        ],
    )
    spec = _table_spec(table, ["Qc = Kc"])
    spec.deterministic.composition = LessonComposition(
        title="Reaction direction",
        result=CompositionResult(expression="Qc = Kc"),
    )
    assert validate_lesson_spec(spec) == []


# ── exact visibility ────────────────────────────────────────────────────────


def _deep_tree(detail: str, claim: str) -> VisualSpec:
    return VisualSpec(
        concept="Visibility",
        deterministic=DeterministicVisual(
            title="Deep tree",
            task=_task("hierarchy_structure", ["concept_map"], [claim]),
            scene=VisualScene(
                scene_kind="concept_map",
                concept_map=VisualConceptMap(
                    layout="tree",
                    nodes=[
                        MapNode(id="root", label="Root", detail=detail),
                        MapNode(id="branch", label="Branch", parent="root"),
                        MapNode(id="leaf", label="Leaf", parent="branch"),
                    ],
                ),
            ),
        ),
    )


def test_truncated_hero_detail_cannot_satisfy_a_claim() -> None:
    detail = (
        "Opening clause stays readable in the node box, yet the hidden final "
        "exact value x = 42"
    )
    spec = _deep_tree(detail, "x = 42")
    hero = render_hero_geometry(spec.deterministic)
    visible = svg_text_content(hero)
    assert "x = 42" not in visible
    assert "\u2026" in visible
    errors = validate_lesson_spec(spec)
    assert any("exact claim missing" in error for error in errors)


def test_cycle_edge_label_is_not_counted_until_it_is_drawn() -> None:
    spec = VisualSpec(
        concept="Cycle",
        deterministic=DeterministicVisual(
            title="Cycle",
            task=_task("cyclic_process", ["concept_map"], ["ATP yield"]),
            scene=VisualScene(
                scene_kind="concept_map",
                concept_map=VisualConceptMap(
                    layout="cycle",
                    nodes=[
                        MapNode(id="a", label="Stage A"),
                        MapNode(id="b", label="Stage B"),
                        MapNode(id="c", label="Stage C"),
                    ],
                    # A prose-only edge label is echo-filtered from the
                    # reasoning block and the cycle hero draws no edge labels,
                    # so nothing on the page prints it.
                    edges=[MapEdge(source="a", target="b", label="ATP yield")],
                ),
            ),
        ),
    )
    hero_text = rendered_scene_text(spec.deterministic)
    assert "ATP yield" not in hero_text
    # The claim must fail rather than be silently counted from the spec.
    errors = validate_lesson_spec(spec)
    assert any("exact claim missing" in error for error in errors)


def test_normalizing_a_spec_is_idempotent_and_leaves_other_families_alone() -> None:
    spec = _deep_tree("short detail", "Root")
    gated, report = normalize_visual_spec(spec)
    assert gated is spec
    assert not report.changed


def test_truncated_table_cell_cannot_satisfy_a_claim() -> None:
    cell = (
        "This cell contains a long explanation that ends with the exact value "
        "TARGET42"
    )
    assert len(cell) <= 80
    table = _table(
        ["Condition", "Direction"],
        [
            ("Qc < Kc", [cell, "forward"]),
            ("Qc > Kc", ["excess products", "backward"]),
            ("Qc = Kc", ["at equilibrium", "no net shift"]),
        ],
    )
    spec = _table_spec(table, ["TARGET42"])
    hero_text = rendered_scene_text(spec.deterministic)
    assert "TARGET42" not in hero_text
    assert "\u2026" in hero_text
    errors = validate_lesson_spec(spec)
    assert any("exact claim missing" in error for error in errors)


def test_detail_beyond_the_card_cap_cannot_satisfy_a_claim() -> None:
    nodes = [
        MapNode(id="root", label="Root"),
        MapNode(id="b1", label="Branch 1", parent="root"),
        MapNode(id="b2", label="Branch 2", parent="root"),
        *[
            MapNode(
                id=f"l{i}",
                label=f"Leaf {i}",
                parent="b1" if i < 5 else "b2",
                detail="detail HIDDEN42" if i == 8 else f"detail {i}",
            )
            for i in range(9)
        ],
    ]
    spec = VisualSpec(
        concept="Taxonomy",
        deterministic=DeterministicVisual(
            title="Taxonomy",
            task=_task("hierarchy_structure", ["concept_map"], ["HIDDEN42"]),
            scene=VisualScene(
                scene_kind="concept_map",
                concept_map=VisualConceptMap(layout="tree", nodes=nodes),
            ),
        ),
    )
    hero_text = rendered_scene_text(spec.deterministic)
    assert "HIDDEN42" not in hero_text
    errors = validate_lesson_spec(spec)
    assert any("exact claim missing" in error for error in errors)
