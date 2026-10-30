"""Adaptive visual grammars: hierarchies, cycles, timelines, feature matrices.

These tests cover the information structures that used to collapse into the
generic card scene: a shallow taxonomy must render horizontally with its
details visible, a deep tree must stay vertically layered without collisions,
a feature matrix must print every cell verbatim, and the composed lesson must
not repeat content the hero already shows.
"""

from __future__ import annotations

import html
import re
import shutil
from typing import Iterator

import pytest

from app.models.schemas import (
    ComparisonColumn,
    ComparisonRow,
    ComparisonTable,
    CompositionCallout,
    CompositionReasoningStep,
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
from app.utils.visual_lesson import (
    LessonCallout,
    LessonContent,
    build_lesson_typst,
    concept_map_is_horizontal,
    extract_lesson_content,
    select_family,
    should_use_v3,
    validate_lesson_spec,
)
from app.utils.visual_renderer import (
    render_deterministic_visual,
    render_hero_geometry,
)


def _task(kind: str, families: list[str], claims: list[str]) -> VisualTask:
    return VisualTask(
        task_type=kind,
        learning_outcome="",
        required_elements=[],
        required_relationships=[],
        exact_claims=claims,
        acceptable_families=families,
        assumptions=[],
        missing=[],
    )


def _shallow_cell_tree() -> VisualSpec:
    return VisualSpec(
        concept="Cell types",
        deterministic=DeterministicVisual(
            title="Types of Cells",
            task=_task(
                "hierarchy_structure",
                ["concept_map"],
                ["Prokaryotic", "Eukaryotic", "Archaeal"],
            ),
            scene=VisualScene(
                scene_kind="concept_map",
                caption="Cells are classified by nuclear structure.",
                concept_map=VisualConceptMap(
                    layout="tree",
                    nodes=[
                        MapNode(id="cell", label="Cell", detail="basic unit of life"),
                        MapNode(
                            id="prok", label="Prokaryotic", parent="cell",
                            detail="no true nucleus",
                        ),
                        MapNode(
                            id="euk", label="Eukaryotic", parent="cell",
                            detail="membrane-bound nucleus",
                        ),
                        MapNode(
                            id="arch", label="Archaeal", parent="cell",
                            detail="distinct membrane lipids",
                        ),
                    ],
                    edges=[
                        MapEdge(source="cell", target="prok", label="type"),
                        MapEdge(source="cell", target="euk", label="type"),
                        MapEdge(source="cell", target="arch", label="type"),
                    ],
                ),
            ),
        ),
    )


def _deep_bst_tree() -> VisualSpec:
    return VisualSpec(
        concept="Binary search tree",
        deterministic=DeterministicVisual(
            title="Binary Search Tree",
            task=_task("hierarchy_structure", ["concept_map"], ["8", "3", "10"]),
            scene=VisualScene(
                scene_kind="concept_map",
                caption="Keys smaller than a node go left; larger keys go right.",
                concept_map=VisualConceptMap(
                    layout="tree",
                    nodes=[
                        MapNode(id="n8", label="8", detail="root"),
                        MapNode(id="n3", label="3", parent="n8"),
                        MapNode(id="n10", label="10", parent="n8"),
                        MapNode(id="n1", label="1", parent="n3"),
                        MapNode(id="n6", label="6", parent="n3"),
                        MapNode(id="n14", label="14", parent="n10"),
                    ],
                ),
            ),
        ),
    )


def _chemistry_table() -> VisualSpec:
    return VisualSpec(
        concept="Reaction quotient",
        deterministic=DeterministicVisual(
            title="Reaction Direction Rules",
            task=_task(
                "compare_variants",
                ["comparison"],
                ["Qc < Kc", "Qc > Kc"],
            ),
            scene=VisualScene(
                scene_kind="comparison",
                caption="Comparing Qc with Kc gives the direction of shift.",
                comparison=VisualComparison(
                    panels=[],
                    table=ComparisonTable(
                        rows=[
                            ComparisonRow(label="Condition"),
                            ComparisonRow(label="Direction"),
                        ],
                        columns=[
                            ComparisonColumn(
                                label="Qc < Kc",
                                values=["excess reactants", "forward"],
                            ),
                            ComparisonColumn(
                                label="Qc > Kc",
                                values=["excess products", "backward"],
                            ),
                        ],
                        verdict="The system shifts toward equilibrium when Qc != Kc.",
                    ),
                ),
            ),
        ),
    )


def _cycle_map() -> VisualSpec:
    return VisualSpec(
        concept="Cell cycle",
        deterministic=DeterministicVisual(
            title="The Cell Cycle",
            task=_task("cyclic_process", ["concept_map"], ["G1", "S", "G2", "M"]),
            scene=VisualScene(
                scene_kind="concept_map",
                caption="The cycle repeats through growth and division.",
                concept_map=VisualConceptMap(
                    layout="cycle",
                    nodes=[
                        MapNode(id="g1", label="G1", detail="growth"),
                        MapNode(id="s", label="S", detail="replication"),
                        MapNode(id="g2", label="G2", detail="preparation"),
                        MapNode(id="m", label="M", detail="mitosis"),
                    ],
                ),
            ),
        ),
    )


_BOX_RE = re.compile(
    r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" '
    r'height="([-\d.]+)" rx="12\.0"'
)


def _node_boxes(svg: str) -> Iterator[tuple[float, float, float, float]]:
    for match in _BOX_RE.finditer(svg):
        yield tuple(float(group) for group in match.groups())  # type: ignore[misc]


def test_shallow_taxonomy_is_horizontal() -> None:
    spec = _shallow_cell_tree()
    assert validate_lesson_spec(spec) == []
    assert select_family(spec) == "concept_map"
    assert concept_map_is_horizontal(spec.deterministic.scene.concept_map)  # type: ignore[union-attr]

    hero = render_hero_geometry(spec.deterministic)
    boxes = list(_node_boxes(hero))
    assert len(boxes) == 4
    root_x, root_y, root_w, _ = boxes[0]
    child_boxes = boxes[1:]
    assert all(x > root_x + root_w for x, _, _, _ in child_boxes)
    # Every detail is printed inside its node, so the composer must not repeat
    # the same text as callout cards.
    for detail in ("basic unit of life", "no true nucleus", "membrane-bound nucleus"):
        assert detail in hero
    content = extract_lesson_content(spec, hero_svg=hero)
    assert content.callouts == []


def test_deep_tree_is_vertical_and_collision_free() -> None:
    spec = _deep_bst_tree()
    assert validate_lesson_spec(spec) == []
    assert not concept_map_is_horizontal(spec.deterministic.scene.concept_map)  # type: ignore[union-attr]

    hero = render_hero_geometry(spec.deterministic)
    boxes = list(_node_boxes(hero))
    assert len(boxes) == 6
    for index, (ax, ay, aw, ah) in enumerate(boxes):
        for bx, by, bw, bh in boxes[index + 1:]:
            assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
    # Root above the second level; leaves at the bottom.
    assert boxes[0][1] < boxes[1][1] < boxes[3][1]


def test_vertical_tree_leaf_details_become_cards_but_headers_stay_in_hero() -> None:
    spec = _deep_bst_tree()
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    cmap.nodes.append(MapNode(id="note", label="Balanced", detail="O(log n) search", parent="n8"))
    hero = render_hero_geometry(spec.deterministic)
    assert "root" in hero  # non-leaf detail is drawn inside the node box
    content = extract_lesson_content(spec, hero_svg=hero)
    # Root detail is already in the hero; only the leaf detail becomes a card.
    assert [(c.heading, c.body) for c in content.callouts] == [
        ("Balanced", "O(log n) search")
    ]


def test_cycle_stage_details_are_printed_as_cards() -> None:
    spec = _cycle_map()
    content = extract_lesson_content(spec, hero_svg="<svg/>")
    assert [c.heading for c in content.callouts] == ["G1", "S", "G2", "M"]
    assert [c.body for c in content.callouts] == [
        "growth", "replication", "preparation", "mitosis"
    ]


def test_hero_rendering_is_deterministic() -> None:
    spec = _shallow_cell_tree()
    assert render_hero_geometry(spec.deterministic) == render_hero_geometry(
        spec.deterministic
    )


def test_tree_rejects_two_roots() -> None:
    spec = _shallow_cell_tree()
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    cmap.nodes[1].parent = ""
    errors = validate_lesson_spec(spec)
    assert any("exactly one root" in error for error in errors)


def test_tree_rejects_unknown_parent() -> None:
    spec = _shallow_cell_tree()
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    cmap.nodes[1].parent = "missing"
    errors = validate_lesson_spec(spec)
    assert any("unknown parent" in error for error in errors)


def test_tree_rejects_parent_cycle() -> None:
    spec = _shallow_cell_tree()
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    # Replace two children with a two-node cycle while keeping the root.
    cmap.nodes = [
        MapNode(id="root", label="Root"),
        MapNode(id="a", label="A", parent="b"),
        MapNode(id="b", label="B", parent="a"),
    ]
    errors = validate_lesson_spec(spec)
    assert any("cycle" in error for error in errors)


def test_tree_rejects_too_few_nodes_and_too_many_cycle_stages() -> None:
    spec = _shallow_cell_tree()
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    cmap.nodes = cmap.nodes[:2]
    assert any("at least 3" in error for error in validate_lesson_spec(spec))

    cycle = _cycle_map()
    cycle_map = cycle.deterministic.scene.concept_map  # type: ignore[union-attr]
    cycle_map.nodes = [
        MapNode(id=f"n{i}", label=f"Stage {i}") for i in range(7)
    ]
    assert any("at most 6" in error for error in validate_lesson_spec(cycle))


def test_feature_table_renders_every_cell_verbatim() -> None:
    spec = _chemistry_table()
    assert validate_lesson_spec(spec) == []
    # SVG escapes comparison operators; the rendered glyphs are the originals.
    hero = html.unescape(render_hero_geometry(spec.deterministic))
    for value in (
        "Qc < Kc",
        "Qc > Kc",
        "Condition",
        "Direction",
        "excess reactants",
        "excess products",
        "forward",
        "backward",
    ):
        assert value in hero
    assert "FEATURE" in hero


def test_table_rejects_ragged_or_empty_cells() -> None:
    spec = _chemistry_table()
    table = spec.deterministic.scene.comparison.table  # type: ignore[union-attr]
    table.columns[1].values = ["excess products"]
    errors = validate_lesson_spec(spec)
    assert any("values" in error for error in errors)

    spec = _chemistry_table()
    table = spec.deterministic.scene.comparison.table  # type: ignore[union-attr]
    table.columns[0].values[0] = ""
    errors = validate_lesson_spec(spec)
    assert any("empty" in error for error in errors)


def test_table_rejects_one_row_or_one_column() -> None:
    spec = _chemistry_table()
    table = spec.deterministic.scene.comparison.table  # type: ignore[union-attr]
    table.rows = table.rows[:1]
    table.columns[0].values = table.columns[0].values[:1]
    table.columns[1].values = table.columns[1].values[:1]
    errors = validate_lesson_spec(spec)
    assert any("at least 2 labeled rows" in error for error in errors)

    spec = _chemistry_table()
    table = spec.deterministic.scene.comparison.table  # type: ignore[union-attr]
    table.columns = table.columns[:1]
    errors = validate_lesson_spec(spec)
    assert any("at least 2 labeled columns" in error for error in errors)


def test_comparison_table_suppresses_duplicate_reasoning_and_prefers_verdict() -> None:
    spec = _chemistry_table()
    det = spec.deterministic
    det.composition = LessonComposition(
        title="Reaction Direction Rules",
        framing="Compare Qc with Kc.",
        callouts=[
            CompositionCallout(id="dup", label="Qc < Kc", value="forward"),
            CompositionCallout(
                id="promoted-claim-1",
                label="ΔG = −RT ln K",
                value="equilibrium constant relation",
            ),
        ],
        reasoning=[
            CompositionReasoningStep(
                id="r1", expression="Qc < Kc", explanation="forward"
            ),
            CompositionReasoningStep(
                id="r2", expression="Qc > Kc", explanation="backward"
            ),
        ],
        result=CompositionResult(expression="Qc = Kc"),
        takeaway="The system shifts toward equilibrium.",
    )
    content = extract_lesson_content(spec, hero_svg="<svg/>")
    assert content.layout == "compare"
    assert content.reasoning == ""
    assert content.reasoning_steps == []
    # A chip the table already prints is suppressed; a promoted equation that
    # exists nowhere in the hero must stay visible.
    assert [(c.heading, c.body) for c in content.callouts] == [
        ("ΔG = −RT ln K", "equilibrium constant relation")
    ]
    assert content.result_text == "The system shifts toward equilibrium when Qc != Kc."


def test_section_numbers_follow_reading_order() -> None:
    content = LessonContent(
        title="Reaction Direction Rules",
        hero_svg="<svg/>",
        hero_kind="comparison",
        layout="compare",
        reasoning="one row",
        result_text="Qc = Kc",
        takeaway="Compare the quotient with the equilibrium constant.",
    )
    src = build_lesson_typst(content)
    result_at = src.index('#sec("02", "THE RESULT")')
    reasoning_at = src.index('#sec("03", "HOW IT WORKS")')
    takeaway_at = src.index('#sec("04", "REMEMBER THIS")')
    assert result_at < reasoning_at < takeaway_at
    assert '#sec("03", "THE RESULT")' not in src


def test_label_colon_does_not_break_exact_claim_matching() -> None:
    spec = _deep_bst_tree()
    spec.deterministic.task = _task(
        "hierarchy_structure",
        ["concept_map"],
        ["Worst-case: O(n) for skewed tree"],
    )
    cmap = spec.deterministic.scene.concept_map  # type: ignore[union-attr]
    cmap.nodes.append(
        MapNode(
            id="perf",
            label="Worst-case",
            detail="O(n) for skewed tree",
            parent="n8",
        )
    )
    assert validate_lesson_spec(spec) == []


def test_layout_selection_is_content_structural() -> None:
    assert extract_lesson_content(_shallow_cell_tree(), hero_svg="<svg/>").layout == "map_tree"
    assert extract_lesson_content(_cycle_map(), hero_svg="<svg/>").layout == "map_cycle"
    assert extract_lesson_content(_chemistry_table(), hero_svg="<svg/>").layout == "compare"


def test_should_use_v3_routes_new_families(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "EXPLAIN_VISUALLY_V3", True)
    assert should_use_v3(_shallow_cell_tree()) is True
    assert should_use_v3(_chemistry_table()) is True


def test_taxonomy_heading_synonym_is_ok_when_claims_are_complete() -> None:
    spec = _shallow_cell_tree()
    spec.deterministic.task = VisualTask(
        task_type="hierarchy_structure",
        learning_outcome="",
        required_elements=["Prokaryotic", "Summation Notation"],
        required_relationships=[],
        exact_claims=["no true nucleus"],
        acceptable_families=["concept_map"],
        assumptions=[],
        missing=[],
    )
    # "no true nucleus" is printed in the Prokaryotic node; the synonym
    # heading is tolerated because the visible claim is the stronger contract.
    assert validate_lesson_spec(spec) == []


def test_taxonomy_heading_synonym_still_fails_when_a_claim_is_missing() -> None:
    spec = _shallow_cell_tree()
    spec.deterministic.task = VisualTask(
        task_type="hierarchy_structure",
        learning_outcome="",
        required_elements=["Prokaryotic", "Summation Notation"],
        required_relationships=[],
        exact_claims=["Σ i = n(n+1)/2"],
        acceptable_families=["concept_map"],
        assumptions=[],
        missing=[],
    )
    errors = validate_lesson_spec(spec)
    assert any("required element unsatisfied" in error for error in errors)
    assert any("exact claim missing" in error for error in errors)


def _typst_present() -> bool:
    return shutil.which("typst") is not None


@pytest.mark.skipif(not _typst_present(), reason="typst CLI not installed")
def test_composed_lessons_render_to_png() -> None:
    from app.utils.visual_lesson import render_visual_lesson

    for spec in (_shallow_cell_tree(), _chemistry_table(), _cycle_map()):
        lesson = render_visual_lesson(spec, out_format="png")
        assert lesson.data[:8] == b"\x89PNG\r\n\x1a\n"
        assert lesson.fallback_used is False
