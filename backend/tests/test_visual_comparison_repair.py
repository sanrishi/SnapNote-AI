"""Focused repair for an empty comparison payload.

Re-emitting the complete nested VisualSpec to fill one missing table proved
unreliable on dense boards: the model returned the same null payload twice.
The focused repair asks for the payload only, then merges it. These tests keep
that contract honest: a successful table is merged and validated, a garbage
response returns None (so the route refuses instead of drawing nothing), and
the empty-payload reason is routed to the focused path.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.models.schemas import VisualSpec
from app.services import vision_service
from app.utils.visual_lesson import validate_lesson_spec
from test_visual_grammars import _chemistry_table


def _empty_comparison_spec() -> VisualSpec:
    spec = _chemistry_table()
    assert spec.deterministic.scene is not None
    spec.deterministic.scene.comparison = None
    return spec


def test_empty_comparison_reason_is_detected() -> None:
    assert vision_service._needs_comparison_payload_repair(
        "comparison has no comparison payload"
    )
    assert vision_service._needs_comparison_payload_repair(
        "comparison needs at least 2 labeled panels or a feature table"
    )
    assert not vision_service._needs_comparison_payload_repair(
        "exact claim missing from scene: 'x = 1'"
    )


def test_focused_repair_merges_table_and_passes_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "table": {
            "rows": [{"label": "Qc < Kc"}, {"label": "Qc > Kc"}],
            "columns": [
                {"label": "Direction", "values": ["forward", "backward"]},
                {"label": "Excess", "values": ["reactants", "products"]},
            ],
            "corner": "CONDITION",
            "verdict": "The system shifts to restore equilibrium.",
        }
    }
    monkeypatch.setattr(
        vision_service,
        "_call_gemini",
        AsyncMock(return_value=json.dumps(payload)),
    )
    spec = _empty_comparison_spec()

    repaired = asyncio.run(
        vision_service._repair_comparison_payload(spec, b"image", None)
    )

    assert repaired is not None
    table = repaired.deterministic.scene.comparison.table  # type: ignore[union-attr]
    assert table is not None
    assert table.corner == "CONDITION"
    assert [c.label for c in table.columns] == ["Direction", "Excess"]
    assert validate_lesson_spec(repaired) == []


def test_focused_repair_returns_none_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vision_service,
        "_call_gemini",
        AsyncMock(return_value=json.dumps({"table": {"rows": [], "columns": []}})),
    )

    repaired = asyncio.run(
        vision_service._repair_comparison_payload(_empty_comparison_spec(), b"image", None)
    )

    assert repaired is None


def test_thin_repair_routes_empty_comparison_to_focused_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _empty_comparison_spec()
    focused = AsyncMock(return_value=spec)
    monkeypatch.setattr(vision_service, "_repair_comparison_payload", focused)

    out = asyncio.run(
        vision_service.repair_thin_spec(
            spec, b"image", None, "comparison has no comparison payload"
        )
    )

    assert out is spec
    focused.assert_awaited_once()
