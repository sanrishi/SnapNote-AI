"""Deterministic quality gates for comparison matrices.

Two failure modes motivated this module:

1. **Redundant dimensions.** A live chemistry planner emitted a ``Condition``
   column whose every cell repeated that row's label (``Qc < Kc``,
   ``Qc > Kc``). The matrix was grounded but spent half its width saying
   nothing. A redundant axis is removed when the table still has at least two
   real comparison columns; otherwise the table is left for the validator to
   refuse (never silently rendered with a wasted dimension).

2. **Unverified synthesis.** A verdict is the table's teaching conclusion, so
   it must be traceable to the data, not a plausible-sounding sentence. The
   planner supplies ``verdict_basis`` cell references; this module verifies
   those cells exist and span more than one fact, rejects invented numbers and
   single-cell restatements, and DROPS a synthesis it cannot trace (the matrix
   itself still renders). A verdict is never repaired by inventing evidence.

Pure functions only: no I/O, no model calls, no rounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.schemas import (
    ComparisonColumn,
    ComparisonRow,
    ComparisonTable,
    ComparisonVerdictBasis,
    VisualScene,
    VisualSpec,
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_CLAUSE_SPLIT_RE = re.compile(r"[;\n]|\.\s+|,\s+(?=[A-Z])|(?:\s+(?:and|while|whereas|but)\s+)", re.IGNORECASE)
_FEATURE_CORNERS = {
    "feature",
    "features",
    "dimension",
    "dimensions",
    "metric",
    "metrics",
    "property",
    "properties",
}

# Row/column labels that name a shared feature rather than an alternative.
# Used only to catch a planner's mislabelled ``corner`` (e.g. corner="STATE"
# over rows Condition/Direction); it never overrides explicit cell evidence.
_FEATURE_LABEL_WORDS = {
    "condition",
    "direction",
    "mean",
    "variance",
    "mgf",
    "moment generating function",
    "structure",
    "example",
    "definition",
    "function",
    "role",
    "location",
    "shape",
    "size",
    "charge",
    "mass",
    "speed",
    "time complexity",
    "space complexity",
    "stability",
    "advantage",
    "advantages",
    "disadvantage",
    "disadvantages",
    "pros",
    "cons",
    "formula",
    "equation",
    "units",
    "notes",
    "description",
    "mechanism",
    "rate",
    "substrate",
    "product",
    "catalyst",
    "kinetics",
    "stereochemistry",
    "bond angle",
}


def _norm(text: str) -> str:
    """Comparison key for labels/cells (presentation-only differences)."""
    return re.sub(r"[\s×·*:]+", "", (text or "").casefold())


def _flat(text: str) -> str:
    """Case/whitespace-insensitive text for clause evidence checks."""
    return " ".join((text or "").casefold().split())


def _clauses(text: str) -> list[str]:
    return [clause.strip() for clause in _CLAUSE_SPLIT_RE.split(text or "") if clause.strip()]


def _value_present(value: str, text: str) -> bool:
    """True when a table value is actually stated inside ``text``.

    Exact word-sequence matching: a one-character value (``l``, ``0``) needs
    token boundaries so it cannot match inside an unrelated word, while
    formulas (``np(1-p)``) match as substrings. Pure punctuation ("-") is not
    evidence of anything.
    """
    needle = _flat(value)
    haystack = _flat(text)
    if not needle or not any(ch.isalnum() for ch in needle):
        return False
    if len(needle) == 1:
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return needle in haystack


def _clauses_with(label: str, value: str, verdict: str) -> int:
    """Count verdict clauses that state ``label`` and ``value`` together."""
    needle = _flat(label)
    if not needle:
        return 0
    return sum(
        1
        for clause in _clauses(verdict)
        if needle in _flat(clause) and _value_present(value, clause)
    )


def _is_feature_label(label: str) -> bool:
    """True when a label names a shared feature, not a compared alternative.

    Parenthetical units (``Mean (μ)``) are separators, and multi-word feature
    names (``moment generating function``) match as a phrase. The heuristic is
    deliberately asymmetric: it only breaks ties on a clear imbalance (see
    ``_infer_alternatives_are_columns``), so a subject term like "product"
    cannot flip an otherwise explicitly oriented table on its own.
    """
    flat = " " + re.sub(r"[^a-z0-9]+", " ", _flat(label)).strip() + " "
    if not flat.strip():
        return False
    return any(f" {word} " in flat for word in _FEATURE_LABEL_WORDS)


def _infer_alternatives_are_columns(table: ComparisonTable) -> bool | None:
    """Infer the comparison axis from labels; None when the labels are neutral.

    A table is usually oriented so the alternatives are the axis with proper
    names (``Qc < Kc``, ``Binomial``) and the shared features are the axis with
    generic nouns (``Condition``, ``Mean``). This catches a planner corner that
    names the wrong axis, without trusting subject vocabulary. A margin of two
    labels keeps one coincidental feature word (an alternative called
    "Product", "Mean" or "Function") from overriding the planner's corner.
    """
    rows, columns = _labels(table)
    row_hits = sum(1 for row in rows if _is_feature_label(row.label))
    column_hits = sum(1 for column in columns if _is_feature_label(column.label))
    if row_hits >= column_hits + 2:
        return True
    if column_hits >= row_hits + 2:
        return False
    return None


def _alternatives_are_columns(table: ComparisonTable) -> bool:
    """Whether the columns are the compared alternatives (vs. shared features).

    An explicit feature corner (``FEATURE``, ``METRIC``, …) is authoritative;
    label inference only repairs a missing or nonsensical corner.
    """
    if _norm(table.corner) in _FEATURE_CORNERS:
        return True
    inferred = _infer_alternatives_are_columns(table)
    if inferred is not None:
        return inferred
    return False


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def _labels(table: ComparisonTable) -> tuple[list[ComparisonRow], list[ComparisonColumn]]:
    rows = [r for r in table.rows if (r.label or "").strip()]
    columns = [c for c in table.columns if (c.label or "").strip()]
    return rows, columns


def _table_text(table: ComparisonTable) -> str:
    rows, columns = _labels(table)
    bits = [table.corner or "", *(r.label for r in rows), *(c.label for c in columns)]
    if all(len(c.values) == len(rows) for c in columns):
        for column in columns:
            bits.extend(column.values)
    return " ".join(bits)


def verdict_issue(table: ComparisonTable) -> str | None:
    """Why the verdict is not evidence-traceable, or None when it is.

    Verified properties (all deterministic):
    - every number in the synthesis already appears in the matrix;
    - the synthesis is not a verbatim restatement of one cell/label;
    - explicit evidence references resolve to real, non-empty cells;
    - every cited cell's outcome is stated in the SAME clause as that
      alternative (not merely somewhere in the sentence);
    - cited evidence spans at least two alternatives along the table's real
      comparison axis (columns for a FEATURE corner, rows otherwise);
    - without references, the synthesis itself states outcomes for at least
      two alternatives, derived from the cells.
    """
    verdict = " ".join((table.verdict or "").split())
    if not verdict:
        return None
    rows, columns = _labels(table)
    if len(rows) < 2 or len(columns) < 2:
        return "table has no comparison axis for a verdict"
    if len({len(c.values) for c in columns}) > 1:
        return "table columns have inconsistent row counts"
    if any(len(c.values) != len(rows) for c in columns):
        return "table columns do not match the row count"

    invented = sorted(_numbers(verdict) - _numbers(_table_text(table)))
    if invented:
        return f"verdict introduces value(s) absent from the table: {', '.join(invented)}"

    normalized_verdict = _norm(verdict)
    restated = {
        _norm(bit)
        for bit in [
            *(r.label for r in rows),
            *(c.label for c in columns),
            *(value for c in columns for value in c.values),
        ]
    }
    if normalized_verdict in restated:
        return "verdict merely restates a single cell or label"

    columns_are_alternatives = _alternatives_are_columns(table)
    basis = [b for b in table.verdict_basis if (b.column or "").strip() or (b.row or "").strip()]
    if basis:
        column_index = {_norm(c.label): i for i, c in enumerate(columns)}
        row_index = {_norm(r.label): i for i, r in enumerate(rows)}
        cited: list[tuple[int, int, str]] = []
        for evidence in basis:
            ci = column_index.get(_norm(evidence.column))
            ri = row_index.get(_norm(evidence.row))
            if ci is None or ri is None:
                return (
                    "verdict evidence references an unknown cell: "
                    f"column={evidence.column!r} row={evidence.row!r}"
                )
            value = (columns[ci].values[ri] or "").strip()
            if not value:
                return (
                    "verdict evidence references an empty cell: "
                    f"column={columns[ci].label!r} row={rows[ri].label!r}"
                )
            cited.append((ci, ri, value))
        if len({(ci, ri) for ci, ri, _ in cited}) < 2:
            return "verdict evidence must cite at least two distinct cells"
        if columns_are_alternatives:
            if len({ci for ci, _, _ in cited}) < 2:
                return "verdict evidence must span at least two alternatives (columns)"
        elif len({ri for _, ri, _ in cited}) < 2:
            return "verdict evidence must span at least two alternatives (rows)"
        for ci, ri, value in cited:
            column_label = columns[ci].label
            row_label = rows[ri].label
            if (
                _clauses_with(column_label, value, verdict)
                + _clauses_with(row_label, value, verdict)
                == 0
            ):
                return (
                    "verdict does not state the cited outcome "
                    f"{value!r} for {column_label!r} / {row_label!r} in the same clause"
                )
        return None

    # No explicit evidence: derive the trace from the verdict itself. Each
    # matched alternative must be named in a clause that also states one of
    # its actual cell values; a sentence that merely mentions two labels is
    # not a synthesis.
    if columns_are_alternatives:
        matched = [
            column.label
            for column in columns
            if any(_clauses_with(column.label, column.values[ri], verdict) for ri in range(len(rows)))
        ]
        axis = "columns"
    else:
        matched = [
            row.label
            for ri, row in enumerate(rows)
            if any(_clauses_with(row.label, column.values[ri], verdict) for column in columns)
        ]
        axis = "rows"
    if len(matched) < 2:
        return (
            "verdict does not state outcomes for two alternatives along the "
            f"{axis} axis"
        )
    return None


def _duplicate_column_indices(columns: list[ComparisonColumn]) -> set[int]:
    """Indices of columns that repeat an earlier column exactly."""
    drop: set[int] = set()
    for i, column in enumerate(columns):
        for j in range(i):
            if j in drop:
                continue
            other = columns[j]
            if _norm(column.label) != _norm(other.label):
                continue
            if len(column.values) != len(other.values):
                continue
            if all(_norm(a) == _norm(b) for a, b in zip(column.values, other.values)):
                drop.add(i)
                break
    return drop


def redundant_column_indices(table: ComparisonTable) -> set[int]:
    """Columns whose every cell merely repeats that row's label.

    Such a column adds no information: the row label is already printed in
    the label column. It is safe to remove only while at least two real
    columns remain, which :func:`normalize_comparison_table` enforces.
    """
    rows, columns = _labels(table)
    redundant: set[int] = set()
    for index, column in enumerate(columns):
        values = list(column.values)
        if len(values) != len(rows) or not rows:
            continue
        if all(
            (value or "").strip()
            and _norm(value) == _norm(row.label)
            for value, row in zip(values, rows)
        ):
            redundant.add(index)
    return redundant


@dataclass
class TableQualityReport:
    """What the deterministic gate changed, for logs and tests."""

    dropped_columns: list[str] = field(default_factory=list)
    dropped_verdict: str = ""
    verdict_reason: str = ""
    corrected_corner: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.dropped_columns or self.dropped_verdict or self.corrected_corner)


def normalize_comparison_table(table: ComparisonTable) -> tuple[ComparisonTable, TableQualityReport]:
    """Return a quality-gated copy of a comparison table (pure).

    Redundant/duplicate columns are removed only when at least two real
    columns survive; otherwise the table is returned unchanged so the
    validator refuses it and the bounded repair can propose a genuinely
    two-dimensional matrix. A verdict that cannot be traced is cleared.
    """
    report = TableQualityReport()
    working = table.model_copy(deep=True)
    rows, columns = _labels(working)
    if len(rows) >= 2 and len(columns) >= 2:
        drop = redundant_column_indices(working) | _duplicate_column_indices(columns)
        if drop and len(columns) - len(drop) >= 2:
            for index in sorted(drop):
                report.dropped_columns.append(columns[index].label)
            kept = [column for index, column in enumerate(columns) if index not in drop]
            working.columns = kept
            report.notes.append(
                "removed redundant comparison column(s): "
                + ", ".join(report.dropped_columns)
            )
        rows, columns = _labels(working)
        # A corner like "STATE" over rows Condition/Direction is a mislabel:
        # the rows are shared features. Correcting it is presentation-only and
        # happens only when label evidence is clear (two-label margin), never
        # from subject vocabulary alone.
        if (
            len(rows) >= 2
            and len(columns) >= 2
            and all((r.label or "").strip() for r in rows)
            and all((c.label or "").strip() for c in columns)
            and _infer_alternatives_are_columns(working) is True
            and _norm(working.corner) not in _FEATURE_CORNERS
        ):
            report.corrected_corner = working.corner
            working.corner = "FEATURE"
            report.notes.append(
                f"corrected comparison corner {report.corrected_corner!r} -> 'FEATURE'"
            )

    issue = verdict_issue(working)
    if issue is not None:
        report.dropped_verdict = " ".join((working.verdict or "").split())
        report.verdict_reason = issue
        working.verdict = ""
        working.verdict_basis = []
    return working, report


def normalize_visual_spec(spec: VisualSpec) -> tuple[VisualSpec, TableQualityReport]:
    """Apply the comparison gate to a whole spec, returning a safe copy."""
    report = TableQualityReport()
    scene: VisualScene | None = spec.deterministic.scene
    if (
        scene is None
        or scene.scene_kind != "comparison"
        or scene.comparison is None
        or scene.comparison.table is None
    ):
        return spec, report
    normalized, report = normalize_comparison_table(scene.comparison.table)
    if not report.changed:
        return spec, report
    copy = spec.model_copy(deep=True)
    assert copy.deterministic.scene is not None
    assert copy.deterministic.scene.comparison is not None
    copy.deterministic.scene.comparison.table = normalized
    return copy, report


def table_quality_errors(table: ComparisonTable | None) -> list[str]:
    """Validation errors for dimensions/verdicts that escaped normalization."""
    errors: list[str] = []
    if table is None:
        return errors
    rows, columns = _labels(table)
    if len(rows) >= 2 and len(columns) >= 2:
        redundant = redundant_column_indices(table)
        if redundant:
            names = ", ".join(repr(columns[i].label) for i in sorted(redundant))
            errors.append(
                f"comparison column(s) {names} repeat the row labels; remove the "
                "redundant dimension or add a real compared feature"
            )
        duplicates = _duplicate_column_indices(columns)
        if duplicates:
            names = ", ".join(repr(columns[i].label) for i in sorted(duplicates))
            errors.append(f"comparison column(s) {names} duplicate another column")
    issue = verdict_issue(table)
    if issue is not None:
        errors.append(f"comparison verdict is not evidence-traceable: {issue}")
    return errors
