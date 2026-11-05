"""Exact verification for worked mathematics inside the Hinglish explanation.

The question-first explanation is written by Gemini, but an exact interval
inequality answer must never be trusted because the prose sounds confident.
This module re-solves the machine-readable statement with the existing
``math_verify`` solver and compares the claimed solution set.

The contract is deliberately narrow:

* no ``problem_expr`` -> ``not_applicable`` (chemistry, biology, physics,
  computer science, and any maths question that is not a single interval
  inequality);
* ``problem_expr`` that is not an inequality -> metadata is discarded as
  ``not_applicable`` (the model must not downgrade unrelated subjects);
* supported and equal -> ``verified``;
* supported and different / unparseable -> ``mismatch`` (the caller runs one
  bounded repair, then withholds the worked steps and shows only the verified
  result);
* supported-looking but refused by the solver -> ``unsupported`` (never
  labelled verified).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.models.schemas import ExplanationLanguage, StudyExplanation
from app.utils.math_intervals import format_interval_set, parse_interval_set
from app.utils.math_verify import SolveResult, solve_problem

ExplanationCheckStatus = Literal[
    "not_applicable", "verified", "mismatch", "unsupported"
]

_RELATION_RE = re.compile(r"<=|>=|<|>|\u2264|\u2265")
_LOG_BASE_WORDS_RE = re.compile(
    r"\blog\s*base\s*(\([^()]*\)|[^\s(]+)\s+of\b", re.IGNORECASE
)


def _normalize_problem_expr(problem: str) -> str:
    """Accept the natural "log base (x+3) of (...)" phrasing as a safety net.

    The prompt asks for ``log_{x+3}(...)`` directly. This normalization only
    rewrites the base marker; the solver still re-solves and verifies the
    result exactly, so a mis-normalized input cannot create a false pass.
    """

    def replace(match: re.Match[str]) -> str:
        base = match.group(1).strip()
        if base.startswith("(") and base.endswith(")"):
            base = base[1:-1].strip()
        return f"log_{{{base}}}"

    return _LOG_BASE_WORDS_RE.sub(replace, problem)


def _solve_claim(problem: str) -> SolveResult:
    return solve_problem(_normalize_problem_expr(problem))


@dataclass(frozen=True)
class ExplanationCheck:
    """Outcome of re-solving one explanation's interval claim."""

    status: ExplanationCheckStatus
    expected: str = ""
    reason: str = ""


def check_explanation_math(explanation: StudyExplanation | None) -> ExplanationCheck:
    """Re-solve the explanation's ``problem_expr`` and compare its solution set."""
    if explanation is None:
        return ExplanationCheck("not_applicable")
    problem = explanation.problem_expr.strip()
    if not problem:
        return ExplanationCheck("not_applicable")
    # ``math_verify`` is an exact solver for one real variable x. A relation
    # in other symbols (Q < Kc, v < c, n log n < ...) is not an interval
    # problem and must not be reported as an unverified maths failure.
    if _RELATION_RE.search(problem) is None or re.search(r"\bx\b", problem) is None:
        return ExplanationCheck("not_applicable")

    result = _solve_claim(problem)
    if not result.supported or result.solution is None:
        return ExplanationCheck(
            "unsupported",
            reason=result.reason or "the exact solver could not verify this problem",
        )

    expected = format_interval_set(result.solution)
    claimed = explanation.solution_set.strip()
    if not claimed:
        return ExplanationCheck(
            "mismatch",
            expected=expected,
            reason="the explanation did not state a final solution set",
        )
    parsed = parse_interval_set(claimed)
    if isinstance(parsed, str):
        return ExplanationCheck(
            "mismatch",
            expected=expected,
            reason=f"the claimed solution set is not exact interval notation: {parsed}",
        )
    if not parsed.equals(result.solution):
        return ExplanationCheck(
            "mismatch",
            expected=expected,
            reason=f"the claimed solution set {claimed!r} does not match {expected!r}",
        )
    return ExplanationCheck("verified", expected=expected)


def apply_explanation_check(
    explanation: StudyExplanation, check: ExplanationCheck
) -> None:
    """Record a non-mismatch verification outcome on the explanation.

    Mismatches are handled by the caller (bounded repair), never silently
    accepted. ``verified`` and ``verification_note`` are server-owned fields:
    a model can never make an unverified claim look verified.
    """
    explanation.verified = False
    explanation.verification_note = ""
    if check.status == "verified":
        explanation.verified = True
        explanation.solution_set = check.expected
        return
    if check.status == "unsupported":
        explanation.confidence = "review"
        explanation.verification_note = (
            "SnapNote ka exact solver is solution ko independently verify nahi "
            "kar paaya — textbook se check karo."
            if explanation.language == ExplanationLanguage.HINGLISH
            else "SnapNote's exact solver could not independently verify this "
            "solution — check it against your textbook before relying on it."
        )


def withhold_unverified_solution(
    explanation: StudyExplanation, check: ExplanationCheck
) -> None:
    """Drop unverified reasoning, keeping only the exact solver's result.

    Used after a failed bounded repair. The student never sees a wrong worked
    answer presented as the professor's solution: the verified final set is
    shown, the unverifiable steps are removed, and the block is marked review.
    """
    explanation.steps = []
    explanation.recap = []
    explanation.confidence = "review"
    explanation.verified = bool(check.expected)
    explanation.verification_note = (
        "Worked steps exactly verify nahi ho paaye, isliye sirf "
        "solver-verified final result dikhaya gaya hai."
        if explanation.language == ExplanationLanguage.HINGLISH
        else "The worked steps could not be verified exactly, so only the "
        "solver-verified final result is shown."
    )
    if check.expected:
        explanation.solution_set = check.expected
        explanation.exam_answer = f"Verified solution: {check.expected}"
    else:
        explanation.solution_set = ""
        explanation.exam_answer = ""
