"""Fail-safe contracts for the OCR engine (P0: affordable text extraction).

Production runs on a 512 MB instance; the engine must initialize there, and
any recoverable init/inference failure must degrade to Gemini escalation via
the empty-result gate — never raise, never fabricate text.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from app.services import ocr_service


@pytest.fixture(autouse=True)
def reset_engine_state(monkeypatch):
    monkeypatch.setattr(ocr_service, "_engine", None)
    monkeypatch.setattr(ocr_service, "_engine_error", None)
    yield


def _box() -> list[list[float]]:
    return [[0.0, 0.0], [120.0, 0.0], [120.0, 30.0], [0.0, 30.0]]


def _tok(text: str, conf: float = 1.0) -> tuple:
    return (_box(), text, conf)


class _FakeOutput:
    def __init__(self, boxes, txts, scores):
        self.boxes = boxes
        self.txts = txts
        self.scores = scores


def test_to_ocr_result_converts_numpy_arrays():
    out = _FakeOutput(
        boxes=[np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)],
        txts=["Name"],
        scores=[0.99],
    )
    assert ocr_service._to_ocr_result(out) == [
        ([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]], "Name", 0.99)
    ]


def test_to_ocr_result_empty_output_is_empty_list():
    assert ocr_service._to_ocr_result(_FakeOutput(None, None, None)) == []
    assert ocr_service._to_ocr_result(_FakeOutput([], (), [])) == []


def test_inference_failure_degrades_to_empty_result(monkeypatch, caplog):
    class ExplodingEngine:
        def __call__(self, image):
            raise RuntimeError("onnx session exploded")

    monkeypatch.setattr(ocr_service, "_engine", ExplodingEngine())
    with caplog.at_level(logging.ERROR):
        result = ocr_service.read_raw(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result == []
    assert "OCR inference failed" in caplog.text


def test_init_failure_is_cached_and_logged(monkeypatch, caplog):
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise OSError("model file missing")

    monkeypatch.setattr(ocr_service, "_create_engine", boom)
    with caplog.at_level(logging.ERROR):
        assert ocr_service._get_engine() is None
        assert ocr_service._get_engine() is None
    assert calls["n"] == 1, "failed init must be cached, not retried per request"
    assert "OCR engine unavailable" in caplog.text


def test_read_raw_with_cached_init_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(ocr_service, "_engine_error", "OSError: nope")
    assert ocr_service.read_raw(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_engine_init_success_logs_observability(monkeypatch, caplog):
    class WorkingEngine:
        def __call__(self, image):
            return _FakeOutput([], (), [])

    monkeypatch.setattr(ocr_service, "_create_engine", WorkingEngine)
    with caplog.at_level(logging.INFO):
        assert ocr_service._get_engine() is not None
    assert "OCR engine initialized" in caplog.text
    assert "engine=rapidocr" in caplog.text


def test_ocr_importable_never_initializes_engine(monkeypatch):
    def must_not_run():
        raise AssertionError("ocr_importable must not initialize the engine")

    monkeypatch.setattr(ocr_service, "_create_engine", must_not_run)
    assert ocr_service.ocr_importable() is True  # rapidocr is a pinned dependency
    assert ocr_service.ocr_importable() is True


# ── Gate: mixed digit-letter heuristic operates on words, not whole lines ──


def test_gate_allows_numbered_lines_and_years_in_clean_notes():
    tokens = [
        _tok("1. All living organisms are composed of one or more cells"),
        _tok("Robert Hooke (1665) first observed cells in a cork slice"),
        _tok("The cell theory states that cells arise from pre-existing cells"),
        _tok("Microscopes revealed the fundamental structure of living organisms"),
        _tok("Cell biology studies the basic units of life in great detail"),
    ]
    assert ocr_service.low_quality_result(tokens) is False


def test_gate_flags_glued_digit_letter_words():
    tokens = [
        _tok("x2y l0g of the function graph appears below"),
        _tok("4ac solves the quadratic expression for roots"),
        _tok("algebraic manipulation shows the final result clearly"),
        _tok("substitute the known values into the equation carefully"),
        _tok("calculate the numeric answer for this problem statement"),
    ]
    assert ocr_service.low_quality_result(tokens) is True
