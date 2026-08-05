import io
import json
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw, ImageFont

try:
    _font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    _font = ImageFont.load_default()


def _draw_text(draw, xy, text, **kw):
    draw.text(xy, text, fill="black", font=_font, **kw)


@pytest.fixture
def sample_text_image():
    buf = io.BytesIO()
    img = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(img)
    _draw_text(draw, (30, 20), "Introduction to Statistics")
    _draw_text(draw, (30, 80), "Mean is the average of all values")
    _draw_text(draw, (30, 140), "Median is the middle value")
    _draw_text(draw, (30, 200), "Mode is the most frequent value")
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_table_image():
    buf = io.BytesIO()
    img = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img)
    headers = ["Name", "Age", "Score"]
    data = [["Alice", "20", "95"], ["Bob", "21", "87"], ["Charlie", "19", "91"]]
    y = 20
    for row in [headers] + data:
        x = 30
        for cell in row:
            _draw_text(draw, (x, y), cell)
            x += 180
        y += 60
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_tight_table_image():
    buf = io.BytesIO()
    img = Image.new("RGB", (800, 350), "white")
    draw = ImageDraw.Draw(img)
    rows_data = [
        ["x", "1", "2", "3", "4"],
        ["y", "5", "6", "7", "8"],
        ["z", "9", "10", "11", "12"],
        ["w", "13", "14", "15", "16"],
        ["v", "17", "18", "19", "20"],
    ]
    y = 10
    for row in rows_data:
        x = 20
        for cell in row:
            _draw_text(draw, (x, y), cell)
            x += 120
        y += 40
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_diagram_image():
    buf = io.BytesIO()
    img = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 200, 280, 380], outline="black", width=3)
    draw.ellipse([350, 150, 550, 350], outline="black", width=3)
    draw.line([80, 200, 350, 150], fill="black", width=3)
    _draw_text(draw, (100, 390), "Process A")
    _draw_text(draw, (400, 360), "Component B")
    _draw_text(draw, (20, 20), "System Architecture Diagram")
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def oversized_image():
    return b"\xff\xd8\xff\xe0" * (5 * 1024 * 1024)


@pytest.fixture
def invalid_image():
    return b"this is not an image file"


@pytest.fixture
def mock_vision(monkeypatch):
    mock_response = type("obj", (), {"text": "## Diagram Type: flowchart\n### Description\nA simple flow\n\n### Labels\n- Process A\n- Component B"})()
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)
    return mock_model.generate_content_async


@pytest.fixture
def mock_study_notes(monkeypatch):
    """Mock the structured JSON Gemini response used by the diagram tier."""
    payload = json.dumps({
        "topic": {"title": "System Architecture", "is_probable": False},
        "visible_content": {"headings": ["System Architecture Diagram"], "equations": [], "labels": ["Process A", "Component B"], "statements": []},
        "study_notes": ["Process A is a rectangle.", "Component B is an ellipse.", "Process A connects to Component B."],
        "simple_explanation": "This appears to represent a system with two main components linked together.",
        "formula_box": [],
        "diagram_interpretation": {"present": True, "visible_elements": ["Process A", "Component B", "connecting line"], "likely_interpretation": ["The line likely shows a data flow from Process A to Component B."]},
        "uncertainties": ["The exact meaning of the connecting line cannot be confirmed from this frame."],
        "key_takeaway": "A system with two linked components.",
    })
    mock_response = type("obj", (), {"text": payload})()
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("app.services.vision_service.model", mock_model)
    return mock_model.generate_content_async


@pytest.fixture
def mock_vision_failure(monkeypatch):
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=Exception("Gemini API timeout"))
    monkeypatch.setattr("app.services.vision_service.model", mock_model)
    return mock_model.generate_content_async
