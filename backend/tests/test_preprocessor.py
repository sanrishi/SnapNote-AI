import io
import threading
import time

import pytest
from PIL import Image

from app.config import settings
from app.services.preprocessor import enhance_for_vision, preprocess
from app.utils.image_window import image_window


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(w: int, h: int, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── OCR working-resolution cap (bounds per-request memory transients) ──


def test_preprocess_caps_over_limit_long_edge():
    out = preprocess(_png(3000, 1500))
    assert out.shape == (1024, 2048)  # h, w — capped exactly at the limit


def test_preprocess_leaves_under_limit_untouched():
    out = preprocess(_png(900, 550))
    assert out.shape == (550, 900)
    out = preprocess(_png(1600, 1362))  # fixture-scale slide
    assert out.shape == (1362, 1600)


def test_preprocess_still_rejects_invalid_bytes():
    with pytest.raises(ValueError):
        preprocess(b"not an image")


# ── enhance_for_vision: bounded decode + unchanged output contract ──


def test_enhance_for_vision_large_jpeg_caps_and_keeps_aspect():
    out = enhance_for_vision(_jpeg(4000, 3000))
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"
    assert img.size == (settings.MAX_VISION_LONG_EDGE, 960)


def test_enhance_for_vision_small_png_not_upscaled():
    out = enhance_for_vision(_png(900, 550))
    img = Image.open(io.BytesIO(out))
    assert img.size == (900, 550)


# ── image_window: heavy image windows never overlap across threads ──


def test_image_window_serializes_threads():
    order: list[str] = []
    first_holds = threading.Event()
    release_first = threading.Event()

    def holder() -> None:
        with image_window():
            order.append("a-in")
            first_holds.set()
            release_first.wait(timeout=10)
            order.append("a-out")

    def waiter() -> None:
        first_holds.wait(timeout=10)
        with image_window():
            order.append("b-in")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    time.sleep(0.3)
    assert order == ["a-in"]  # second window must block until the first releases
    release_first.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert order == ["a-in", "a-out", "b-in"]
