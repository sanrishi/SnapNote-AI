import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.config import settings

logger = logging.getLogger(__name__)


def load_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes to OpenCV BGR array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = load_image_bytes(image_bytes)
    if img is None:
        raise ValueError("Invalid image data")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def enhance_for_vision(image_bytes: bytes) -> bytes:
    """Enhance + downscale image for Vision LLM.

    Downscales the long edge to ``MAX_VISION_LONG_EDGE`` and outputs JPEG so
    phone screenshots don't balloon before hitting Gemini. Logs the size
    reduction so the improvement can be verified.
    """
    orig_size = len(image_bytes)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    long_edge = max(orig_w, orig_h)
    if long_edge > settings.MAX_VISION_LONG_EDGE:
        scale = settings.MAX_VISION_LONG_EDGE / long_edge
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img = img.filter(ImageFilter.SHARPEN)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=settings.VISION_JPEG_QUALITY, optimize=True)
    out = buf.getvalue()

    logger.info(
        "enhance_for_vision: %d KB %dx%d -> %d KB %dx%d (JPEG q%d)",
        orig_size // 1024, orig_w, orig_h,
        len(out) // 1024, img.size[0], img.size[1],
        settings.VISION_JPEG_QUALITY,
    )
    return out
