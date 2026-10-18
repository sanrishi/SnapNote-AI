import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.config import settings
from app.utils.image_window import image_window

logger = logging.getLogger(__name__)


def load_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes to OpenCV BGR array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def preprocess(image_bytes: bytes) -> np.ndarray:
    with image_window():
        img = load_image_bytes(image_bytes)
        if img is None:
            raise ValueError("Invalid image data")

        # Cap working resolution before the heavy ops: grayscale/denoise
        # transients scale with pixels (measured ~+125 MB at 12 MP), so an
        # uncapped 10 MB phone screenshot can spike memory on its own.
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > settings.OCR_MAX_LONG_EDGE:
            scale = settings.OCR_MAX_LONG_EDGE / long_edge
            img = cv2.resize(
                img,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

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
    with image_window():
        orig_size = len(image_bytes)
        img = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = img.size
        # JPEG DCT-scaled decode (no-op for PNG/WebP): bounds the source
        # decode buffer of large camera JPEGs before the resize below.
        img.draft("RGB", (settings.MAX_VISION_LONG_EDGE, settings.MAX_VISION_LONG_EDGE))
        img = img.convert("RGB")

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
