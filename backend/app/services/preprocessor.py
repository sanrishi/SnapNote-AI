import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


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
    """Enhance image for Vision LLM (less aggressive, keep natural look)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.filter(ImageFilter.SHARPEN)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
