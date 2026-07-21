from app.config import settings
from app.exceptions import ImageTooLargeError


def validate_image_size(image_bytes: bytes) -> None:
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise ImageTooLargeError(max_mb=settings.MAX_IMAGE_SIZE_MB)
