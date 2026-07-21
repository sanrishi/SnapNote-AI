import logging

logger = logging.getLogger(__name__)


class SnapNoteError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthError(SnapNoteError):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class CreditLimitError(SnapNoteError):
    def __init__(self):
        super().__init__("Credit limit exceeded", status_code=429)


class ImageTooLargeError(SnapNoteError):
    def __init__(self, max_mb: int):
        super().__init__(f"Image too large. Max {max_mb}MB allowed.", status_code=413)


class InvalidInputError(SnapNoteError):
    def __init__(self, message: str = "Invalid input"):
        super().__init__(message, status_code=422)


class UpstreamError(SnapNoteError):
    def __init__(self, service: str, detail: str = ""):
        msg = f"{service} failed: {detail}" if detail else f"{service} failed"
        super().__init__(msg, status_code=502)
