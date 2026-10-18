"""Process-wide serialization of memory-heavy image windows.

Image decode/denoise/prep and OCR inference each allocate transient buffers
that scale with image pixels (measured: ~+55 MB at 2.2 MP rising to ~+125 MB
at 12 MP per call). On the 512 MB production instance N concurrent requests
multiply that transient by N and can OOM the container (observed: 502 +
container restart under 3 concurrent requests). Serializing these windows
keeps peak memory at steady state + ONE transient regardless of request
concurrency; on the 0.1-CPU instance this costs no real throughput because
the CPU is the bottleneck either way.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_IMAGE_WINDOW = threading.Semaphore(1)


@contextmanager
def image_window() -> Iterator[None]:
    _IMAGE_WINDOW.acquire()
    try:
        yield
    finally:
        _IMAGE_WINDOW.release()
