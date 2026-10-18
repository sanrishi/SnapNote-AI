"""Process-wide serialization of memory-heavy image windows.

Image decode/denoise/prep and OCR inference each allocate transient buffers
that scale with image pixels (measured: ~+55 MB at 2.2 MP rising to ~+125 MB
at 12 MP per call). On the 512 MB production instance N concurrent requests
multiply that transient by N and can OOM the container (observed: 502 +
container restart under 3 concurrent requests). Serializing these windows
keeps peak memory at steady state + ONE transient regardless of request
concurrency; on the 0.1-CPU instance this costs no real throughput because
the CPU is the bottleneck either way.

The semaphore alone was not enough: when concurrent requests were dispatched
to different default-pool threads, each thread kept its own allocator arena
high-water marks and freed transients were not reusable across threads (the
convoy still OOM-killed the container). ``run_in_image_window`` therefore
pins ALL heavy image work to ONE dedicated worker thread, and the Dockerfile
caps glibc arenas (``MALLOC_ARENA_MAX``).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")

_IMAGE_WINDOW = threading.Semaphore(1)
_HEAVY_WORKERS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-window")


@contextmanager
def image_window() -> Iterator[None]:
    _IMAGE_WINDOW.acquire()
    try:
        yield
    finally:
        _IMAGE_WINDOW.release()


async def run_in_image_window(fn: Callable[..., T], *args: Any) -> T:
    """Run a heavy image function on the single dedicated worker thread.

    One thread owns every large transient allocation, so allocator high-water
    marks cannot multiply across threads, and submissions queue in order.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_HEAVY_WORKERS, fn, *args)
