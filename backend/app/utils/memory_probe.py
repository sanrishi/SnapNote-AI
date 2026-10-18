"""Linux container memory observability for /health (measurement only).

Reads cgroup v2 (memory.current / memory.peak / memory.max) with cgroup v1
fallbacks, plus the process's current RSS from /proc/self/status.

Why this exists: `ru_maxrss` (exposed as `rss_mb`) counts resident pages
including mmapped model/text files, which overstates pressure against a
cgroup limit. `memory.current` is the number the kernel actually accounts
when deciding whether to OOM-kill the container — the two must be
distinguishable when judging headroom on the 512 MB instance.

All reads are best-effort: any failure yields None so /health can never
break because of monitoring.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _read_bytes(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
        if raw == "max":
            return None
        return int(raw)
    except Exception:
        return None


def _to_mb(value: int | None) -> float | None:
    return None if value is None else round(value / 1048576, 1)


def current_rss_mb() -> float | None:
    """Current resident set size in MB (Linux only; None elsewhere)."""
    if sys.platform != "linux":
        return None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        return None
    return None


def cgroup_memory_mb() -> dict[str, float | None]:
    """cgroup memory usage/peak/limit in MB (Linux only; None when unreadable)."""
    result: dict[str, float | None] = {
        "current_mb": None,
        "peak_mb": None,
        "limit_mb": None,
    }
    if sys.platform != "linux":
        return result
    candidates = (
        (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory.peak"),
            Path("/sys/fs/cgroup/memory.max"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
    )
    for current, peak, limit in candidates:
        value = _read_bytes(current)
        if value is None:
            continue
        result["current_mb"] = _to_mb(value)
        result["peak_mb"] = _to_mb(_read_bytes(peak))
        result["limit_mb"] = _to_mb(_read_bytes(limit))
        break
    return result
