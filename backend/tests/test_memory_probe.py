"""memory_probe: best-effort contract — never raises, stable shape."""
from app.utils.memory_probe import cgroup_memory_mb, current_rss_mb


def test_cgroup_memory_shape():
    mem = cgroup_memory_mb()
    assert set(mem) == {"current_mb", "peak_mb", "limit_mb"}
    for value in mem.values():
        assert value is None or (isinstance(value, float) and value > 0)


def test_current_rss_shape():
    value = current_rss_mb()
    assert value is None or (isinstance(value, float) and value > 0)
