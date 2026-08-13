import logging
import os
import sqlite3
import threading
import time

from app.config import settings

logger = logging.getLogger(__name__)

_db_path = settings.CREDITS_DB_PATH
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(_db_path))
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute(
            "CREATE TABLE IF NOT EXISTS device_credits ("
            "  device_id TEXT PRIMARY KEY,"
            "  credits_remaining INTEGER NOT NULL DEFAULT 50,"
            "  credits_used INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        _local.conn.execute(
            "CREATE TABLE IF NOT EXISTS visual_explanations ("
            "  diagram_id TEXT PRIMARY KEY,"
            "  device_id TEXT NOT NULL,"
            "  study_notes_json TEXT NOT NULL DEFAULT '',"
            "  visual_url TEXT NOT NULL DEFAULT '',"
            "  generated_at INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        try:
            _local.conn.execute(
                "ALTER TABLE visual_explanations ADD COLUMN study_notes_json TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        _local.conn.commit()
    return _local.conn


def init_device(device_id: str) -> tuple[int, int]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT credits_remaining, credits_used FROM device_credits WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    if row is not None:
        return row["credits_remaining"], row["credits_used"]
    conn.execute(
        "INSERT OR IGNORE INTO device_credits (device_id, credits_remaining, credits_used) VALUES (?, ?, 0)",
        (device_id, settings.FREE_CREDITS_MONTHLY),
    )
    conn.commit()
    return settings.FREE_CREDITS_MONTHLY, 0


def get_credits(device_id: str) -> tuple[int, int]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT credits_remaining, credits_used FROM device_credits WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    if row is None:
        return init_device(device_id)
    return row["credits_remaining"], row["credits_used"]


def add_credits(device_id: str, amount: int) -> int:
    conn = _get_conn()
    init_device(device_id)
    conn.execute(
        "UPDATE device_credits SET credits_remaining = credits_remaining + ? WHERE device_id = ?",
        (amount, device_id),
    )
    conn.commit()
    remaining, _ = get_credits(device_id)
    logger.info("Added %d credits to %s (now %d)", amount, device_id[:8], remaining)
    return remaining


def use_credits(device_id: str, amount: int) -> int:
    conn = _get_conn()
    remaining, used = get_credits(device_id)
    if remaining < amount:
        return remaining
    conn.execute(
        "UPDATE device_credits SET credits_remaining = credits_remaining - ?, credits_used = credits_used + ? WHERE device_id = ?",
        (amount, amount, device_id),
    )
    conn.commit()
    new_remaining, _ = get_credits(device_id)
    logger.info("Used %d credits for %s (now %d)", amount, device_id[:8], new_remaining)
    return new_remaining


def record_diagram_grant(device_id: str, diagram_id: str, study_notes_json: str) -> None:
    """Record a freshly completed 5-credit diagram result.

    This is the ONLY way a device earns an Explain Visually entitlement. It is
    called after a successful 5-credit diagram extraction, never at request
    start. One generation per purchased result: once visual_url is set it stays
    immutable and the image model is never called again for that diagram_id.
    The study notes are stored server-side so the visual route can reuse them
    without trusting client input.
    """
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO visual_explanations (diagram_id, device_id, study_notes_json, generated_at) VALUES (?, ?, ?, ?)",
        (diagram_id, device_id, study_notes_json, int(time.time())),
    )
    conn.commit()


def get_visual_entitlement(diagram_id: str) -> tuple[str, str, str] | None:
    """Return (device_id, visual_url, study_notes_json) for a diagram_id, or None.

    visual_url == "" means entitled but not yet generated (lazy on click).
    A non-empty visual_url is immutable — reuse it, never re-generate.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT device_id, visual_url, study_notes_json FROM visual_explanations WHERE diagram_id = ?",
        (diagram_id,),
    ).fetchone()
    if row is None:
        return None
    return row["device_id"], row["visual_url"], row["study_notes_json"]


def set_visual_url(diagram_id: str, device_id: str, visual_url: str) -> bool:
    """Persist the generated visual URL. Refuses to overwrite an existing URL."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE visual_explanations SET visual_url = ? WHERE diagram_id = ? AND device_id = ? AND visual_url = ''",
        (visual_url, diagram_id, device_id),
    )
    conn.commit()
    return cur.rowcount > 0
