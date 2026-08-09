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
            "  credits_used INTEGER NOT NULL DEFAULT 0,"
            "  last_diagram_paid_at INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        try:
            _local.conn.execute(
                "ALTER TABLE device_credits ADD COLUMN last_diagram_paid_at INTEGER NOT NULL DEFAULT 0"
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


def mark_diagram_paid(device_id: str) -> None:
    conn = _get_conn()
    init_device(device_id)
    conn.execute(
        "UPDATE device_credits SET last_diagram_paid_at = ? WHERE device_id = ?",
        (int(time.time()), device_id),
    )
    conn.commit()


def diagram_paid_recently(device_id: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT last_diagram_paid_at FROM device_credits WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    if row is None or row["last_diagram_paid_at"] == 0:
        return False
    return (int(time.time()) - row["last_diagram_paid_at"]) <= settings.REGENERATE_WINDOW_SECONDS
