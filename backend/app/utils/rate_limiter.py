"""Per-device sliding-window rate limits, stored in the durable database.

The count-and-insert pair runs in one transaction. SQLite uses BEGIN IMMEDIATE
(a database-wide write lock); PostgreSQL locks the device's credit row with
SELECT ... FOR UPDATE so two simultaneous requests for the same device cannot
both pass the limit check.
"""

from __future__ import annotations

import logging
import time

from app.config import settings
from app.exceptions import SnapNoteError
from app.storage.database import get_database

logger = logging.getLogger(__name__)


class RateLimitError(SnapNoteError):
    def __init__(self, message: str = "Rate limit exceeded. Try again in a minute."):
        super().__init__(message, status_code=429)


def check_rate_limits(device_id: str) -> None:
    """Enforce per-minute and per-day limits; fail open on storage errors."""
    if not device_id or not device_id.strip():
        return
    now = int(time.time())
    database = get_database()
    try:
        with database.transaction() as tx:
            tx.execute(
                "INSERT INTO device_credits (device_id, credits_remaining, credits_used) "
                "VALUES (?, ?, 0) ON CONFLICT (device_id) DO NOTHING",
                (device_id, settings.FREE_CREDITS_MONTHLY),
            )
            if database.kind == "postgresql":
                tx.execute(
                    "SELECT device_id FROM device_credits WHERE device_id = ? FOR UPDATE",
                    (device_id,),
                )
            # Opportunistic cleanup (old rows outside the daily window).
            tx.execute("DELETE FROM request_log WHERE ts < ?", (now - 86400,))
            per_min_row = tx.execute(
                "SELECT COUNT(*) AS n FROM request_log "
                "WHERE device_id = ? AND ts > ?",
                (device_id, now - 60),
            ).fetchone()
            per_min = int(per_min_row["n"]) if per_min_row else 0
            if per_min >= settings.RATE_LIMIT_PER_MIN:
                raise RateLimitError(
                    "Too many requests — please wait a minute and try again."
                )

            per_day_row = tx.execute(
                "SELECT COUNT(*) AS n FROM request_log "
                "WHERE device_id = ? AND ts > ?",
                (device_id, now - 86400),
            ).fetchone()
            per_day = int(per_day_row["n"]) if per_day_row else 0
            if per_day >= settings.DAILY_REQ_LIMIT:
                raise RateLimitError(
                    "Daily request limit reached. Try again tomorrow."
                )

            tx.execute(
                "INSERT INTO request_log (device_id, ts) VALUES (?, ?)",
                (device_id, now),
            )
    except RateLimitError:
        raise
    except Exception as exc:
        # Fail open on a transient storage issue rather than blocking every
        # student; the error is logged so a persistent outage is visible.
        logger.warning("Rate limiter DB error: %s", type(exc).__name__)
        return
