import logging
import sqlite3
import time

from app.config import settings
from app.exceptions import SnapNoteError

logger = logging.getLogger(__name__)


class RateLimitError(SnapNoteError):
    def __init__(self, message: str = "Rate limit exceeded. Try again in a minute."):
        super().__init__(message, status_code=429)


def _conn() -> sqlite3.Connection:
    from app.utils.credits_store import _get_conn

    return _get_conn()


def check_rate_limits(device_id: str) -> None:
    """Enforce per-device rate limits from settings.

    Uses the shared credits.db (request_log table) so limits survive restarts
    and work with the single Render disk. Sliding windows:
      - per-minute: RATE_LIMIT_PER_MIN requests in last 60s
      - per-day: DAILY_REQ_LIMIT requests in last 86400s
    """
    if not device_id or not device_id.strip():
        return
    now = int(time.time())
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        # Opportunistic cleanup (old rows outside daily window).
        conn.execute("DELETE FROM request_log WHERE ts < ?", (now - 86400,))
        per_min = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE device_id = ? AND ts > ?",
            (device_id, now - 60),
        ).fetchone()[0]
        if per_min >= settings.RATE_LIMIT_PER_MIN:
            conn.execute("ROLLBACK")
            raise RateLimitError("Too many requests — please wait a minute and try again.")

        per_day = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE device_id = ? AND ts > ?",
            (device_id, now - 86400),
        ).fetchone()[0]
        if per_day >= settings.DAILY_REQ_LIMIT:
            conn.execute("ROLLBACK")
            raise RateLimitError("Daily request limit reached. Try again tomorrow.")

        conn.execute("INSERT INTO request_log (device_id, ts) VALUES (?, ?)", (device_id, now))
        conn.execute("COMMIT")
    except RateLimitError:
        raise
    except sqlite3.Error as e:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        logger.warning("Rate limiter DB error: %s", e)
        # Fail open on DB error so a transient SQLite issue doesn't block all users.
        return
