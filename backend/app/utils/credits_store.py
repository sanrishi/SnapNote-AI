"""User/credit repository shared by SQLite (local/rollback) and PostgreSQL.

Every balance mutation is a single conditional SQL statement (or one explicit
transaction for payment claims), so concurrency safety does not depend on the
engine or on application-level locking. The PostgreSQL adapter translates
``?`` placeholders and mapping rows; the domain logic below stays identical.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from typing import Mapping, TypedDict

from app.config import settings
from app.storage.database import SQLiteDatabase, get_database

logger = logging.getLogger(__name__)


class UserRecord(TypedDict):
    id: str
    email: str
    password_hash: str
    name: str
    picture: str
    credits_remaining: int
    credits_used: int
    created_at: int


class OrderRecord(TypedDict):
    order_id: str
    device_id: str
    plan: str
    credits: int
    amount: int
    status: str
    razorpay_payment_id: str | None


def _as_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value or 0))


def _as_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def _as_optional_str(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


def _user_record(row: Mapping[str, object]) -> UserRecord:
    return UserRecord(
        id=_as_str(row, "id"),
        email=_as_str(row, "email"),
        password_hash=_as_str(row, "password_hash"),
        name=_as_str(row, "name"),
        picture=_as_str(row, "picture"),
        credits_remaining=_as_int(row, "credits_remaining"),
        credits_used=_as_int(row, "credits_used"),
        created_at=_as_int(row, "created_at"),
    )


def _order_record(row: Mapping[str, object]) -> OrderRecord:
    return OrderRecord(
        order_id=_as_str(row, "order_id"),
        device_id=_as_str(row, "device_id"),
        plan=_as_str(row, "plan"),
        credits=_as_int(row, "credits"),
        amount=_as_int(row, "amount"),
        status=_as_str(row, "status"),
        razorpay_payment_id=_as_optional_str(row, "razorpay_payment_id"),
    )


def _get_conn() -> sqlite3.Connection:
    """Legacy raw SQLite handle for tests and one-off migration scripts.

    Production/domain code must use the repository functions below; this helper
    exists because older regression tests seed balances with direct SQL.
    """
    database = get_database()
    if not isinstance(database, SQLiteDatabase):
        raise RuntimeError(
            "Raw _get_conn() is SQLite-only; use the repository functions "
            "when DATABASE_URL is configured"
        )
    return database.raw_sqlite_connection()


# ── Device credits ──────────────────────────────────────────────────────────


def init_device(device_id: str) -> tuple[int, int]:
    database = get_database()
    database.execute(
        "INSERT INTO device_credits (device_id, credits_remaining, credits_used) "
        "VALUES (?, ?, 0) ON CONFLICT (device_id) DO NOTHING",
        (device_id, settings.FREE_CREDITS_MONTHLY),
    )
    row = database.execute(
        "SELECT credits_remaining, credits_used FROM device_credits WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("device credits row could not be created")
    return _as_int(row, "credits_remaining"), _as_int(row, "credits_used")


def get_credits(device_id: str) -> tuple[int, int]:
    row = get_database().execute(
        "SELECT credits_remaining, credits_used FROM device_credits WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    if row is None:
        return init_device(device_id)
    return _as_int(row, "credits_remaining"), _as_int(row, "credits_used")


def peek_credits(device_id: str) -> tuple[int, int]:
    """Read a balance without creating a device row."""
    row = get_database().execute(
        "SELECT credits_remaining, credits_used FROM device_credits WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    if row is None:
        return settings.FREE_CREDITS_MONTHLY, 0
    return _as_int(row, "credits_remaining"), _as_int(row, "credits_used")


def reset_device_credits(device_id: str) -> tuple[int, int]:
    """Give a device its full free allowance again (DEBUG-only endpoint)."""
    get_database().execute(
        "INSERT INTO device_credits (device_id, credits_remaining, credits_used) "
        "VALUES (?, ?, 0) ON CONFLICT (device_id) DO UPDATE SET "
        "credits_remaining = excluded.credits_remaining, credits_used = 0",
        (device_id, settings.FREE_CREDITS_MONTHLY),
    )
    return settings.FREE_CREDITS_MONTHLY, 0


def add_credits(device_id: str, amount: int) -> int:
    if amount < 0:
        raise ValueError("credit amount cannot be negative")
    database = get_database()
    init_device(device_id)
    database.execute(
        "UPDATE device_credits SET credits_remaining = credits_remaining + ? "
        "WHERE device_id = ?",
        (amount, device_id),
    )
    remaining, _ = get_credits(device_id)
    logger.info("Added %d credits to %s (now %d)", amount, device_id[:8], remaining)
    return remaining


def use_credits(device_id: str, amount: int) -> int:
    """Atomically deduct credits. Raises CreditLimitError if insufficient.

    One conditional UPDATE is atomic on both engines: concurrent requests
    cannot both observe the same balance and overspend.
    """
    from app.exceptions import CreditLimitError

    if amount <= 0:
        raise ValueError("credit amount must be positive")
    database = get_database()
    init_device(device_id)
    updated = database.execute(
        "UPDATE device_credits "
        "SET credits_remaining = credits_remaining - ?, "
        "    credits_used = credits_used + ? "
        "WHERE device_id = ? AND credits_remaining >= ?",
        (amount, amount, device_id, amount),
    )
    if updated.rowcount == 0:
        raise CreditLimitError()
    new_remaining, _ = get_credits(device_id)
    logger.info("Used %d credits for %s (now %d)", amount, device_id[:8], new_remaining)
    return new_remaining


def try_use_credits(device_id: str, amount: int) -> bool:
    """Non-raising variant — returns False if insufficient."""
    from app.exceptions import CreditLimitError

    try:
        use_credits(device_id, amount)
        return True
    except CreditLimitError:
        return False


# ── Explain Visually entitlements ───────────────────────────────────────────


def record_diagram_grant(device_id: str, diagram_id: str, study_notes_json: str) -> None:
    """Record a completed 5-credit diagram result (idempotent)."""
    get_database().execute(
        "INSERT INTO visual_explanations "
        "(diagram_id, device_id, study_notes_json, generated_at) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (diagram_id) DO NOTHING",
        (diagram_id, device_id, study_notes_json, int(time.time())),
    )


def get_visual_entitlement(diagram_id: str) -> tuple[str, str, str, str, str] | None:
    """Return (device_id, visual_url, render_mode, visual_svg, study_notes_json)."""
    row = get_database().execute(
        "SELECT device_id, visual_url, render_mode, visual_svg, study_notes_json "
        "FROM visual_explanations WHERE diagram_id = ?",
        (diagram_id,),
    ).fetchone()
    if row is None:
        return None
    return (
        _as_str(row, "device_id"),
        _as_str(row, "visual_url"),
        _as_str(row, "render_mode"),
        _as_str(row, "visual_svg"),
        _as_str(row, "study_notes_json"),
    )


def set_visual_result(
    diagram_id: str,
    device_id: str,
    render_mode: str,
    visual_url: str = "",
    visual_svg: str = "",
) -> bool:
    """Persist the generated visual result. Refuses to overwrite an existing result."""
    updated = get_database().execute(
        "UPDATE visual_explanations SET visual_url = ?, render_mode = ?, visual_svg = ? "
        "WHERE diagram_id = ? AND device_id = ? AND visual_url = '' AND visual_svg = ''",
        (visual_url, render_mode, visual_svg, diagram_id, device_id),
    )
    return updated.rowcount > 0


# ── Payment orders ──────────────────────────────────────────────────────────


def get_order(order_id: str) -> OrderRecord | None:
    row = get_database().execute(
        "SELECT order_id, device_id, plan, credits, amount, status, "
        "razorpay_payment_id FROM razorpay_orders WHERE order_id = ?",
        (order_id,),
    ).fetchone()
    return None if row is None else _order_record(row)


def put_order(
    order_id: str, device_id: str, plan: str, credits: int, amount: int
) -> None:
    get_database().execute(
        "INSERT INTO razorpay_orders "
        "(order_id, device_id, plan, credits, amount, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'created', ?) ON CONFLICT (order_id) DO NOTHING",
        (order_id, device_id, plan, credits, amount, int(time.time())),
    )


def complete_order_and_credit(
    order_id: str, device_id: str, payment_id: str
) -> tuple[bool, int]:
    """Atomically claim a paid order and credit its device exactly once.

    The status claim and the balance update share one transaction. A second
    concurrent verify blocks on the order row, then sees ``status='completed'``
    and adds nothing. Credits always go to the device stored on the order.
    """
    from app.exceptions import InvalidInputError

    database = get_database()
    init_device(device_id)
    try:
        with database.transaction() as tx:
            row = tx.execute(
                "SELECT device_id, credits FROM razorpay_orders "
                "WHERE order_id = ?",
                (order_id,),
            ).fetchone()
            if row is None:
                balance = tx.execute(
                    "SELECT credits_remaining FROM device_credits WHERE device_id = ?",
                    (device_id,),
                ).fetchone()
                return False, _as_int(balance, "credits_remaining") if balance else 0
            if _as_str(row, "device_id") != device_id:
                raise InvalidInputError(message="Order does not belong to this device")
            claimed = tx.execute(
                "UPDATE razorpay_orders SET status = 'completed', "
                "razorpay_payment_id = ? WHERE order_id = ? AND status = 'created'",
                (payment_id, order_id),
            )
            if claimed.rowcount == 0:
                balance = tx.execute(
                    "SELECT credits_remaining FROM device_credits WHERE device_id = ?",
                    (device_id,),
                ).fetchone()
                return False, _as_int(balance, "credits_remaining") if balance else 0
            credits = _as_int(row, "credits")
            tx.execute(
                "UPDATE device_credits SET credits_remaining = credits_remaining + ? "
                "WHERE device_id = ?",
                (credits, device_id),
            )
            balance = tx.execute(
                "SELECT credits_remaining FROM device_credits WHERE device_id = ?",
                (device_id,),
            ).fetchone()
            remaining = _as_int(balance, "credits_remaining") if balance else 0
    except InvalidInputError:
        raise
    except Exception as exc:
        logger.error("Order completion failed for %s: %s", order_id[:12], type(exc).__name__)
        raise
    logger.info(
        "Order %s credited %d for %s (now %d)",
        order_id[:12],
        credits,
        device_id[:8],
        remaining,
    )
    return True, remaining


# ── Users (credits stick to email, not clearable localStorage) ──────────────


def create_user(email: str, password_hash: str, name: str = "", picture: str = "") -> str:
    user_id = uuid.uuid4().hex
    database = get_database()
    try:
        database.execute(
            "INSERT INTO users "
            "(id, email, password_hash, name, picture, credits_remaining, "
            " credits_used, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (
                user_id,
                email.lower().strip(),
                password_hash,
                name.strip(),
                picture.strip(),
                settings.FREE_CREDITS_MONTHLY,
                int(time.time()),
            ),
        )
        return user_id
    except Exception as exc:
        if database.is_unique_violation(exc):
            from app.exceptions import InvalidInputError

            raise InvalidInputError(message="Email already registered. Please log in.") from exc
        raise


def set_user_picture(user_id: str, picture: str) -> None:
    get_database().execute(
        "UPDATE users SET picture = ? WHERE id = ? AND picture = ''",
        (picture.strip(), user_id),
    )


def get_user_by_email(email: str) -> UserRecord | None:
    row = get_database().execute(
        "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    return None if row is None else _user_record(row)


def get_user_by_id(user_id: str) -> UserRecord | None:
    row = get_database().execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return None if row is None else _user_record(row)


def get_user_credits(user_id: str) -> tuple[int, int]:
    row = get_user_by_id(user_id)
    if row is None:
        from app.exceptions import AuthError

        raise AuthError(message="User not found")
    return row["credits_remaining"], row["credits_used"]


def use_user_credits(user_id: str, amount: int) -> int:
    from app.exceptions import AuthError, CreditLimitError

    if amount <= 0:
        raise ValueError("credit amount must be positive")
    database = get_database()
    row = database.execute(
        "SELECT credits_remaining FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if row is None:
        raise AuthError(message="User not found")
    updated = database.execute(
        "UPDATE users SET credits_remaining = credits_remaining - ?, "
        "credits_used = credits_used + ? "
        "WHERE id = ? AND credits_remaining >= ?",
        (amount, amount, user_id, amount),
    )
    if updated.rowcount == 0:
        raise CreditLimitError()
    refreshed = get_user_by_id(user_id)
    if refreshed is None:
        raise AuthError(message="User not found")
    logger.info(
        "Used %d credits for user %s (now %d)",
        amount,
        user_id[:8],
        refreshed["credits_remaining"],
    )
    return refreshed["credits_remaining"]


def add_user_credits(user_id: str, amount: int) -> int:
    if amount < 0:
        raise ValueError("credit amount cannot be negative")
    get_database().execute(
        "UPDATE users SET credits_remaining = credits_remaining + ? WHERE id = ?",
        (amount, user_id),
    )
    row = get_user_by_id(user_id)
    if row is None:
        from app.exceptions import AuthError

        raise AuthError(message="User not found")
    return row["credits_remaining"]
