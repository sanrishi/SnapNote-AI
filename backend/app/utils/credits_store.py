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
        _local.conn = sqlite3.connect(str(_db_path), timeout=30.0, check_same_thread=False, isolation_level=None)
        _local.conn.row_factory = sqlite3.Row
        # WAL for concurrent readers/writers on the single Render disk.
        try:
            _local.conn.execute("PRAGMA journal_mode=WAL;")
            _local.conn.execute("PRAGMA synchronous=NORMAL;")
            _local.conn.execute("PRAGMA foreign_keys=ON;")
        except sqlite3.Error:
            pass
        _local.conn.execute(
            "CREATE TABLE IF NOT EXISTS device_credits ("
            "  device_id TEXT PRIMARY KEY,"
            "  credits_remaining INTEGER NOT NULL DEFAULT 50 CHECK(credits_remaining >= 0),"
            "  credits_used INTEGER NOT NULL DEFAULT 0 CHECK(credits_used >= 0)"
            ")"
        )
        # Enforce non-negative balance on existing DBs (ALTER cannot add CHECK).
        try:
            _local.conn.execute(
                "CREATE TRIGGER IF NOT EXISTS trg_device_credits_no_negative "
                "BEFORE UPDATE ON device_credits FOR EACH ROW "
                "WHEN NEW.credits_remaining < 0 BEGIN "
                "SELECT RAISE(ABORT, 'credits_remaining cannot be negative'); END;"
            )
        except sqlite3.Error:
            pass
        _local.conn.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "  id TEXT PRIMARY KEY,"
            "  email TEXT UNIQUE NOT NULL,"
            "  password_hash TEXT NOT NULL,"
            "  name TEXT NOT NULL DEFAULT '',"
            "  picture TEXT NOT NULL DEFAULT '',"
            "  credits_remaining INTEGER NOT NULL DEFAULT 50 CHECK(credits_remaining >= 0),"
            "  credits_used INTEGER NOT NULL DEFAULT 0 CHECK(credits_used >= 0),"
            "  created_at INTEGER NOT NULL"
            ")"
        )
        # Add picture column to existing DBs (idempotent migration).
        try:
            _local.conn.execute("ALTER TABLE users ADD COLUMN picture TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            _local.conn.execute(
                "CREATE TRIGGER IF NOT EXISTS trg_users_credits_no_negative "
                "BEFORE UPDATE ON users FOR EACH ROW "
                "WHEN NEW.credits_remaining < 0 BEGIN "
                "SELECT RAISE(ABORT, 'credits_remaining cannot be negative'); END;"
            )
        except sqlite3.Error:
            pass
        _local.conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        _local.conn.execute(
            "CREATE TABLE IF NOT EXISTS visual_explanations ("
            "  diagram_id TEXT PRIMARY KEY,"
            "  device_id TEXT NOT NULL,"
            "  study_notes_json TEXT NOT NULL DEFAULT '',"
            "  visual_url TEXT NOT NULL DEFAULT '',"
            "  render_mode TEXT NOT NULL DEFAULT '',"
            "  visual_svg TEXT NOT NULL DEFAULT '',"
            "  generated_at INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        _local.conn.execute(
            "CREATE TABLE IF NOT EXISTS razorpay_orders ("
            "  order_id TEXT PRIMARY KEY,"
            "  device_id TEXT NOT NULL,"
            "  plan TEXT NOT NULL,"
            "  credits INTEGER NOT NULL,"
            "  amount INTEGER NOT NULL,"
            "  status TEXT NOT NULL DEFAULT 'created',"
            "  razorpay_payment_id TEXT,"
            "  created_at INTEGER NOT NULL"
            ")"
        )
        _local.conn.execute(
            "CREATE TABLE IF NOT EXISTS request_log ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  device_id TEXT NOT NULL,"
            "  ts INTEGER NOT NULL"
            ")"
        )
        _local.conn.execute("CREATE INDEX IF NOT EXISTS idx_request_log_device_ts ON request_log(device_id, ts);")
        _local.conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_device ON visual_explanations(device_id);")
        try:
            _local.conn.execute(
                "ALTER TABLE visual_explanations ADD COLUMN study_notes_json TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        try:
            _local.conn.execute(
                "ALTER TABLE visual_explanations ADD COLUMN render_mode TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        try:
            _local.conn.execute(
                "ALTER TABLE visual_explanations ADD COLUMN visual_svg TEXT NOT NULL DEFAULT ''"
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
    """Atomically deduct credits. Raises CreditLimitError if insufficient.

    Uses BEGIN IMMEDIATE so concurrent requests serialize; the DB CHECK +
    trigger prevents a negative balance even under race.
    """
    from app.exceptions import CreditLimitError

    conn = _get_conn()
    # Ensure row exists before the atomic block (INSERT OR IGNORE is idempotent).
    init_device(device_id)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT credits_remaining, credits_used FROM device_credits WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            raise CreditLimitError()
        remaining = row["credits_remaining"]
        if remaining < amount:
            conn.execute("ROLLBACK")
            raise CreditLimitError()
        conn.execute(
            "UPDATE device_credits SET credits_remaining = credits_remaining - ?, credits_used = credits_used + ? WHERE device_id = ?",
            (amount, amount, device_id),
        )
        conn.execute("COMMIT")
    except CreditLimitError:
        raise
    except sqlite3.Error as e:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        # CHECK / trigger abort surfaces as OperationalError — map to 429.
        if "negative" in str(e).lower() or "CHECK" in str(e):
            raise CreditLimitError() from e
        raise
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


def get_visual_entitlement(diagram_id: str) -> tuple[str, str, str, str, str] | None:
    """Return (device_id, visual_url, render_mode, visual_svg, study_notes_json).

    visual_url == "" and visual_svg == "" means entitled but not yet generated
    (lazy on click). A non-empty visual_url/visual_svg is immutable — reuse it,
    never re-generate. render_mode is "deterministic" | "generative" | "".
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT device_id, visual_url, render_mode, visual_svg, study_notes_json "
        "FROM visual_explanations WHERE diagram_id = ?",
        (diagram_id,),
    ).fetchone()
    if row is None:
        return None
    return (
        row["device_id"],
        row["visual_url"],
        row["render_mode"],
        row["visual_svg"],
        row["study_notes_json"],
    )


def set_visual_result(
    diagram_id: str,
    device_id: str,
    render_mode: str,
    visual_url: str = "",
    visual_svg: str = "",
) -> bool:
    """Persist the generated visual result. Refuses to overwrite an existing result."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE visual_explanations SET visual_url = ?, render_mode = ?, visual_svg = ? "
        "WHERE diagram_id = ? AND device_id = ? AND visual_url = '' AND visual_svg = ''",
        (visual_url, render_mode, visual_svg, diagram_id, device_id),
    )
    conn.commit()
    return cur.rowcount > 0


# ── Users (for signup/login — credits stick to email, not clearable localStorage) ──

def create_user(email: str, password_hash: str, name: str = "", picture: str = "") -> str:
    import uuid

    user_id = uuid.uuid4().hex
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, name, picture, credits_remaining, credits_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (user_id, email.lower().strip(), password_hash, name.strip(), picture.strip(), settings.FREE_CREDITS_MONTHLY, int(time.time())),
        )
        conn.commit()
        return user_id
    except sqlite3.IntegrityError as e:
        if "UNIQUE" in str(e) or "users.email" in str(e):
            from app.exceptions import InvalidInputError

            raise InvalidInputError(message="Email already registered. Please log in.")
        raise


def set_user_picture(user_id: str, picture: str) -> None:
    conn = _get_conn()
    conn.execute("UPDATE users SET picture = ? WHERE id = ? AND picture = ''", (picture.strip(), user_id))
    conn.commit()


def get_user_by_email(email: str) -> sqlite3.Row | None:
    conn = _get_conn()
    return conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()


def get_user_by_id(user_id: str) -> sqlite3.Row | None:
    conn = _get_conn()
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_credits(user_id: str) -> tuple[int, int]:
    row = get_user_by_id(user_id)
    if row is None:
        from app.exceptions import AuthError

        raise AuthError(message="User not found")
    return row["credits_remaining"], row["credits_used"]


def use_user_credits(user_id: str, amount: int) -> int:
    from app.exceptions import CreditLimitError

    conn = _get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT credits_remaining FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            from app.exceptions import AuthError

            raise AuthError(message="User not found")
        if row["credits_remaining"] < amount:
            conn.execute("ROLLBACK")
            raise CreditLimitError()
        conn.execute(
            "UPDATE users SET credits_remaining = credits_remaining - ?, credits_used = credits_used + ? WHERE id = ?",
            (amount, amount, user_id),
        )
        conn.execute("COMMIT")
    except CreditLimitError:
        raise
    except sqlite3.Error as e:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        if "negative" in str(e).lower() or "CHECK" in str(e):
            raise CreditLimitError() from e
        raise
    row = get_user_by_id(user_id)
    assert row is not None
    logger.info("Used %d credits for user %s (now %d)", amount, user_id[:8], row["credits_remaining"])
    return row["credits_remaining"]


def add_user_credits(user_id: str, amount: int) -> int:
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET credits_remaining = credits_remaining + ? WHERE id = ?", (amount, user_id)
    )
    conn.commit()
    row = get_user_by_id(user_id)
    assert row is not None
    return row["credits_remaining"]
