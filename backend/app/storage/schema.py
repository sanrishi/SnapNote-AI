"""Canonical schema for the SnapNote user/credit database.

SQLite is the legacy/local backend and remains the rollback target. PostgreSQL
(Neon in production) is the durable backend. Both schemas are intentionally
kept in lockstep: every column, constraint, and index exists in both so the
same repository code can run against either engine.

The schema is additive-only. Existing databases may predate columns that were
added over time, so the initializers also run idempotent ``ALTER TABLE``
statements where the engine supports them.
"""

from __future__ import annotations

TABLE_NAMES: tuple[str, ...] = (
    "users",
    "device_credits",
    "razorpay_orders",
    "request_log",
    "visual_explanations",
)

# Column order is part of the migration contract: backups, merges, and the
# PostgreSQL importer all use it so a row can round-trip without guessing.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "users": (
        "id",
        "email",
        "password_hash",
        "name",
        "picture",
        "credits_remaining",
        "credits_used",
        "created_at",
    ),
    "device_credits": ("device_id", "credits_remaining", "credits_used"),
    "razorpay_orders": (
        "order_id",
        "device_id",
        "plan",
        "credits",
        "amount",
        "status",
        "razorpay_payment_id",
        "created_at",
    ),
    "request_log": ("id", "device_id", "ts"),
    "visual_explanations": (
        "diagram_id",
        "device_id",
        "study_notes_json",
        "visual_url",
        "render_mode",
        "visual_svg",
        "generated_at",
    ),
}

KEY_COLUMNS: dict[str, str] = {
    "users": "id",
    "device_credits": "device_id",
    "razorpay_orders": "order_id",
    "request_log": "id",
    "visual_explanations": "diagram_id",
}

SCHEMA_VERSION = 1

# Executed in order on every SQLite connection initialization. Every statement
# is idempotent so an existing credits.db is upgraded in place.
SQLITE_SCHEMA: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS snapnote_schema ("
    "  id INTEGER PRIMARY KEY CHECK (id = 1),"
    "  version INTEGER NOT NULL"
    ")",
    "CREATE TABLE IF NOT EXISTS device_credits ("
    "  device_id TEXT PRIMARY KEY,"
    "  credits_remaining INTEGER NOT NULL DEFAULT 50 CHECK(credits_remaining >= 0),"
    "  credits_used INTEGER NOT NULL DEFAULT 0 CHECK(credits_used >= 0)"
    ")",
    "CREATE TABLE IF NOT EXISTS users ("
    "  id TEXT PRIMARY KEY,"
    "  email TEXT UNIQUE NOT NULL,"
    "  password_hash TEXT NOT NULL,"
    "  name TEXT NOT NULL DEFAULT '',"
    "  picture TEXT NOT NULL DEFAULT '',"
    "  credits_remaining INTEGER NOT NULL DEFAULT 50 CHECK(credits_remaining >= 0),"
    "  credits_used INTEGER NOT NULL DEFAULT 0 CHECK(credits_used >= 0),"
    "  created_at INTEGER NOT NULL"
    ")",
    "CREATE TABLE IF NOT EXISTS visual_explanations ("
    "  diagram_id TEXT PRIMARY KEY,"
    "  device_id TEXT NOT NULL,"
    "  study_notes_json TEXT NOT NULL DEFAULT '',"
    "  visual_url TEXT NOT NULL DEFAULT '',"
    "  render_mode TEXT NOT NULL DEFAULT '',"
    "  visual_svg TEXT NOT NULL DEFAULT '',"
    "  generated_at INTEGER NOT NULL DEFAULT 0"
    ")",
    "CREATE TABLE IF NOT EXISTS razorpay_orders ("
    "  order_id TEXT PRIMARY KEY,"
    "  device_id TEXT NOT NULL,"
    "  plan TEXT NOT NULL,"
    "  credits INTEGER NOT NULL,"
    "  amount INTEGER NOT NULL,"
    "  status TEXT NOT NULL DEFAULT 'created',"
    "  razorpay_payment_id TEXT,"
    "  created_at INTEGER NOT NULL"
    ")",
    "CREATE TABLE IF NOT EXISTS request_log ("
    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "  device_id TEXT NOT NULL,"
    "  ts INTEGER NOT NULL"
    ")",
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    "CREATE INDEX IF NOT EXISTS idx_visual_device ON visual_explanations(device_id)",
    "CREATE INDEX IF NOT EXISTS idx_request_log_device_ts ON request_log(device_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_orders_device ON razorpay_orders(device_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON razorpay_orders(status)",
)

# Columns added after the original credits.db shipped. SQLite has no
# ``ADD COLUMN IF NOT EXISTS``, so the initializer catches "duplicate column".
SQLITE_LEGACY_ALTERS: tuple[str, ...] = (
    "ALTER TABLE users ADD COLUMN picture TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE visual_explanations ADD COLUMN study_notes_json TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE visual_explanations ADD COLUMN render_mode TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE visual_explanations ADD COLUMN visual_svg TEXT NOT NULL DEFAULT ''",
)

SQLITE_TRIGGERS: tuple[str, ...] = (
    "CREATE TRIGGER IF NOT EXISTS trg_device_credits_no_negative "
    "BEFORE UPDATE ON device_credits FOR EACH ROW "
    "WHEN NEW.credits_remaining < 0 BEGIN "
    "SELECT RAISE(ABORT, 'credits_remaining cannot be negative'); END",
    "CREATE TRIGGER IF NOT EXISTS trg_users_credits_no_negative "
    "BEFORE UPDATE ON users FOR EACH ROW "
    "WHEN NEW.credits_remaining < 0 BEGIN "
    "SELECT RAISE(ABORT, 'credits_remaining cannot be negative'); END",
)

# PostgreSQL runs as a single transaction under an advisory lock, so the DDL
# below can use the richer server syntax (BIGSERIAL, ADD COLUMN IF NOT EXISTS).
POSTGRES_SCHEMA: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS snapnote_schema ("
    "  id SMALLINT PRIMARY KEY CHECK (id = 1),"
    "  version INTEGER NOT NULL"
    ")",
    "CREATE TABLE IF NOT EXISTS device_credits ("
    "  device_id TEXT PRIMARY KEY,"
    "  credits_remaining INTEGER NOT NULL DEFAULT 50 CHECK(credits_remaining >= 0),"
    "  credits_used INTEGER NOT NULL DEFAULT 0 CHECK(credits_used >= 0)"
    ")",
    "CREATE TABLE IF NOT EXISTS users ("
    "  id TEXT PRIMARY KEY,"
    "  email TEXT UNIQUE NOT NULL,"
    "  password_hash TEXT NOT NULL,"
    "  name TEXT NOT NULL DEFAULT '',"
    "  picture TEXT NOT NULL DEFAULT '',"
    "  credits_remaining INTEGER NOT NULL DEFAULT 50 CHECK(credits_remaining >= 0),"
    "  credits_used INTEGER NOT NULL DEFAULT 0 CHECK(credits_used >= 0),"
    "  created_at BIGINT NOT NULL"
    ")",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS picture TEXT NOT NULL DEFAULT ''",
    "CREATE TABLE IF NOT EXISTS visual_explanations ("
    "  diagram_id TEXT PRIMARY KEY,"
    "  device_id TEXT NOT NULL,"
    "  study_notes_json TEXT NOT NULL DEFAULT '',"
    "  visual_url TEXT NOT NULL DEFAULT '',"
    "  render_mode TEXT NOT NULL DEFAULT '',"
    "  visual_svg TEXT NOT NULL DEFAULT '',"
    "  generated_at BIGINT NOT NULL DEFAULT 0"
    ")",
    "ALTER TABLE visual_explanations ADD COLUMN IF NOT EXISTS study_notes_json TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE visual_explanations ADD COLUMN IF NOT EXISTS render_mode TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE visual_explanations ADD COLUMN IF NOT EXISTS visual_svg TEXT NOT NULL DEFAULT ''",
    "CREATE TABLE IF NOT EXISTS razorpay_orders ("
    "  order_id TEXT PRIMARY KEY,"
    "  device_id TEXT NOT NULL,"
    "  plan TEXT NOT NULL,"
    "  credits INTEGER NOT NULL,"
    "  amount INTEGER NOT NULL,"
    "  status TEXT NOT NULL DEFAULT 'created',"
    "  razorpay_payment_id TEXT,"
    "  created_at BIGINT NOT NULL"
    ")",
    "CREATE TABLE IF NOT EXISTS request_log ("
    "  id BIGSERIAL PRIMARY KEY,"
    "  device_id TEXT NOT NULL,"
    "  ts BIGINT NOT NULL"
    ")",
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    "CREATE INDEX IF NOT EXISTS idx_visual_device ON visual_explanations(device_id)",
    "CREATE INDEX IF NOT EXISTS idx_request_log_device_ts ON request_log(device_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_orders_device ON razorpay_orders(device_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON razorpay_orders(status)",
    "INSERT INTO snapnote_schema (id, version) VALUES (1, 1) "
    "ON CONFLICT (id) DO NOTHING",
)

SQLITE_SCHEMA = SQLITE_SCHEMA + (
    "INSERT INTO snapnote_schema (id, version) VALUES (1, 1) "
    "ON CONFLICT (id) DO NOTHING",
)
