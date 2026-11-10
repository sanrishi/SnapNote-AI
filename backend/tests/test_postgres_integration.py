"""PostgreSQL integration tests (Neon-compatible).

Skipped unless TEST_DATABASE_URL is set. The fixture creates a throwaway schema
inside the test database, so it never touches public tables or production data.

Run locally against a Neon TEST branch:
    $env:TEST_DATABASE_URL = "postgresql://..."
    python -m pytest backend/tests/test_postgres_integration.py -q
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

from app.config import settings
from app.exceptions import CreditLimitError, InvalidInputError
from app.storage.database import PostgresDatabase, get_database, set_database
from app.storage.schema import TABLE_NAMES
from app.utils import credits_store as store
from app.utils.rate_limiter import RateLimitError, check_rate_limits
from scripts._db_tools import sqlite_rows, summarize_all
from scripts.db_migrate_postgres import _apply, _verify
from scripts.db_rollback_export import export_to_sqlite

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not configured",
)


@pytest.fixture
def pg_database(tmp_path):
    import psycopg

    url = os.environ["TEST_DATABASE_URL"]
    schema = f"snapnote_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    database = PostgresDatabase(
        url,
        pool_min_size=1,
        pool_max_size=4,
        connect_timeout_seconds=10,
        search_path=schema,
    )
    database.initialize()
    previous = get_database()
    set_database(database)
    try:
        yield database, url, schema, tmp_path
    finally:
        set_database(previous)
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def _device() -> str:
    return f"pg-{uuid.uuid4().hex}"


def test_postgres_schema_and_credit_lifecycle(pg_database):
    database, _url, _schema, _tmp = pg_database
    device = _device()
    assert store.init_device(device) == (settings.FREE_CREDITS_MONTHLY, 0)
    assert store.add_credits(device, 10) == settings.FREE_CREDITS_MONTHLY + 10
    assert store.use_credits(device, 5) == settings.FREE_CREDITS_MONTHLY + 5
    with pytest.raises(CreditLimitError):
        store.use_credits(device, 10_000)
    health = database.health()
    assert health["backend"] == "postgresql"
    assert health["reachable"] is True


def test_postgres_concurrent_deduction_is_atomic(pg_database):
    device = _device()
    store.reset_device_credits(device)
    outcomes: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            store.use_credits(device, 1)
            result = True
        except CreditLimitError:
            result = False
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(outcomes) == 10
    assert store.get_credits(device)[0] == settings.FREE_CREDITS_MONTHLY - 10


def test_postgres_order_completion_credits_once(pg_database):
    device = _device()
    store.reset_device_credits(device)
    order_id = f"order-{uuid.uuid4().hex}"
    store.put_order(order_id, device, "starter", 50, 4900)
    outcomes: list[tuple[bool, int]] = []
    lock = threading.Lock()

    def worker() -> None:
        result = store.complete_order_and_credit(
            order_id, device, f"pay-{uuid.uuid4().hex}"
        )
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(1 for completed, _ in outcomes if completed) == 1
    assert store.get_credits(device)[0] == settings.FREE_CREDITS_MONTHLY + 50
    assert store.get_order(order_id)["status"] == "completed"


def test_postgres_order_rejects_other_device(pg_database):
    owner = _device()
    stranger = _device()
    order_id = f"order-{uuid.uuid4().hex}"
    store.put_order(order_id, owner, "starter", 50, 4900)
    with pytest.raises(InvalidInputError):
        store.complete_order_and_credit(order_id, stranger, "pay-x")
    assert store.get_order(order_id)["status"] == "created"


def test_postgres_users_and_visual_immutability(pg_database):
    email = f"pg-{uuid.uuid4().hex}@example.com"
    user_id = store.create_user(email, "hash", "PG")
    assert store.use_user_credits(user_id, 5) == settings.FREE_CREDITS_MONTHLY - 5
    with pytest.raises(InvalidInputError):
        store.create_user(email, "hash")
    device = _device()
    diagram_id = f"diagram-{uuid.uuid4().hex}"
    store.record_diagram_grant(device, diagram_id, "{}")
    assert store.set_visual_result(diagram_id, device, "deterministic", visual_svg="<svg/>")
    assert not store.set_visual_result(
        diagram_id, device, "deterministic", visual_svg="<svg>other</svg>"
    )
    assert store.get_visual_entitlement(diagram_id)[3] == "<svg/>"


def test_postgres_rate_limiter(pg_database, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MIN", 2)
    device = _device()
    check_rate_limits(device)
    check_rate_limits(device)
    with pytest.raises(RateLimitError):
        check_rate_limits(device)


def test_postgres_migration_and_rollback_roundtrip(pg_database):
    database, url, schema, tmp_path = pg_database
    source_db = tmp_path / "source.db"
    from app.storage.database import SQLiteDatabase

    sqlite = SQLiteDatabase(source_db)
    sqlite.initialize()
    raw = sqlite.raw_sqlite_connection()
    device = _device()
    order_id = f"order-{uuid.uuid4().hex}"
    diagram_id = f"diagram-{uuid.uuid4().hex}"
    raw.execute(
        "INSERT INTO device_credits (device_id, credits_remaining, credits_used) "
        "VALUES (?, ?, ?)",
        (device, 42, 8),
    )
    raw.execute(
        "INSERT INTO users (id, email, password_hash, name, picture, "
        "credits_remaining, credits_used, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("user-1", "pg-migrate@example.com", "hash", "Name", "", 50, 3, 1),
    )
    raw.execute(
        "INSERT INTO razorpay_orders (order_id, device_id, plan, credits, amount, "
        "status, razorpay_payment_id, created_at) "
        "VALUES (?, ?, 'starter', 50, 4900, 'completed', 'pay-1', 2)",
        (order_id, device),
    )
    raw.execute(
        "INSERT INTO request_log (device_id, ts) VALUES (?, ?)", (device, 3)
    )
    raw.execute(
        "INSERT INTO visual_explanations (diagram_id, device_id, study_notes_json, "
        "visual_url, render_mode, visual_svg, generated_at) "
        "VALUES (?, ?, '{}', '', 'deterministic', '<svg/>', 4)",
        (diagram_id, device),
    )
    raw.commit()
    source_rows = {table: sqlite_rows(raw, table) for table in TABLE_NAMES}
    sqlite.close()

    inserted = _apply(database, source_rows)
    assert all(inserted[table] == len(source_rows[table]) for table in TABLE_NAMES)
    ok, source_summary, target_summary = _verify(source_rows, database)
    assert ok, (source_summary, target_summary)

    # Resume/retry is insert-only and remains idempotent.
    second = _apply(database, source_rows)
    assert all(second[table] == 0 for table in TABLE_NAMES)

    rollback_path = tmp_path / "rollback.db"
    manifest = export_to_sqlite(url, rollback_path, search_path=schema)
    assert manifest["tables"] == summarize_all(source_rows)
    rollback = SQLiteDatabase(rollback_path)
    rollback.initialize()
    try:
        rollback_rows = {
            table: sqlite_rows(rollback.raw_sqlite_connection(), table)
            for table in TABLE_NAMES
        }
    finally:
        rollback.close()
    assert summarize_all(rollback_rows) == summarize_all(source_rows)
