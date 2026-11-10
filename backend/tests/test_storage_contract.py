"""Engine-neutral storage contract, exercised against SQLite.

The same repository functions run on PostgreSQL in
``test_postgres_integration.py``; these tests catch SQL dialect mistakes locally.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from app.config import settings
from app.exceptions import CreditLimitError, InvalidInputError
from app.storage.database import SQLiteDatabase, get_database, set_database
from app.utils import credits_store as store
from app.utils.rate_limiter import RateLimitError, check_rate_limits


@pytest.fixture
def sqlite_db(tmp_path):
    previous = get_database()
    database = SQLiteDatabase(tmp_path / "credits.db")
    database.initialize()
    set_database(database)
    try:
        yield database
    finally:
        set_database(previous)


def _device() -> str:
    return f"test-{uuid.uuid4().hex}"


def test_sqlite_schema_version(sqlite_db):
    row = sqlite_db.execute(
        "SELECT version FROM snapnote_schema WHERE id = 1"
    ).fetchone()
    assert row is not None
    assert int(row["version"]) == 1


def test_credit_lifecycle(sqlite_db):
    device = _device()
    assert store.init_device(device) == (settings.FREE_CREDITS_MONTHLY, 0)
    assert store.get_credits(device) == (settings.FREE_CREDITS_MONTHLY, 0)
    assert store.add_credits(device, 10) == settings.FREE_CREDITS_MONTHLY + 10
    assert store.use_credits(device, 5) == settings.FREE_CREDITS_MONTHLY + 5
    remaining, used = store.get_credits(device)
    assert remaining == settings.FREE_CREDITS_MONTHLY + 5
    assert used == 5
    assert store.peek_credits("never-created") == (settings.FREE_CREDITS_MONTHLY, 0)
    assert store.reset_device_credits(device) == (settings.FREE_CREDITS_MONTHLY, 0)


def test_overdraft_is_rejected(sqlite_db):
    device = _device()
    store.init_device(device)
    with pytest.raises(CreditLimitError):
        store.use_credits(device, settings.FREE_CREDITS_MONTHLY + 1)
    assert store.get_credits(device)[0] == settings.FREE_CREDITS_MONTHLY


def test_concurrent_deduction_is_atomic(sqlite_db):
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


def test_concurrent_order_completion_credits_once(sqlite_db):
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


def test_order_cannot_be_credited_to_another_device(sqlite_db):
    owner = _device()
    stranger = _device()
    order_id = f"order-{uuid.uuid4().hex}"
    store.put_order(order_id, owner, "starter", 50, 4900)
    with pytest.raises(InvalidInputError):
        store.complete_order_and_credit(order_id, stranger, "pay-x")
    assert store.get_credits(stranger)[0] == settings.FREE_CREDITS_MONTHLY
    assert store.get_order(order_id)["status"] == "created"


def test_user_lifecycle_and_unique_email(sqlite_db):
    email = f"user-{uuid.uuid4().hex}@example.com"
    user_id = store.create_user(email, "hash", "Test User")
    row = store.get_user_by_email(email)
    assert row is not None
    assert row["id"] == user_id
    assert store.get_user_credits(user_id) == (settings.FREE_CREDITS_MONTHLY, 0)
    assert store.use_user_credits(user_id, 5) == settings.FREE_CREDITS_MONTHLY - 5
    assert store.add_user_credits(user_id, 10) == settings.FREE_CREDITS_MONTHLY + 5
    with pytest.raises(InvalidInputError):
        store.create_user(email, "hash", "Duplicate")


def test_visual_entitlement_is_immutable(sqlite_db):
    device = _device()
    diagram_id = f"diagram-{uuid.uuid4().hex}"
    store.record_diagram_grant(device, diagram_id, '{"topic":"x"}')
    assert store.get_visual_entitlement(diagram_id) == (
        device,
        "",
        "",
        "",
        '{"topic":"x"}',
    )
    assert store.set_visual_result(diagram_id, device, "generative", visual_url="https://a")
    assert not store.set_visual_result(
        diagram_id, device, "generative", visual_url="https://b"
    )
    assert store.get_visual_entitlement(diagram_id)[1] == "https://a"


def test_transaction_rolls_back_on_error(sqlite_db):
    device = _device()
    store.init_device(device)
    with pytest.raises(RuntimeError):
        with sqlite_db.transaction() as tx:
            tx.execute(
                "UPDATE device_credits SET credits_remaining = 1 WHERE device_id = ?",
                (device,),
            )
            raise RuntimeError("boom")
    assert store.get_credits(device)[0] == settings.FREE_CREDITS_MONTHLY


def test_rate_limiter_is_atomic_and_enforced(sqlite_db, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MIN", 2)
    device = _device()
    check_rate_limits(device)
    check_rate_limits(device)
    with pytest.raises(RateLimitError):
        check_rate_limits(device)


def test_sqlite_health(sqlite_db):
    health = sqlite_db.health()
    assert health["backend"] == "sqlite"
    assert health["reachable"] is True
    assert int(health["size_bytes"]) > 0
