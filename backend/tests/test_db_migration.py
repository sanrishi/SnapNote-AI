"""Tests for backup, merge, and migration safety tooling.

No PostgreSQL is required here; the PostgreSQL round-trip is covered in
``test_postgres_integration.py`` when TEST_DATABASE_URL is configured.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from app.storage.database import SQLiteDatabase
from scripts._db_tools import sha256_file, summarize_all, sqlite_rows
from scripts.db_backup_sqlite import _label, backup_one
from scripts.db_merge_sqlite import merge


def _create_db(path):
    database = SQLiteDatabase(path)
    database.initialize()
    return database


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_primary(path) -> None:
    database = _create_db(path)
    conn = database.raw_sqlite_connection()
    conn.execute(
        "INSERT INTO device_credits (device_id, credits_remaining, credits_used) "
        "VALUES ('shared', 10, 5), ('root-only', 50, 0)"
    )
    conn.execute(
        "INSERT INTO users (id, email, password_hash, name, picture, "
        "credits_remaining, credits_used, created_at) "
        "VALUES ('u1', 'root@example.com', 'h', 'Root', '', 40, 10, 1)"
    )
    conn.execute(
        "INSERT INTO razorpay_orders (order_id, device_id, plan, credits, amount, "
        "status, razorpay_payment_id, created_at) "
        "VALUES ('o-shared', 'shared', 'starter', 50, 4900, 'completed', 'pay-root', 1)"
    )
    conn.execute(
        "INSERT INTO request_log (device_id, ts) VALUES ('shared', 100)"
    )
    conn.execute(
        "INSERT INTO visual_explanations (diagram_id, device_id, study_notes_json, "
        "visual_url, render_mode, visual_svg, generated_at) "
        "VALUES ('d-shared', 'shared', 'root-notes', 'https://root', 'generative', '', 1)"
    )
    conn.commit()
    database.close()


def _seed_secondary(path) -> None:
    database = _create_db(path)
    conn = database.raw_sqlite_connection()
    conn.execute(
        "INSERT INTO device_credits (device_id, credits_remaining, credits_used) "
        "VALUES ('shared', 99, 1), ('backend-only', 20, 2)"
    )
    conn.execute(
        "INSERT INTO users (id, email, password_hash, name, picture, "
        "credits_remaining, credits_used, created_at) "
        "VALUES ('u1', 'backend@example.com', 'h', 'Backend', '', 30, 0, 2)"
    )
    conn.execute(
        "INSERT INTO users (id, email, password_hash, name, picture, "
        "credits_remaining, credits_used, created_at) "
        "VALUES ('u2', 'backend2@example.com', 'h', 'Backend 2', '', 20, 0, 3)"
    )
    conn.execute(
        "INSERT INTO users (id, email, password_hash, name, picture, "
        "credits_remaining, credits_used, created_at) "
        "VALUES ('u3', 'root@example.com', 'h', 'Duplicate email', '', 10, 0, 4)"
    )
    conn.execute(
        "INSERT INTO razorpay_orders (order_id, device_id, plan, credits, amount, "
        "status, razorpay_payment_id, created_at) "
        "VALUES ('o-shared', 'shared', 'popular', 120, 9900, 'created', NULL, 2)"
    )
    conn.execute(
        "INSERT INTO razorpay_orders (order_id, device_id, plan, credits, amount, "
        "status, razorpay_payment_id, created_at) "
        "VALUES ('o-backend', 'backend-only', 'starter', 50, 4900, 'completed', 'pay-b', 3)"
    )
    conn.execute(
        "INSERT INTO request_log (device_id, ts) VALUES ('shared', 100), "
        "('backend-only', 200), ('backend-only', 200)"
    )
    conn.execute(
        "INSERT INTO visual_explanations (diagram_id, device_id, study_notes_json, "
        "visual_url, render_mode, visual_svg, generated_at) "
        "VALUES ('d-shared', 'shared', 'backend-notes', '', 'deterministic', '<svg/>', 2)"
    )
    conn.execute(
        "INSERT INTO visual_explanations (diagram_id, device_id, study_notes_json, "
        "visual_url, render_mode, visual_svg, generated_at) "
        "VALUES ('d-backend', 'backend-only', '', 'https://backend', 'generative', '', 3)"
    )
    conn.commit()
    database.close()


def test_backup_is_verified_and_does_not_modify_source(tmp_path):
    source = tmp_path / "credits.db"
    _seed_primary(source)
    before = _sha(source)
    info = backup_one(source, tmp_path / "backup", export_json=True)
    assert _sha(source) == before
    assert info["integrity_check"] == "ok"
    assert info["tables"]["device_credits"]["rows"] == 2
    assert info["tables"]["users"]["credits_remaining_sum"] == 40
    label = _label(source)
    backup = tmp_path / "backup" / f"{label}.db"
    assert backup.exists()
    assert not (tmp_path / "backup" / f"{label}.db-wal").exists()
    exported = json.loads((tmp_path / "backup" / f"{label}.json").read_text())
    assert len(exported["users"]) == 1
    devices = {row["device_id"]: row for row in exported["device_credits"]}
    assert devices["shared"]["credits_remaining"] == 10


def test_merge_root_wins_archives_conflicts_and_remaps_visuals(tmp_path):
    primary = tmp_path / "root.db"
    secondary = tmp_path / "backend.db"
    out = tmp_path / "merged.db"
    _seed_primary(primary)
    _seed_secondary(secondary)
    primary_before = _sha(primary)
    secondary_before = _sha(secondary)

    report = merge(primary, secondary, out)

    assert _sha(primary) == primary_before
    assert _sha(secondary) == secondary_before
    conn = sqlite3.connect(out)
    conn.row_factory = sqlite3.Row
    try:
        devices = {
            row["device_id"]: (row["credits_remaining"], row["credits_used"])
            for row in conn.execute("SELECT * FROM device_credits")
        }
        assert devices["shared"] == (10, 5)  # root wins
        assert devices["backend-only"] == (20, 2)
        assert devices["root-only"] == (50, 0)

        users = {row["id"]: row["email"] for row in conn.execute("SELECT * FROM users")}
        assert users == {"u1": "root@example.com", "u2": "backend2@example.com"}

        orders = {
            row["order_id"]: (row["plan"], row["status"])
            for row in conn.execute("SELECT * FROM razorpay_orders")
        }
        assert orders["o-shared"] == ("starter", "completed")
        assert orders["o-backend"] == ("starter", "completed")

        logs = [
            (row["device_id"], row["ts"])
            for row in conn.execute("SELECT device_id, ts FROM request_log ORDER BY id")
        ]
        assert logs == [("shared", 100), ("backend-only", 200)]

        visuals = {
            row["diagram_id"]: row["study_notes_json"]
            for row in conn.execute("SELECT * FROM visual_explanations")
        }
        assert visuals["d-shared"] == "root-notes"
        remapped = [key for key in visuals if key.startswith("d-shared-merged-")]
        assert len(remapped) == 1
        assert visuals[remapped[0]] == "backend-notes"
        assert visuals["d-backend"] == ""

        archive = conn.execute("SELECT table_name, row_key, reason FROM merge_archive").fetchall()
        archived = {(row["table_name"], row["row_key"]) for row in archive}
        assert ("device_credits", "shared") in archived
        assert ("users", "u1") in archived
        assert ("users", "u3") in archived
        assert ("razorpay_orders", "o-shared") in archived
        assert ("visual_explanations", "d-shared") in archived
        assert report["checks"] == {
            "integrity_ok": True,
            "device_remaining_sum": True,
            "user_remaining_sum": True,
            "completed_order_credits_sum": True,
        }
    finally:
        conn.close()


def test_merge_refuses_existing_output(tmp_path):
    primary = tmp_path / "root.db"
    secondary = tmp_path / "backend.db"
    out = tmp_path / "merged.db"
    _seed_primary(primary)
    _seed_secondary(secondary)
    out.write_text("existing")
    try:
        merge(primary, secondary, out)
    except FileExistsError:
        pass
    else:
        raise AssertionError("merge must not overwrite an existing file")


def test_summaries_are_stable_for_rows(tmp_path):
    source = tmp_path / "credits.db"
    _seed_primary(source)
    database = _create_db(source)
    rows = {table: sqlite_rows(database.raw_sqlite_connection(), table) for table in (
        "users",
        "device_credits",
        "razorpay_orders",
        "request_log",
        "visual_explanations",
    )}
    database.close()
    first = summarize_all(rows)
    second = summarize_all({key: list(reversed(value)) for key, value in rows.items()})
    assert first == second
    assert first["device_credits"]["sha256"]
    assert sha256_file(source)
