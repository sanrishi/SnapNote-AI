"""Merge the two local SQLite databases into a new, verified database.

Policy (chosen by the operator):
  * the root ``credits.db`` is authoritative for every overlapping key;
  * backend-only rows are added;
  * conflicting backend snapshots are preserved in ``merge_archive``;
  * request_log rows are de-duplicated by (device_id, ts) and re-keyed;
  * colliding visual diagram ids are remapped so the backend result is kept;
  * the source files are opened read-only and never modified.

The output is a NEW file. Existing source databases stay untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.database import SQLiteDatabase  # noqa: E402
from app.storage.schema import KEY_COLUMNS, TABLE_COLUMNS, TABLE_NAMES  # noqa: E402
from scripts._db_tools import (  # noqa: E402
    DEFAULT_BACKUP_ROOT,
    open_sqlite,
    sqlite_rows,
    summarize_all,
    utc_stamp,
    write_json,
)


def _insert_row(conn: sqlite3.Connection, table: str, row: dict[str, object]) -> None:
    columns = TABLE_COLUMNS[table]
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(row.get(column) for column in columns),
    )


def _select_key(conn: sqlite3.Connection, table: str, key: object) -> dict[str, object] | None:
    key_column = KEY_COLUMNS[table]
    row = conn.execute(
        f"SELECT {', '.join(TABLE_COLUMNS[table])} FROM {table} "
        f"WHERE {key_column} = ? LIMIT 1",
        (key,),
    ).fetchone()
    return None if row is None else dict(row)


def _archive(
    conn: sqlite3.Connection,
    source: str,
    table: str,
    key: object,
    reason: str,
    row: dict[str, object],
) -> None:
    conn.execute(
        "INSERT INTO merge_archive "
        "(source_db, table_name, row_key, reason, row_json, archived_at) "
        "VALUES (?, ?, ?, ?, ?, strftime('%s','now'))",
        (source, table, str(key), reason, json.dumps(row, sort_keys=True, default=str)),
    )


def _audit(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO merge_audit (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value, sort_keys=True, default=str)),
    )


def _new_diagram_id(existing: set[str], diagram_id: str, row: dict[str, object]) -> str:
    digest = hashlib.sha256(
        json.dumps(row, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    for length in (8, 12, 16, 32):
        candidate = f"{diagram_id}-merged-{digest[:length]}"
        if candidate not in existing:
            return candidate
    raise RuntimeError(f"could not remap diagram id {diagram_id}")


def merge(primary: Path, secondary: Path, out: Path) -> dict[str, object]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out} (choose a new path)")

    primary_conn = open_sqlite(primary, readonly=True)
    secondary_conn = open_sqlite(secondary, readonly=True)
    try:
        primary_rows = {table: sqlite_rows(primary_conn, table) for table in TABLE_NAMES}
        secondary_rows = {
            table: sqlite_rows(secondary_conn, table) for table in TABLE_NAMES
        }
    finally:
        primary_conn.close()
        secondary_conn.close()

    database = SQLiteDatabase(out)
    conn = database.raw_sqlite_connection()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS merge_archive ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " source_db TEXT NOT NULL,"
            " table_name TEXT NOT NULL,"
            " row_key TEXT NOT NULL,"
            " reason TEXT NOT NULL,"
            " row_json TEXT NOT NULL,"
            " archived_at INTEGER NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS merge_audit ("
            " key TEXT PRIMARY KEY,"
            " value TEXT NOT NULL"
            ")"
        )

        for table in TABLE_NAMES:
            for row in primary_rows[table]:
                _insert_row(conn, table, row)

        stats: dict[str, dict[str, int]] = {
            table: {"inserted": 0, "skipped": 0, "conflicts": 0, "remapped": 0}
            for table in TABLE_NAMES
        }
        expected_device_remaining = sum(
            int(row.get("credits_remaining") or 0)
            for row in primary_rows["device_credits"]
        )
        expected_user_remaining = sum(
            int(row.get("credits_remaining") or 0) for row in primary_rows["users"]
        )
        expected_completed_credits = sum(
            int(row.get("credits") or 0)
            for row in primary_rows["razorpay_orders"]
            if str(row.get("status") or "") == "completed"
        )

        for row in secondary_rows["users"]:
            user_id = row["id"]
            existing = _select_key(conn, "users", user_id)
            if existing is not None:
                if existing != row:
                    _archive(conn, str(secondary), "users", user_id, "id_conflict_root_wins", row)
                    stats["users"]["conflicts"] += 1
                else:
                    stats["users"]["skipped"] += 1
                continue
            email = row["email"]
            email_owner = conn.execute(
                "SELECT id FROM users WHERE email = ? LIMIT 1", (email,)
            ).fetchone()
            if email_owner is not None:
                _archive(
                    conn,
                    str(secondary),
                    "users",
                    user_id,
                    f"email_conflict_with_{email_owner['id']}",
                    row,
                )
                stats["users"]["conflicts"] += 1
                continue
            _insert_row(conn, "users", row)
            stats["users"]["inserted"] += 1
            expected_user_remaining += int(row.get("credits_remaining") or 0)

        for row in secondary_rows["device_credits"]:
            device_id = row["device_id"]
            existing = _select_key(conn, "device_credits", device_id)
            if existing is not None:
                if existing != row:
                    _archive(
                        conn,
                        str(secondary),
                        "device_credits",
                        device_id,
                        "device_conflict_root_wins",
                        row,
                    )
                    stats["device_credits"]["conflicts"] += 1
                else:
                    stats["device_credits"]["skipped"] += 1
                continue
            _insert_row(conn, "device_credits", row)
            stats["device_credits"]["inserted"] += 1
            expected_device_remaining += int(row.get("credits_remaining") or 0)

        for row in secondary_rows["razorpay_orders"]:
            order_id = row["order_id"]
            existing = _select_key(conn, "razorpay_orders", order_id)
            if existing is not None:
                if existing != row:
                    _archive(
                        conn,
                        str(secondary),
                        "razorpay_orders",
                        order_id,
                        "external_order_id_conflict_root_wins",
                        row,
                    )
                    stats["razorpay_orders"]["conflicts"] += 1
                else:
                    stats["razorpay_orders"]["skipped"] += 1
                continue
            _insert_row(conn, "razorpay_orders", row)
            stats["razorpay_orders"]["inserted"] += 1
            if str(row.get("status") or "") == "completed":
                expected_completed_credits += int(row.get("credits") or 0)

        secondary_logs = sorted(
            secondary_rows["request_log"], key=lambda item: int(item.get("id") or 0)
        )
        for row in secondary_logs:
            duplicate = conn.execute(
                "SELECT id FROM request_log WHERE device_id = ? AND ts = ? LIMIT 1",
                (row["device_id"], row["ts"]),
            ).fetchone()
            if duplicate is not None:
                stats["request_log"]["skipped"] += 1
                continue
            conn.execute(
                "INSERT INTO request_log (device_id, ts) VALUES (?, ?)",
                (row["device_id"], row["ts"]),
            )
            stats["request_log"]["inserted"] += 1

        existing_diagram_ids = {
            str(row["diagram_id"]) for row in sqlite_rows(conn, "visual_explanations")
        }
        for row in secondary_rows["visual_explanations"]:
            diagram_id = str(row["diagram_id"])
            existing = _select_key(conn, "visual_explanations", diagram_id)
            if existing is None:
                _insert_row(conn, "visual_explanations", row)
                existing_diagram_ids.add(diagram_id)
                stats["visual_explanations"]["inserted"] += 1
                continue
            if existing == row:
                stats["visual_explanations"]["skipped"] += 1
                continue
            new_id = _new_diagram_id(existing_diagram_ids, diagram_id, row)
            remapped = dict(row)
            remapped["diagram_id"] = new_id
            _insert_row(conn, "visual_explanations", remapped)
            existing_diagram_ids.add(new_id)
            _archive(
                conn,
                str(secondary),
                "visual_explanations",
                diagram_id,
                f"diagram_id_conflict_remapped_to_{new_id}",
                row,
            )
            stats["visual_explanations"]["conflicts"] += 1
            stats["visual_explanations"]["remapped"] += 1

        max_log_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM request_log"
        ).fetchone()[0]
        conn.execute(
            "UPDATE sqlite_sequence SET seq = ? WHERE name = 'request_log'",
            (int(max_log_id),),
        )
        if conn.execute(
            "SELECT 1 FROM sqlite_sequence WHERE name = 'request_log'"
        ).fetchone() is None:
            conn.execute(
                "INSERT INTO sqlite_sequence (name, seq) VALUES ('request_log', ?)",
                (int(max_log_id),),
            )

        merged_rows = {table: sqlite_rows(conn, table) for table in TABLE_NAMES}
        merged_summary = summarize_all(merged_rows)

        for table in TABLE_NAMES:
            key_column = KEY_COLUMNS[table]
            duplicate = conn.execute(
                f"SELECT {key_column}, COUNT(*) AS n FROM {table} "
                f"GROUP BY {key_column} HAVING COUNT(*) > 1 LIMIT 1"
            ).fetchone()
            if duplicate is not None:
                raise RuntimeError(f"duplicate key in merged {table}: {duplicate[0]}")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]) != "ok":
            raise RuntimeError(f"merged database failed integrity_check: {integrity}")
        conn.commit()

        report: dict[str, object] = {
            "created_at": utc_stamp(),
            "primary": str(primary.resolve()),
            "secondary": str(secondary.resolve()),
            "merged": str(out.resolve()),
            "policy": "root_wins_backend_only_added_conflicts_archived",
            "before": {
                "primary": summarize_all(primary_rows),
                "secondary": summarize_all(secondary_rows),
            },
            "after": merged_summary,
            "actions": stats,
            "checks": {
                "integrity_ok": True,
                "device_remaining_sum": int(
                    merged_summary["device_credits"]["credits_remaining_sum"]
                )
                == expected_device_remaining,
                "user_remaining_sum": int(
                    merged_summary["users"]["credits_remaining_sum"]
                )
                == expected_user_remaining,
                "completed_order_credits_sum": int(
                    merged_summary["razorpay_orders"]["completed_credits_sum"]
                )
                == expected_completed_credits,
            },
            "archive_rows": int(
                conn.execute("SELECT COUNT(*) FROM merge_archive").fetchone()[0]
            ),
        }
        if not all(report["checks"].values()):
            raise RuntimeError(f"merge reconciliation failed: {report['checks']}")
        _audit(conn, "merge_report", report)
        conn.commit()
    finally:
        database.close()

    report_path = out.with_name(out.stem + ".report.json")
    write_json(report_path, report)
    print(json.dumps(report["after"], indent=2, sort_keys=True))
    print(f"Merged database: {out.resolve()}")
    print(f"Reconciliation report: {report_path.resolve()}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        default=str(DEFAULT_BACKUP_ROOT.parent / "credits.db"),
        help="authoritative SQLite database",
    )
    parser.add_argument(
        "--secondary",
        default=str(DEFAULT_BACKUP_ROOT.parent / "backend" / "credits.db"),
        help="database whose non-conflicting rows are added",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_BACKUP_ROOT / "pre-migration" / "credits.merged.db"),
        help="new merged database path",
    )
    args = parser.parse_args(argv)
    try:
        merge(Path(args.primary), Path(args.secondary), Path(args.out))
    except Exception as exc:
        print(f"Merge failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
