"""Copy the merged SQLite snapshot into PostgreSQL (Neon) with verification.

Safety contract:
  * default is a DRY RUN: it creates the schema if missing, refuses a non-empty
    target, and prints the plan without inserting rows;
  * ``--apply`` writes in ONE transaction and never deletes/truncates;
  * ``--mode resume`` only inserts missing rows, refuses conflicting rows;
  * source and target row fingerprints must match before success;
  * the SQLite source is opened read-only and is never modified.

Usage (from the repository root):
    python -m scripts.db_migrate_postgres --source backups/.../credits.merged.db
    python -m scripts.db_migrate_postgres --source ... --apply

DATABASE_URL is read from the environment first, then from backend/.env.
Never pass a connection string on the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.database import PostgresDatabase  # noqa: E402
from app.storage.schema import KEY_COLUMNS, TABLE_COLUMNS, TABLE_NAMES  # noqa: E402
from scripts._db_tools import (  # noqa: E402
    DEFAULT_BACKUP_ROOT,
    database_rows,
    open_sqlite,
    resolve_database_url,
    safe_dsn,
    sqlite_rows,
    summarize_all,
    utc_stamp,
    write_json,
)

IMPORT_ORDER: tuple[str, ...] = (
    "users",
    "device_credits",
    "razorpay_orders",
    "request_log",
    "visual_explanations",
)


def _source_rows(source: Path) -> dict[str, list[dict[str, object]]]:
    conn = open_sqlite(source, readonly=True)
    try:
        return {table: sqlite_rows(conn, table) for table in TABLE_NAMES}
    finally:
        conn.close()


def _target_counts(database: PostgresDatabase) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in TABLE_NAMES:
        row = database.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        counts[table] = int(row["n"]) if row else 0
    return counts


def _insert_sql(table: str) -> str:
    columns = TABLE_COLUMNS[table]
    placeholders = ", ".join("?" for _ in columns)
    return (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({KEY_COLUMNS[table]}) DO NOTHING"
    )


def _apply(database: PostgresDatabase, rows_by_table: dict[str, list[dict[str, object]]]) -> dict[str, int]:
    inserted = {table: 0 for table in TABLE_NAMES}
    skipped = {table: 0 for table in TABLE_NAMES}
    with database.transaction() as tx:
        done = 0
        for table in IMPORT_ORDER:
            sql = _insert_sql(table)
            for row in rows_by_table[table]:
                result = tx.execute(
                    sql, tuple(row.get(column) for column in TABLE_COLUMNS[table])
                )
                if result.rowcount > 0:
                    inserted[table] += 1
                else:
                    skipped[table] += 1
                done += 1
                if done % 500 == 0:
                    print(f"  imported {done} rows...")
        tx.execute(
            "SELECT setval("
            " pg_get_serial_sequence('request_log', 'id'),"
            " GREATEST((SELECT COALESCE(MAX(id), 1) FROM request_log), 1),"
            " true)"
        )
    return {table: inserted[table] for table in TABLE_NAMES} | {
        f"skipped_{table}": skipped[table] for table in TABLE_NAMES
    }


def _verify(
    source_rows: dict[str, list[dict[str, object]]],
    database: PostgresDatabase,
) -> tuple[bool, dict[str, object], dict[str, object]]:
    target_rows = {table: database_rows(database, table) for table in TABLE_NAMES}
    source_summary = summarize_all(source_rows)
    target_summary = summarize_all(target_rows)
    return source_summary == target_summary, source_summary, target_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_BACKUP_ROOT / "pre-migration" / "credits.merged.db"),
        help="merged SQLite snapshot to import",
    )
    parser.add_argument("--env-file", default=None, help="gitignored env file with DATABASE_URL")
    parser.add_argument("--apply", action="store_true", help="write to PostgreSQL (default: dry run)")
    parser.add_argument(
        "--mode",
        choices=("fresh", "resume"),
        default="fresh",
        help="fresh refuses a non-empty target; resume only inserts missing rows",
    )
    parser.add_argument("--verify-only", action="store_true", help="compare source and target only")
    parser.add_argument(
        "--backup-dir",
        default=str(DEFAULT_BACKUP_ROOT / "pre-migration"),
        help="where a non-empty target is exported before resume",
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1
    url = resolve_database_url(Path(args.env_file) if args.env_file else None)
    if not url:
        print(
            "DATABASE_URL is not set (environment or backend/.env).",
            file=sys.stderr,
        )
        return 1

    source_rows = _source_rows(source)
    print(f"Source: {source.resolve()} ({safe_dsn(url)})")

    database = PostgresDatabase(url, pool_min_size=1, pool_max_size=3)
    try:
        database.initialize()
        counts = _target_counts(database)
        non_empty = {table: count for table, count in counts.items() if count}
        print(f"Target rows before: {counts}")

        if args.verify_only:
            ok, source_s, target_s = _verify(source_rows, database)
            print(json.dumps({"match": ok, "source": source_s, "target": target_s}, indent=2))
            return 0 if ok else 2

        if non_empty:
            if args.mode == "fresh":
                print(
                    "Refusing fresh import into a non-empty target. "
                    "Use --mode resume (insert-only) or a clean Neon branch.",
                    file=sys.stderr,
                )
                return 3
            backup_path = Path(args.backup_dir) / f"postgres-before-resume-{utc_stamp()}.json"
            write_json(backup_path, {table: database_rows(database, table) for table in TABLE_NAMES})
            print(f"Non-empty target exported to {backup_path.resolve()}")

        plan = {
            table: len(source_rows[table]) for table in TABLE_NAMES
        }
        print(f"Rows to import (mode={args.mode}): {plan}")
        if not args.apply:
            print("Dry run only. Re-run with --apply to write.")
            return 0

        inserted = _apply(database, source_rows)
        ok, source_s, target_s = _verify(source_rows, database)
        report = {
            "created_at": utc_stamp(),
            "source": str(source.resolve()),
            "target": safe_dsn(url),
            "mode": args.mode,
            "rows_before": counts,
            "inserted": inserted,
            "match": ok,
            "source_summary": source_s,
            "target_summary": target_s,
        }
        report_path = Path(args.backup_dir) / f"postgres-import-{utc_stamp()}.json"
        write_json(report_path, report)
        print(json.dumps({"match": ok, "inserted": inserted}, indent=2))
        print(f"Report: {report_path.resolve()}")
        if not ok:
            print("VERIFICATION FAILED: source and target fingerprints differ.", file=sys.stderr)
            return 2
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
