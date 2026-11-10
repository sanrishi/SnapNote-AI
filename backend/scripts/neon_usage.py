"""Report Neon/PostgreSQL storage and connection usage against the Free plan.

Neon Free includes 0.5 GiB of storage. This script makes the storage number
visible before it becomes an outage, and exposes the same size on /health.
Compute hours are visible in the Neon console; this script cannot query them.

Usage:
    python -m scripts.neon_usage
    python -m scripts.neon_usage --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.database import PostgresDatabase  # noqa: E402
from app.storage.schema import TABLE_NAMES  # noqa: E402
from scripts._db_tools import (  # noqa: E402
    resolve_database_url,
    safe_dsn,
    utc_stamp,
)

NEON_FREE_STORAGE_BYTES = 512 * 1024 * 1024  # 0.5 GiB


def usage_report(url: str, *, max_storage_bytes: int = NEON_FREE_STORAGE_BYTES) -> dict[str, object]:
    database = PostgresDatabase(url, pool_min_size=1, pool_max_size=2)
    try:
        database.initialize()
        db_row = database.execute(
            "SELECT current_database() AS db,"
            " pg_database_size(current_database()) AS size_bytes,"
            " (SELECT count(*) FROM pg_stat_activity "
            "  WHERE datname = current_database()) AS connections,"
            " current_setting('max_connections')::int AS max_connections,"
            " current_setting('server_version') AS server_version"
        ).fetchone()
        if db_row is None:
            raise RuntimeError("database metadata query returned no row")
        tables: dict[str, dict[str, int]] = {}
        for table in TABLE_NAMES:
            count_row = database.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()
            size_row = database.execute(
                "SELECT pg_total_relation_size(?) AS size_bytes", (table,)
            ).fetchone()
            tables[table] = {
                "rows": int(count_row["n"]) if count_row else 0,
                "size_bytes": int(size_row["size_bytes"]) if size_row else 0,
            }
        size_bytes = int(db_row["size_bytes"] or 0)
        percent = (size_bytes / max_storage_bytes * 100.0) if max_storage_bytes else 0.0
        return {
            "checked_at": utc_stamp(),
            "target": safe_dsn(url),
            "database": str(db_row["db"]),
            "server_version": str(db_row["server_version"]),
            "storage_bytes": size_bytes,
            "storage_mb": round(size_bytes / (1024 * 1024), 2),
            "free_limit_bytes": max_storage_bytes,
            "free_limit_mb": round(max_storage_bytes / (1024 * 1024), 2),
            "storage_percent": round(percent, 2),
            "connections": int(db_row["connections"] or 0),
            "max_connections": int(db_row["max_connections"] or 0),
            "tables": tables,
            "compute_hours_note": "Check the Neon console for monthly compute-hour usage.",
            "warning": (
                "storage above 80% of the Free limit"
                if percent >= 80
                else ""
            ),
        }
    finally:
        database.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--max-storage-bytes", type=int, default=NEON_FREE_STORAGE_BYTES)
    parser.add_argument("--json", action="store_true", help="machine-readable output only")
    args = parser.parse_args(argv)
    url = resolve_database_url(Path(args.env_file) if args.env_file else None)
    if not url:
        print("DATABASE_URL is not set (environment or backend/.env).", file=sys.stderr)
        return 1
    try:
        report = usage_report(url, max_storage_bytes=args.max_storage_bytes)
    except Exception as exc:
        print(f"Usage check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"Neon storage: {report['storage_mb']} MB / {report['free_limit_mb']} MB "
            f"({report['storage_percent']}%)"
        )
        print(
            f"Connections: {report['connections']} / {report['max_connections']} "
            f"on {report['target']}"
        )
        for table, stats in report["tables"].items():
            print(f"  {table}: {stats['rows']} rows, {stats['size_bytes']} bytes")
        note = report["compute_hours_note"]
        print(note)
        if report["warning"]:
            print(f"WARNING: {report['warning']}", file=sys.stderr)
    return 1 if report["storage_percent"] >= 90 else 0


if __name__ == "__main__":
    raise SystemExit(main())
