"""Export PostgreSQL back to a self-contained SQLite file (rollback artifact).

This is intentionally separate from the migration importer: rollback must be a
fresh, verified snapshot of the live database, not an assumption that the old
SQLite file is still current. The exported SQLite file can be used by setting
``DATABASE_URL=""`` and ``CREDITS_DB_PATH=<exported file>``.

Usage (from the repository root):
    python -m scripts.db_rollback_export
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.database import SQLiteDatabase  # noqa: E402
from app.storage.schema import KEY_COLUMNS, TABLE_COLUMNS, TABLE_NAMES  # noqa: E402
from scripts._db_tools import (  # noqa: E402
    DEFAULT_BACKUP_ROOT,
    resolve_database_url,
    safe_dsn,
    sha256_file,
    summarize_all,
    utc_stamp,
    write_json,
)


def _export(url: str, search_path: str = "") -> dict[str, list[dict[str, object]]]:
    from psycopg import connect
    from psycopg.rows import dict_row

    if search_path and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", search_path):
        raise ValueError("search_path must be a simple SQL identifier")
    kwargs: dict[str, object] = {"row_factory": dict_row}
    if search_path:
        kwargs["options"] = f"-c search_path={search_path}"
    conn = connect(url, **kwargs)
    try:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        exported: dict[str, list[dict[str, object]]] = {}
        for table in TABLE_NAMES:
            columns = TABLE_COLUMNS[table]
            order = "id" if table == "request_log" else KEY_COLUMNS[table]
            result = conn.execute(
                f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order}"
            )
            exported[table] = [dict(row) for row in result.fetchall()]
        conn.execute("COMMIT")
        return exported
    finally:
        conn.close()


def export_to_sqlite(
    url: str,
    out: Path,
    search_path: str = "",
) -> dict[str, object]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    rows_by_table = _export(url, search_path=search_path)
    database = SQLiteDatabase(out)
    conn = database.raw_sqlite_connection()
    try:
        for table in TABLE_NAMES:
            columns = TABLE_COLUMNS[table]
            placeholders = ", ".join("?" for _ in columns)
            for row in rows_by_table[table]:
                conn.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) "
                    f"VALUES ({placeholders})",
                    tuple(row.get(column) for column in columns),
                )
        max_log_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM request_log"
        ).fetchone()[0]
        conn.execute(
            "UPDATE sqlite_sequence SET seq = ? WHERE name = 'request_log'",
            (int(max_log_id),),
        )
        conn.commit()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]) != "ok":
            raise RuntimeError(f"rollback export failed integrity_check: {integrity}")
        written_rows = {table: [] for table in TABLE_NAMES}
        for table in TABLE_NAMES:
            columns = TABLE_COLUMNS[table]
            written_rows[table] = [
                dict(row)
                for row in conn.execute(
                    f"SELECT {', '.join(columns)} FROM {table}"
                ).fetchall()
            ]
    finally:
        database.close()

    source_summary = summarize_all(rows_by_table)
    target_summary = summarize_all(written_rows)
    if source_summary != target_summary:
        raise RuntimeError("rollback export verification failed")
    return {
        "created_at": utc_stamp(),
        "source": safe_dsn(url),
        "target": str(out.resolve()),
        "target_bytes": out.stat().st_size,
        "target_sha256": sha256_file(out),
        "tables": target_summary,
        "integrity_check": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=None, help="gitignored env file with DATABASE_URL")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_BACKUP_ROOT / "rollback" / f"postgres-export-{utc_stamp()}.db"),
        help="new SQLite rollback file",
    )
    parser.add_argument("--search-path", default="", help="optional schema (tests/ops)")
    args = parser.parse_args(argv)
    url = resolve_database_url(Path(args.env_file) if args.env_file else None)
    if not url:
        print("DATABASE_URL is not set (environment or backend/.env).", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = export_to_sqlite(url, out, search_path=args.search_path)
    except Exception as exc:
        print(f"Rollback export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    manifest_path = out.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["tables"], indent=2, sort_keys=True))
    print(f"SQLite rollback file: {out.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
