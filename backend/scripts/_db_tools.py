"""Shared helpers for the backup, merge, migration, and rollback scripts.

These scripts are deliberately boring: they copy rows, compare fingerprints,
and never mutate the source database. Credentials are read from the environment
or a gitignored env file and are never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from app.storage.database import Database
from app.storage.schema import KEY_COLUMNS, TABLE_COLUMNS, TABLE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKUP_ROOT = REPO_ROOT / "backups"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def open_sqlite(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    if readonly:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def checkpoint_sqlite(path: Path) -> dict[str, object]:
    """Fold the WAL into the main file before copying or hashing it."""
    conn = open_sqlite(path)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        return {
            "wal_checkpoint": list(row) if row is not None else [],
            "integrity_check": str(integrity[0]) if integrity is not None else "unknown",
        }
    finally:
        conn.close()


def copy_sqlite(source: Path, destination: Path) -> None:
    """Use SQLite's backup API for a consistent copy, including WAL pages."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = open_sqlite(source)
    dst = sqlite3.connect(str(destination))
    try:
        src.backup(dst)
        dst.commit()
        # A backup artifact should be one self-contained file: fold the WAL
        # pages into the main database and leave no -wal/-shm sidecars behind.
        dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.commit()
    finally:
        dst.close()
        src.close()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(destination) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def sqlite_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, object]]:
    columns = TABLE_COLUMNS[table]
    sql = f"SELECT {', '.join(columns)} FROM {table}"
    if table == "request_log":
        sql += " ORDER BY id"
    elif table == "users":
        sql += " ORDER BY id"
    else:
        sql += f" ORDER BY {KEY_COLUMNS[table]}"
    return [dict(row) for row in conn.execute(sql)]


def database_rows(database: Database, table: str) -> list[dict[str, object]]:
    """Read all rows through the engine-neutral Database interface."""
    columns = TABLE_COLUMNS[table]
    sql = f"SELECT {', '.join(columns)} FROM {table}"
    if table == "request_log":
        sql += " ORDER BY id"
    elif table == "users":
        sql += " ORDER BY id"
    else:
        sql += f" ORDER BY {KEY_COLUMNS[table]}"
    result = database.execute(sql)
    return [dict(row) for row in result.fetchall()]


def canonical_row(row: Mapping[str, object], columns: Sequence[str]) -> str:
    payload = {column: row.get(column) for column in columns}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def fingerprint(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    lines = sorted(canonical_row(row, columns) for row in rows)
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_table(table: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    columns = TABLE_COLUMNS[table]
    summary: dict[str, object] = {
        "rows": len(rows),
        "sha256": fingerprint(rows, columns),
    }
    if table in ("device_credits", "users"):
        summary["credits_remaining_sum"] = sum(
            int(row.get("credits_remaining") or 0) for row in rows
        )
        summary["credits_used_sum"] = sum(
            int(row.get("credits_used") or 0) for row in rows
        )
    elif table == "razorpay_orders":
        statuses: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status") or "")
            statuses[status] = statuses.get(status, 0) + 1
        summary["status_counts"] = dict(sorted(statuses.items()))
        summary["completed_credits_sum"] = sum(
            int(row.get("credits") or 0)
            for row in rows
            if str(row.get("status") or "") == "completed"
        )
    elif table == "visual_explanations":
        summary["generated_visuals"] = sum(
            1
            for row in rows
            if str(row.get("visual_url") or "") or str(row.get("visual_svg") or "")
        )
    return summary


def summarize_all(rows_by_table: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    return {
        table: summarize_table(table, rows_by_table[table])
        for table in TABLE_NAMES
        if table in rows_by_table
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def resolve_database_url(env_file: Path | None = None) -> str:
    """Environment wins over the gitignored env file; no URL is ever printed."""
    value = os.environ.get("DATABASE_URL", "").strip()
    if value:
        return value
    candidates = [env_file] if env_file else [REPO_ROOT / "backend" / ".env", REPO_ROOT / ".env"]
    for candidate in candidates:
        if candidate is None:
            continue
        values = load_env_file(candidate)
        candidate_url = (values.get("DATABASE_URL") or "").strip()
        if candidate_url:
            return candidate_url
    return ""


def safe_dsn(url: str) -> str:
    parsed = urlsplit(url)
    return f"host={parsed.hostname or '?'} db={(parsed.path or '/?').lstrip('/') or '?'}"


def ensure_empty_or_forced(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path} (use --force)")
    if path.exists() and path.is_dir():
        raise IsADirectoryError(path)


def quote_identifier(value: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return '"' + value.replace('"', '""') + '"'
