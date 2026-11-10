"""Create verified, timestamped backups of the legacy SQLite databases.

Usage (from the repository root):
    python -m scripts.db_backup_sqlite
    python -m scripts.db_backup_sqlite --source credits.db --source backend/credits.db

The source files are never modified beyond a WAL checkpoint. The command
refuses to overwrite an existing backup directory unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.schema import TABLE_NAMES  # noqa: E402
from scripts._db_tools import (  # noqa: E402
    DEFAULT_BACKUP_ROOT,
    checkpoint_sqlite,
    copy_sqlite,
    ensure_empty_or_forced,
    open_sqlite,
    sha256_file,
    sqlite_rows,
    summarize_all,
    utc_stamp,
    write_json,
)


def _label(source: Path) -> str:
    if source.resolve() == (DEFAULT_BACKUP_ROOT.parent / "credits.db").resolve():
        return "root-credits"
    return f"{source.parent.name}-{source.stem}"


def backup_one(source: Path, out_dir: Path, *, export_json: bool) -> dict[str, object]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    label = _label(source)
    backup_path = out_dir / f"{label}.db"
    if backup_path.exists():
        backup_path.unlink()
    info = checkpoint_sqlite(source)
    copy_sqlite(source, backup_path)

    src_conn = open_sqlite(source, readonly=True)
    dst_conn = open_sqlite(backup_path, readonly=True)
    try:
        source_rows = {table: sqlite_rows(src_conn, table) for table in TABLE_NAMES}
        backup_rows = {table: sqlite_rows(dst_conn, table) for table in TABLE_NAMES}
    finally:
        dst_conn.close()
        src_conn.close()

    source_summary = summarize_all(source_rows)
    backup_summary = summarize_all(backup_rows)
    if source_summary != backup_summary:
        raise RuntimeError(f"backup verification failed for {source}")

    if export_json:
        write_json(out_dir / f"{label}.json", source_rows)

    return {
        "label": label,
        "source": str(source),
        "backup": str(backup_path),
        "source_bytes": source.stat().st_size,
        "backup_bytes": backup_path.stat().st_size,
        "source_sha256": sha256_file(source),
        "backup_sha256": sha256_file(backup_path),
        "integrity_check": info["integrity_check"],
        "wal_checkpoint": info["wal_checkpoint"],
        "tables": source_summary,
        "json_export": str(out_dir / f"{label}.json") if export_json else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="SQLite file to back up (repeatable); defaults to root and backend credits.db",
    )
    parser.add_argument("--out", default=None, help="backup directory")
    parser.add_argument("--no-json", action="store_true", help="skip portable JSON export")
    parser.add_argument("--force", action="store_true", help="allow writing into an existing path")
    args = parser.parse_args(argv)

    if args.source:
        sources = [Path(item) for item in args.source]
    else:
        sources = [
            DEFAULT_BACKUP_ROOT.parent / "credits.db",
            DEFAULT_BACKUP_ROOT.parent / "backend" / "credits.db",
        ]
    sources = [source for source in sources if source.exists()]
    if not sources:
        print("No SQLite databases found; nothing to back up.")
        return 1

    out_dir = Path(args.out) if args.out else DEFAULT_BACKUP_ROOT / f"backup-{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        ensure_empty_or_forced(out_dir, args.force)

    backups = [
        backup_one(source, out_dir, export_json=not args.no_json) for source in sources
    ]
    manifest = {
        "created_at": utc_stamp(),
        "backup_dir": str(out_dir.resolve()),
        "sources": backups,
    }
    write_json(out_dir / "backup_manifest.json", manifest)
    for item in backups:
        tables = item["tables"]
        devices = tables["device_credits"]  # type: ignore[index]
        users = tables["users"]  # type: ignore[index]
        orders = tables["razorpay_orders"]  # type: ignore[index]
        print(
            f"{item['label']}: {devices['rows']} devices, {users['rows']} users, "
            f"{orders['rows']} orders, remaining={devices['credits_remaining_sum']}, "
            f"used={devices['credits_used_sum']}"
        )
    print(f"Backup written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
