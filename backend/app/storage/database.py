"""Database backends with a small, explicit transaction contract.

The repository layer speaks one dialect: ``?`` placeholders, mapping rows, and
``with db.transaction()``. SQLite and PostgreSQL adapters translate that to
their engine. PostgreSQL is the durable production backend (Neon); SQLite stays
available as the local/rollback backend so a failed migration never strands the
service without storage.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from app.storage.schema import (
    POSTGRES_SCHEMA,
    SCHEMA_VERSION,
    SQLITE_LEGACY_ALTERS,
    SQLITE_SCHEMA,
    SQLITE_TRIGGERS,
)

logger = logging.getLogger(__name__)

Params = Sequence[object]
Row = Mapping[str, object]
_QMARK = re.compile(r"\?")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class QueryResult:
    """Materialized result; small queries keep the adapters simple and safe."""

    _rows: tuple[Row, ...]
    rowcount: int

    def fetchone(self) -> Row | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Row]:
        return list(self._rows)


class DBConnection(ABC):
    """A connection that can execute statements inside a transaction."""

    @abstractmethod
    def execute(self, sql: str, params: Params = ()) -> QueryResult:
        raise NotImplementedError


class Database(ABC):
    """Engine-neutral database interface used by the repository layer."""

    kind: str

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, sql: str, params: Params = ()) -> QueryResult:
        raise NotImplementedError

    @abstractmethod
    def transaction(self) -> "Iterator[DBConnection]":
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_unique_violation(self, exc: BaseException) -> bool:
        raise NotImplementedError

    def raw_sqlite_connection(self) -> sqlite3.Connection:
        raise RuntimeError(f"{self.kind} is not a SQLite database")


class _SQLiteConnection(DBConnection):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Params = ()) -> QueryResult:
        cursor = self._conn.execute(sql, tuple(params))
        rows = tuple(dict(row) for row in cursor.fetchall()) if cursor.description else ()
        return QueryResult(rows, cursor.rowcount)


class SQLiteDatabase(Database):
    kind = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def initialize(self) -> None:
        self._raw_connection()

    def _raw_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if isinstance(conn, sqlite3.Connection):
            return conn
        conn = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
        except sqlite3.Error as exc:
            logger.warning("SQLite PRAGMA failed for %s: %s", self.path.name, exc)
        self._ensure_schema(conn)
        self._local.conn = conn
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            for statement in SQLITE_SCHEMA:
                conn.execute(statement)
            for statement in SQLITE_LEGACY_ALTERS:
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        logger.debug("SQLite legacy ALTER skipped: %s", exc)
            for statement in SQLITE_TRIGGERS:
                conn.execute(statement)
            conn.execute(
                "UPDATE snapnote_schema SET version = ? WHERE id = 1 AND version < ?",
                (SCHEMA_VERSION, SCHEMA_VERSION),
            )
            self._schema_ready = True

    def execute(self, sql: str, params: Params = ()) -> QueryResult:
        return _SQLiteConnection(self._raw_connection()).execute(sql, params)

    @contextmanager
    def transaction(self) -> Iterator[DBConnection]:
        conn = self._raw_connection()
        if getattr(self._local, "in_transaction", False):
            raise RuntimeError("nested SQLite transactions are not supported")
        conn.execute("BEGIN IMMEDIATE")
        self._local.in_transaction = True
        try:
            yield _SQLiteConnection(conn)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error as rollback_exc:
                logger.error("SQLite rollback failed: %s", rollback_exc)
            raise
        finally:
            self._local.in_transaction = False

    def health(self) -> dict[str, object]:
        start = time.perf_counter()
        try:
            self.execute("SELECT 1 AS ok")
            total = 0
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(self.path) + suffix)
                if candidate.exists():
                    total += candidate.stat().st_size
            return {
                "backend": self.kind,
                "reachable": True,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "size_bytes": total,
                "connections": None,
                "max_connections": None,
                "server_version": sqlite3.sqlite_version,
            }
        except Exception as exc:
            logger.error("SQLite health probe failed: %s", exc)
            return {
                "backend": self.kind,
                "reachable": False,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "error": type(exc).__name__,
            }

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if isinstance(conn, sqlite3.Connection):
            conn.close()
            self._local.conn = None

    def raw_sqlite_connection(self) -> sqlite3.Connection:
        return self._raw_connection()

    def is_unique_violation(self, exc: BaseException) -> bool:
        return isinstance(exc, sqlite3.IntegrityError) and (
            "unique" in str(exc).lower() or "users.email" in str(exc)
        )


class _PostgresConnection(DBConnection):
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Params = ()) -> QueryResult:
        translated = _QMARK.sub("%s", sql)
        with self._conn.cursor() as cursor:  # type: ignore[attr-defined]
            if params:
                cursor.execute(translated, tuple(params))
            else:
                cursor.execute(translated)
            if cursor.description is None:
                return QueryResult((), cursor.rowcount)
            rows = tuple(dict(row) for row in cursor.fetchall())
            return QueryResult(rows, cursor.rowcount)


class PostgresDatabase(Database):
    kind = "postgresql"

    def __init__(
        self,
        url: str,
        *,
        pool_min_size: int = 1,
        pool_max_size: int = 5,
        connect_timeout_seconds: float = 10.0,
        statement_timeout_ms: int = 15000,
        application_name: str = "snapnote",
        search_path: str = "",
    ) -> None:
        if search_path and not _SAFE_IDENTIFIER.match(search_path):
            raise ValueError("search_path must be a simple SQL identifier")
        self._url = url
        self._connect_timeout = connect_timeout_seconds
        self._statement_timeout_ms = statement_timeout_ms
        self._search_path = search_path
        self._closed = True
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL storage requires psycopg[binary,pool] "
                "(see backend/requirements.txt)"
            ) from exc
        options = f"-c statement_timeout={int(statement_timeout_ms)}"
        if search_path:
            options += f" -c search_path={search_path}"
        kwargs: dict[str, object] = {
            "row_factory": dict_row,
            "autocommit": False,
            "application_name": application_name,
            "connect_timeout": int(max(1, connect_timeout_seconds)),
            "options": options,
        }
        self._pool = ConnectionPool(
            conninfo=url,
            min_size=max(0, min(pool_min_size, pool_max_size)),
            max_size=max(1, pool_max_size),
            kwargs=kwargs,
            check=ConnectionPool.check_connection,
            open=False,
        )

    def initialize(self) -> None:
        if not self._closed:
            return
        last_error: BaseException | None = None
        for attempt in range(1, 4):
            try:
                self._pool.open()
                self._pool.wait(timeout=self._connect_timeout)
                with self._pool.connection() as conn:
                    with conn.transaction():
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT pg_advisory_xact_lock(hashtext('snapnote_schema_v1'))"
                            )
                            for statement in POSTGRES_SCHEMA:
                                cur.execute(statement)
                            cur.execute(
                                "UPDATE snapnote_schema SET version = %s "
                                "WHERE id = 1 AND version < %s",
                                (SCHEMA_VERSION, SCHEMA_VERSION),
                            )
                self._closed = False
                logger.info(
                    "PostgreSQL storage ready (%s, pool=%d-%d, statement_timeout=%dms)",
                    self.safe_dsn,
                    self._pool.min_size,
                    self._pool.max_size,
                    self._statement_timeout_ms,
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "PostgreSQL connection attempt %d/3 failed: %s",
                    attempt,
                    type(exc).__name__,
                )
                try:
                    self._pool.close()
                except Exception:
                    logger.debug("PostgreSQL pool close during retry failed", exc_info=True)
                if attempt < 3:
                    time.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError(
            f"PostgreSQL storage is unreachable ({type(last_error).__name__})"
        ) from last_error

    @property
    def safe_dsn(self) -> str:
        """Host/database summary for logs — never the password or full URL."""
        match = re.match(
            r"^postgres(?:ql)?://(?:[^@/]+@)?([^/:?]+)(?::\d+)?/([^?]+)",
            self._url,
            flags=re.IGNORECASE,
        )
        if not match:
            return "postgresql://<configured>"
        return f"host={match.group(1)} db={match.group(2)}"

    def execute(self, sql: str, params: Params = ()) -> QueryResult:
        self.initialize()
        with self._pool.connection(timeout=self._connect_timeout) as conn:
            return _PostgresConnection(conn).execute(sql, params)

    @contextmanager
    def transaction(self) -> Iterator[DBConnection]:
        self.initialize()
        with self._pool.connection(timeout=self._connect_timeout) as conn:
            with conn.transaction():
                yield _PostgresConnection(conn)

    def health(self) -> dict[str, object]:
        start = time.perf_counter()
        if self._closed:
            return {
                "backend": self.kind,
                "reachable": False,
                "latency_ms": 0.0,
                "error": "not_initialized",
            }
        try:
            with self._pool.connection(timeout=2.0) as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL statement_timeout = 2000")
                        cur.execute(
                            "SELECT pg_database_size(current_database()) AS size_bytes,"
                            " (SELECT count(*) FROM pg_stat_activity "
                            "  WHERE datname = current_database()) AS connections,"
                            " current_setting('max_connections')::int AS max_connections,"
                            " current_setting('server_version') AS server_version"
                        )
                        row = cur.fetchone() or {}
            return {
                "backend": self.kind,
                "reachable": True,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "size_bytes": int(row.get("size_bytes") or 0),
                "connections": int(row.get("connections") or 0),
                "max_connections": int(row.get("max_connections") or 0),
                "server_version": str(row.get("server_version") or ""),
            }
        except Exception as exc:
            logger.error("PostgreSQL health probe failed: %s", type(exc).__name__)
            return {
                "backend": self.kind,
                "reachable": False,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "error": type(exc).__name__,
            }

    def close(self) -> None:
        if not self._closed:
            self._pool.close()
            self._closed = True

    def is_unique_violation(self, exc: BaseException) -> bool:
        return (
            getattr(exc, "sqlstate", "") == "23505"
            or type(exc).__name__ == "UniqueViolation"
        )


_database: Database | None = None
_database_lock = threading.Lock()


def build_database() -> Database:
    from app.config import settings

    if settings.DATABASE_URL:
        return PostgresDatabase(
            settings.DATABASE_URL,
            pool_min_size=settings.DATABASE_POOL_MIN_SIZE,
            pool_max_size=settings.DATABASE_POOL_MAX_SIZE,
            connect_timeout_seconds=settings.DATABASE_CONNECT_TIMEOUT_SECONDS,
            statement_timeout_ms=settings.DATABASE_STATEMENT_TIMEOUT_MS,
            application_name=settings.DATABASE_APPLICATION_NAME,
        )
    return SQLiteDatabase(settings.CREDITS_DB_PATH)


def get_database() -> Database:
    global _database
    if _database is None:
        with _database_lock:
            if _database is None:
                database = build_database()
                database.initialize()
                _database = database
    return _database


def initialize_database() -> str:
    """Open the configured backend and create/upgrade the schema."""
    database = get_database()
    logger.info("Durable storage backend: %s", database.kind)
    return database.kind


def database_health() -> dict[str, object]:
    """Probe without letting a storage outage crash /health."""
    try:
        return get_database().health()
    except Exception as exc:
        logger.error("Database health unavailable: %s", type(exc).__name__)
        return {
            "backend": "unknown",
            "reachable": False,
            "latency_ms": 0.0,
            "error": type(exc).__name__,
        }


def set_database(database: Database | None) -> None:
    """Replace the process singleton (tests and migration tooling only)."""
    global _database
    with _database_lock:
        previous = _database
        _database = database
    if previous is not None and previous is not database:
        previous.close()


def reset_database() -> None:
    set_database(None)
