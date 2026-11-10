"""Configuration and credential-handling tests for the durable database."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.storage.database import PostgresDatabase
from scripts._db_tools import load_env_file, resolve_database_url, safe_dsn


def test_postgres_url_is_normalized_and_never_logged():
    settings = Settings(
        _env_file=None,
        DEBUG=True,
        DATABASE_URL="postgres://user:super-secret@db.example.com:5432/snapnote?sslmode=require",
    )
    assert settings.DATABASE_URL.startswith("postgresql://")
    assert settings.database_backend == "postgres"
    assert "super-secret" not in safe_dsn(settings.DATABASE_URL)
    assert "db.example.com" in safe_dsn(settings.DATABASE_URL)
    database = PostgresDatabase(settings.DATABASE_URL)
    try:
        assert "super-secret" not in database.safe_dsn
        assert "db=snapnote" in database.safe_dsn
    finally:
        database.close()


def test_empty_database_url_keeps_sqlite():
    settings = Settings(_env_file=None, DEBUG=True, DATABASE_URL="")
    assert settings.DATABASE_URL == ""
    assert settings.database_backend == "sqlite"


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///tmp/credits.db",
        "mysql://user:pass@host/db",
        "postgresql://",
        "not-a-url",
    ],
)
def test_invalid_database_urls_are_rejected(url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DEBUG=True, DATABASE_URL=url)


def test_env_file_parser_and_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nDATABASE_URL=postgresql://file-user:file-pass@file-host/db\n"
        "export OTHER='quoted value'\n",
        encoding="utf-8",
    )
    values = load_env_file(env_file)
    assert values["DATABASE_URL"].endswith("@file-host/db")
    assert values["OTHER"] == "quoted value"

    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert resolve_database_url(env_file).endswith("@file-host/db")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://env-user:env-pass@env-host/envdb"
    )
    assert resolve_database_url(env_file).endswith("@env-host/envdb")
