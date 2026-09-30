"""Regression guard: credentials must never be tracked by git.

firebase-credentials.json, .env files, private keys, and service-account
JSON blobs must stay out of the repository (untracked + gitignored).
"""
import subprocess

import pytest


def _tracked_files() -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.splitlines()


def test_no_credential_files_tracked():
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("git unavailable; cannot verify tracked files")
    offenders = [
        f
        for f in tracked
        if f.endswith("firebase-credentials.json")
        or f.endswith(".env")
        or f.endswith(".key")
        or f.endswith(".pem")
        or "adminsdk" in f
    ]
    assert offenders == [], f"credential files must never be tracked: {offenders}"


def test_gitignore_covers_firebase_credentials(tmp_path=None):
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ignore_path = os.path.join(root, ".gitignore")
    if not os.path.exists(ignore_path):
        pytest.skip("no .gitignore at repo root")
    with open(ignore_path, encoding="utf-8") as fh:
        content = fh.read()
    assert "firebase-credentials.json" in content
