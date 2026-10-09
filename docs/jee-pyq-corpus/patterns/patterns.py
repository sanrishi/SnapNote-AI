"""Observed JEE question patterns (initial evidence-backed slice).

Every pattern references verified question IDs present in the corpus.
These are observed recurrences from reviewed batches, not an exhaustive
archetype taxonomy. Coverage notes stay on each record so the UI never
implies complete pattern discovery.
"""
import json
import os

PATTERNS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "jee-question-patterns.json")


def load_patterns() -> list[dict]:
    """Load pattern records. Returns [] when the file is absent."""
    try:
        with open(PATTERNS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    return data if isinstance(data, list) else []
