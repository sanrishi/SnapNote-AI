# SnapNote AI — Engineering Standards

> This file defines the strict engineering rules for this project.
> Every AI agent (and human) MUST follow these rules when writing code.
> Violations must be flagged and fixed before merge.

---

## 1. Strict Typing & Schemas

### Rule 1.1 — No `Any`, no untyped variables
Python files must have full type hints on all function signatures and return types.
```python
# BAD
def process(data):
    return data

# GOOD
def process(image: bytes) -> ExtractionResult:
    ...
```

### Rule 1.2 — Pydantic for every API boundary
Every request body and response must use a Pydantic `BaseModel`.
Raw dicts are forbidden in route handlers.
```python
# BAD
return {"type": "text", "markdown": md, "tags": [], "creditsUsed": 1}

# GOOD
return ExtractionResponse(type="text", markdown=md, tags=[], creditsUsed=1)
```

### Rule 1.3 — Enum for fixed categories
Use `str, Enum` for type fields, not raw strings.
```python
class ExtractionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    DIAGRAM = "diagram"
```

---

## 2. Separation of Concerns

### Rule 2.1 — Three-layer architecture
```
routes/      ← HTTP layer (parse request, call service, return response)
services/    ← Business logic (OCR, Vision LLM, preprocessing)
utils/       ← Pure helpers (image compression, slugify, etc.)
```

### Rule 2.2 — Routes are thin
Route files must NOT contain business logic.
- BAD: Route calls EasyOCR directly
- GOOD: Route calls `ocr_service.extract_text()`, service handles OCR

### Rule 2.3 — One file, one responsibility
- `preprocessor.py` → image preprocessing only
- `ocr_service.py` → OCR + table detection only
- `vision_service.py` → Vision LLM calls only
- `storage_service.py` → R2 upload only

---

## 3. Bulletproof Error Handling

### Rule 3.1 — Custom exception classes
Define domain-specific exceptions in `app/exceptions.py`.
```python
class SnapNoteError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code

class CreditLimitError(SnapNoteError):
    def __init__(self):
        super().__init__("Credit limit exceeded", status_code=429)
```

### Rule 3.2 — Global exception handler
Register a single handler that catches all `SnapNoteError` instances and returns
proper JSON: `{"error": "message", "code": status_code}`.

### Rule 3.3 — Never `try/except: pass`
Every exception must be logged and either:
- Re-raised as a domain-specific `SnapNoteError`
- Handled with a fallback value (with logging)

### Rule 3.4 — Proper HTTP status codes
| Scenario | Status Code |
|----------|-------------|
| Invalid input | 422 |
| Auth failed | 401 |
| Credits exhausted | 429 |
| Image too large | 413 |
| OCR failed | 502 (upstream error) |
| Unexpected | 500 |

---

## 4. Environment & Secrets Management

### Rule 4.1 — Zero hardcoded secrets
API keys, Firebase creds, R2 tokens → NEVER in code.
Always in `.env` file (gitignored) or environment variables.

### Rule 4.2 — Config via Pydantic Settings
Use `pydantic-settings` (not raw `os.getenv`) for all config.
```python
class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(validation_alias="OPENAI_API_KEY")
    R2_BUCKET_NAME: str = "snapnote-diagrams"
    FREE_CREDITS_MONTHLY: int = 50

    model_config = SettingsConfigDict(env_file=".env")
```

### Rule 4.3 — .env.example is required
Keep a `.env.example` with dummy values committed to repo.
Actual `.env` is gitignored.

---

## 5. Testing Standards

### Rule 5.1 — Test every service in isolation
Tests go in `backend/tests/` mirroring the service structure.
```python
tests/
  services/
    test_preprocessor.py
    test_ocr_service.py
    test_vision_service.py
  routes/
    test_extract.py
    test_auth.py
```

### Rule 5.2 — Fixtures over setup code
Use `pytest.fixture` for shared test data (sample images, mock API responses).

### Rule 5.3 — Mock external APIs
Never call real OpenAI/EasyOCR in unit tests.
Use `unittest.mock` or `pytest-mock`.

---

## 6. Git & Commit Standards

### Rule 6.1 — Conventional commits
```
feat:     new feature
fix:      bug fix
refactor: code change with no feature/fix
chore:    build/config/deps
docs:     documentation
```

### Rule 6.2 — No direct pushes to main
Branch → PR → review → merge.

### Rule 6.3 — Sensitive files in .gitignore
`.env`, `*.key`, `firebase-credentials.json`, `__pycache__/`, `.vercel/`

---

## 7. Code Review Checklist

Before every PR, check:
- [ ] All functions have type hints
- [ ] No `Any` or untyped variables
- [ ] All API responses use Pydantic models
- [ ] No `try/except: pass`
- [ ] No hardcoded secrets
- [ ] Services don't import from routes
- [ ] Routes don't contain business logic
- [ ] HTTP status codes are correct
- [ ] Tests pass

---

*Last updated: July 2026*
