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

---

## 8. OCR Confidence Gate Architecture

### Rule 8.1 — Nine heuristics in `low_quality_result()`
The gate checks in order:
1. Empty result → escalate
2. Total chars < 10 → escalate
3. Average confidence < 0.4 → escalate
4. Low-confidence (< 0.3) ratio > 0.3 → escalate
5. Non-alphanumeric char ratio > 0.5 → escalate
6. Single-char token ratio > 0.3 → escalate
7. English-likeness ratio < 0.4 (vowel + no 5-consonant run, tokens ≥ 3 chars) → escalate
8. Average alpha-token length < 4.5 → escalate
9. Token ending with `})]>` ratio > 0.15 → escalate
10. Symbol-containing token ratio > 0.3 → escalate
11. Mixed digit-letter token ratio > 0.2 → escalate

### Rule 8.2 — Known soft spots
- **CS/algorithms content**: Big-O notation (`O(n)`, `O(log n)`) pushes `symbol_ratio` toward 0.3 threshold but is legitimate, not garbled. Gate may under-trigger on clean-format CS slides with heavy notation.
- **Fluent-looking wrong text**: OCR can produce high-confidence wrong output (e.g., `"ecture"` at conf=1.0). No structural heuristic catches this. Gate relies on symbol/mixed-digit signals catching formula-heavy slides.
- **Dark-themed slides**: OCR performs reasonably well (high confidence), so detection quality isn't degraded — but dark backgrounds can change character detection patterns.

### Rule 8.2b — Learning-first output + Trust layer (Layer 3: Verify Before Studying)
- **Product philosophy**: The output answers "what should the student learn?" NOT "what did the AI see?" The 5-credit Diagram result is a complete learning product — no separate revision upsell.
- **Diagram mode StudyNotes order**: `topic` (title; `is_probable` → "Topic inferred from screenshot") → `what_you_should_remember` → `key_formulas` (each with `confidence`: clear | context_needed | possible_extraction_issue) → `understand_it` → `common_mistakes` → `thirty_second_revision` (3-5 bullets) → `visual_context` (1-2 sentences max, no object/axis inventory) → `verify_before_studying` → `uncertainties` → `analogy`.
- **Grounding rule**: NEVER invent missing derivations/equations/definitions. Distinguish visible evidence / safe inference / missing context. If a derivation is cut off, say so — never complete it as the professor's work.
- **Common Mistakes**: never fabricated. Only when supported by visible material or framed as "a general thing to watch for with this type of problem."
- **Proportional uncertainty**: no scary warnings for every formula. Confidence states: clear (no warning) / context_needed (subtle note) / possible_extraction_issue (listed in verify_before_studying).
- `StudyNotes.verify_before_studying: list[str]` holds equations/symbols Gemini could not read with high confidence (confidence = possible_extraction_issue), e.g. `τ = Iα may have been misread as 't = Iα'`.
- Common confusion pairs the prompt watches: τ/t, ω/w, θ/0, μ/u, α/a, v/r.
- Rendered as a red `🛡️ Verify Before Studying` card (frontend + markdown) after Visual Context.
- Trust ordering for QA: formula accuracy > topic accuracy > explanation quality.
- **Text mode** stays the cheap funnel: structured OCR (formulas/labels/text) at 1 credit; `✨ Make This Revision-Ready` (+1) is a separate Gemini call producing the same learning sheet. Revision button is TEXT-ONLY; diagram mode never shows it.
- `RevisionResponse` returns the same `StudyNotes` shape (field `study_notes`) — the revision layer is the funnel on-ramp, not the diagram's upsell.

### Rule 8.3 — Pricing model
- `/extract/text`: 1 credit if OCR succeeds standalone; **5 credits if escalation to Gemini fires** (credit check before Gemini call, line 55-57 of extract.py)
- `/extract/revision`: 1 credit (separate Gemini call for the learning layer)
- `/extract/diagram`: 5 credits always
- 50 free credits/month = **50 OCR-only requests**, **50 revision enhancements**, or **10 Gemini-escalated requests**
- When Gemini returns 429 (rate limit), text endpoint falls back to OCR + a note, charged 1 credit. Diagram endpoint returns a clean "high demand" message.

### Rule 8.4 — Test fixtures
9 fixtures in `stress_test_fixtures/`:
1. Stats probability table (dense formulas) → escalate
2. PID controller flowchart → escalate
3. Engineering math formulas → escalate
4. Hindi Devanagari coaching slide → escalate
5. YouTube coaching slide (Physics Wallah style) → escalate
6. Plain economics prose → stay OCR
7. Biology cell theory prose → stay OCR (untuned, passed cold)
8. Chemistry equilibrium formulas → escalate (untuned, passed cold)
9. Dark-theme CS BST slide (Unacademy style) → escalate (untuned, passed cold)

*Last updated: August 2026*
