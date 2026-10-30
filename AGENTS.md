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
The gate is engine-neutral: it consumes line-level `(box, text, confidence)` tokens
from the OCR adapter, regardless of which engine produced them. The gate checks in order:
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
11. Token ratio > 0.2 where the token contains at least one MIXED digit-letter **word** (e.g. `x2y`, `4ac)`) → escalate.
    Word-level, not line-level, on purpose: garbled formulas glue digits into
    words, while legitimate prose merely contains both across a line
    (`1. All living organisms…`, `Robert Hooke (1665)`) — the line-level test
    misfired on numbered lists and years in clean notes.

### Rule 8.2 — Known soft spots
- **CS/algorithms content**: Big-O notation (`O(n)`, `O(log n)`) pushes `symbol_ratio` toward 0.3 threshold but is legitimate, not garbled. Gate may under-trigger on clean-format CS slides with heavy notation.
- **Fluent-looking wrong text**: OCR can produce high-confidence wrong output (e.g., `"ecture"` at conf=1.0). No structural heuristic catches this. Gate relies on symbol/mixed-digit signals catching formula-heavy slides.
- **Dark-themed slides**: OCR performs reasonably well (high confidence), so detection quality isn't degraded — but dark backgrounds can change character detection patterns.

### Rule 8.2b — Learning-first output + Trust layer (Layer 3: Verify Before Studying)
- **Product philosophy**: The output answers "what should the student learn?" NOT "what did the AI see?" The 5-credit Diagram result is a complete learning product — no separate revision upsell.
- **CORE RULE — Input image = evidence, never output material**: The uploaded screenshot is SOURCE/evidence, not the product. The diagram result must NEVER render the raw uploaded image inside the study cards or the markdown (no `![Diagram](url)` embed, no `<img>` of the source). The student DOES get a clean diagram back: Gemini rebuilds what was on the board as a sanitized vector SVG (`StudyNotes.diagram`), so they see a readable version instead of the messy photo. The only permitted display of the RAW source image is the small `📷 View original screenshot` lightbox action at the very bottom of the result (shown only when `imageUrl` exists). Chain: messy screenshot → clean diagram + understanding → exam-ready material.
- **Diagram mode StudyNotes order**: `topic` (title; `is_probable` → "Topic inferred from screenshot") → `what_you_should_remember` → `key_formulas` (each with `confidence`: clear | context_needed | possible_extraction_issue) → `understand_it` → `common_mistakes` → `thirty_second_revision` (3-5 bullets) → `visual_context` (1-2 sentences max, TEXT ONLY, teaches what the diagram MEANS conceptually — never an object/axis/label inventory) → `diagram` (clean SVG rebuild, rendered as 📐 Diagram card + markdown data-URI embed) → `verify_before_studying` → `uncertainties` → `analogy`.
- **Clean SVG diagram (`StudyNotes.diagram`)**: Gemini reconstructs the visible diagram as a flat, readable vector SVG (boxes/arrows/labels, `present=false` + empty svg when the screenshot has no real diagram). It is strictly grounded — same structure, no invented content, no math alteration. SVG is sanitized server-side via `app/utils/svg_safe.py` (allowlist tags/attrs, strips `on*`/`javascript:`/scripts) BEFORE the API response, and re-sanitized when embedding in markdown. Both the frontend card (`notes.diagram.svg` injected into `.diagram-wrap`) and markdown embed (`![Clean diagram](data:...)`) show the CLEAN svg, never the raw photo.
- **visual_context must teach, not describe**: BAD: "Block diagrams illustrate closed-loop control systems with reference inputs, summing junctions, controllers, processes." GOOD: "The diagram represents a closed-loop control system: the reference input is compared with feedback to form an error signal, which the controller uses to drive the plant toward the desired output."
- **Grounding rule**: NEVER invent missing derivations/equations/definitions. Distinguish visible evidence / safe inference / missing context. If a derivation is cut off, say so — never complete it as the professor's work.
- **Common Mistakes**: never fabricated. Only when supported by visible material or framed as "a general thing to watch for with this type of problem."
- **Proportional uncertainty**: no scary warnings for every formula. Confidence states: clear (no warning) / context_needed (subtle note) / possible_extraction_issue (listed in verify_before_studying).
- `StudyNotes.verify_before_studying: list[str]` holds equations/symbols Gemini could not read with high confidence (confidence = possible_extraction_issue), e.g. `τ = Iα may have been misread as 't = Iα'`.
- Common confusion pairs the prompt watches: τ/t, ω/w, θ/0, μ/u, α/a, v/r.
- Rendered as a red `🛡️ Verify Before Studying` card (frontend + markdown) after Visual Context.
- Trust ordering for QA: formula accuracy > topic accuracy > explanation quality.
- **Text mode** stays the cheap funnel: structured OCR (formulas/labels/text) at 1 credit; `✨ Make This Revision-Ready` (+1) is a separate Gemini call producing the same learning sheet. Revision button is TEXT-ONLY; diagram mode never shows it.
- `RevisionResponse` returns the same `StudyNotes` shape (field `study_notes`) — the revision layer is the funnel on-ramp, not the diagram's upsell.

### Rule 8.2c — Worked mathematics is verified or refused (number_line)

Worked interval mathematics (log inequalities with polynomial base/argument, polynomial/rational inequalities) renders ONLY through the deterministic `number_line` scene kind, and ONLY when an exact verifier re-solves the problem and reproduces every claimed set. Regressions that motivated this: `log_{x+3}(x²−x) < 1` was rendered with a union that dropped the domain intersection (`(-3,-2)∪(-1,3)` instead of `(-3,-2)∪(-1,0)∪(1,3)`).

- **`app/utils/math_verify.py`** is the gate (pure, no LLM/network): `solve_problem()` re-solves with sympy (`sympy==1.14.0` is a production dependency, not dev-only) — full log domain (arg > 0, base > 0, base ≠ 1 as excluded points), monotonicity regime split (base > 1 vs 0 < base < 1 with inequality reversal), case-by-case domain intersection; rational critical points only (degree ≤ 3, integer comparison constants).
- **Safe expression parsing (security)**: expression text is source/model-derived and MUST NOT reach Python `eval` or `sympy.parse_expr`/`sympify` (sympy's `parse_expr` evaluates source strings and is documented unsafe for untrusted input). `_parse_expr()` instead tokenizes a strict mathematical grammar, rewrites implicit multiplication (`2x`, `x(x+1)`, `(x+1)(x-2)`) into explicit syntax, parses with `ast.parse(mode="eval")` (builds a tree, never executes), and constructs sympy expressions only from an allowlist of AST nodes: numbers (int literals → Rational; decimal literals rebuilt from their exact `ast` source segment as `Fraction` — NEVER through a Python float, so a literal that differs beyond double precision is never silently collapsed), variable `x`, unary signs, `+ - * /`, and integer powers bounded to ±10. Function calls, attribute access, imports, strings, and all other syntax are rejected; size (120 chars), node count, and AST depth are bounded. Never add a regex/prompt-only "sanitizer" as a substitute for the AST allowlist.
- **Expression-domain preservation**: division and negative powers track their axis restrictions exactly (`1/(x-1) - 1/(x-1) <= 0` and `x·x⁻¹ > 0` keep `x ≠ 1` / `x ≠ 0` even where sympy would cancel the factor); a divisor/restriction that is not a polynomial with rational zeros (e.g. `1/(1/x)`, `(x+1)/(x²-2)`) is REFUSED explicitly ("cannot be verified exactly") rather than partially verified; literal division by zero (`1/0`, `x/(x-x)`, `0^(-1)`) is refused. Degenerate zero-polynomial relations are handled exactly (`x - x <= 0` → ℝ; `x - x < 0` → ∅). Constant log bases must be positive and ≠ 1 (including zero-polynomial bases); a log whose domain is empty is refused; variable-base comparison constants are bounded (|c| ≤ 10).
- **Completeness is enforced, not assumed**: `verify_number_line()` compares claimed vs computed: final solution set; domain track (REQUIRED whenever the computed domain is not the whole real line — any log or denominator restriction; optional only for pure polynomials), case list (for a variable-base log the count must match EXACTLY — an empty case list is a failure, never a pass), per-case condition (must state the EXACT regime — `<base> > 1` or `0 < <base> < 1` including the positivity bound; a bare `x + 3 < 1` that omits `0 <` is rejected), per-case solution set, and each transformed step (REQUIRED for every branch; must be the same inequality derived from monotonicity — checked algebraically against the derived transformation: any additive rearrangement or nonzero rational multiple passes, a negative multiple must flip the direction, and a rewrite that merely happens to solve to the correct branch set is refused). Every excluded value must be visibly represented (marked point or open domain endpoint); markers for values that are NOT excluded are refused as mislabels. Constant-base logs must OMIT cases. Non-strict boundaries are exact: `x - x <= 0` solves to `(-inf, inf)`, not `∅`.
- **`app/utils/math_intervals.py`** is the exact interval-set algebra: Fraction endpoints, `None` = ±infinity, canonical text `(-3, -2) ∪ (-1, 0) ∪ (1, 3)`, union/intersect/without_points/equals. No floats, no tolerance — set equality is exact. Malformed unions (lead/trail/double separators) are rejected, never silently filtered.
- **Refusal semantics**: unsupported math (irrational critical points, non-polynomial operands, non-integer constants, unparseable claims, division by zero, invalid log base, empty log domain, nested division) returns a precise reason string and the visual is REFUSED — never "assumed correct", never displayed unverified. Every refusal reason is human-readable and doubles as the single bounded repair prompt input. Repair fails → 502 "Visual explanation unavailable for this material." with study notes intact.
- **Render path**: `validate_lesson_spec` → `verify_number_line` wired into the same gate as all families. `number_line` is excluded from the v3 Typst composition path (`should_use_v3` False; `extract_lesson_content` raises `LessonValidationError` for `worked_math`) and renders content-fitted (`_render_number_line_scene`: problem card → domain track → case rows → `FINAL ANSWER — MATHEMATICALLY VERIFIED` track → caption; open = hollow dot, closed = filled dot, excluded points = red hollow + `x ≠ N`). Endpoint labels are exact Fractions (`3/2`, never `1.5`) and close-together labels are staggered onto extra rows — two distinct values are never silently merged into one label. Empty solution sets render explicitly ("No real solutions — the inequality is never true"; empty domain: "The domain is empty…"). No fixed 860px stage. Defense in depth: the renderer re-runs `verify_number_line` itself and refuses to draw (returns "") any scene that fails — the "MATHEMATICALLY VERIFIED" header can never appear on an unverified scene, regardless of caller mistakes.
- **Hosted CI**: `.github/workflows/backend-tests.yml` (Python 3.11 = production base, Typst 0.15.1 pinned like the Dockerfile) runs the full backend suite, the JEE corpus tests, and the corpus coverage audit on every PR to master — no secrets required.

### Rule 8.2d — OCR engine (RapidOCR) and the fail-safe contract
- Production OCR engine = **RapidOCR v3** (`rapidocr==3.10.0` + `onnxruntime==1.31.0`, PP-OCRv6 ONNX models bundled in the wheel), picked by measurement, not preference: EasyOCR/torch import alone is ~322 MB, init ~337 MB, and a dense-formula inference peaked ~958 MB — unfit for the 512 MB production instance. RapidOCR peaked ~300 MB total (lazy model load, ONNX arena disabled, 1 inference thread), init ~1.4 s, ~2–8 s/image at full CPU.
- **Adapter contract**: `read_raw()` returns the engine-neutral internal `OCRResult` = `(4-point box, text, confidence 0–1)` tokens; `raw_to_lines`, `is_table_layout`, and the gate consume only this shape. Swapping engines must not touch those consumers.
- **Fail-safe**: engine init wraps ALL exceptions (not just ImportError), caches the failure (`_engine_error` — one attempt per process, no per-request retry storms) and logs it with engine + reason. Inference errors are caught, logged, and degrade to `[]` → the gate escalates to Gemini — never a 500, never fabricated text. A warm-up call runs at init so a broken engine is observable in structured logs immediately. OS-level OOM kills cannot be caught in Python — that is why the engine is memory-bounded and measured against the 512 MB instance rather than trusted because it imports.
- **Concurrency memory safety (measured, not assumed)**: per-request transients scale with pixels (local measurement: ~+55 MB at 2.2 MP → ~+125 MB at 12 MP for `preprocess`/`enhance_for_vision`), and 3-way concurrency OOM-killed the 512 MB container (502 + restart via Render proxy) in the 2026-10 readiness probe. Three bounds close it: `preprocess()` downscales the long edge to `OCR_MAX_LONG_EDGE` (2048 — slide fixtures and 1080p screenshots are untouched) BEFORE denoise/threshold; `app/utils/image_window.py` serializes the heavy windows (`image_window()` semaphore that `preprocess`, `enhance_for_vision`, and `read_raw` inference all acquire) AND pins ALL heavy image work to ONE dedicated worker thread via `run_in_image_window` (routes call this, never bare `asyncio.to_thread`) — a semaphore alone still OOM'd because freed transients were not reusable across default-pool threads' allocator arenas; and the Dockerfile sets `MALLOC_ARENA_MAX=2`. Net: peak = steady state + ONE transient regardless of request concurrency; on 0.1 CPU no throughput is lost (CPU is the bottleneck).
- **Observability**: `/health` exposes `ocr_engine`, `ocr_importable` (cheap `find_spec` — must NEVER initialize the engine; /health has a 5 s probe timeout), `rss_mb` (Linux peak `ru_maxrss` — counts mmapped model/text pages, so it overstates cgroup pressure), plus `rss_current_mb` (VmRSS now), `cgroup_current_mb` / `cgroup_peak_mb` / `cgroup_limit_mb` (kernel-accounted container memory — the number OOM decisions use) and `uptime_s` (restart detection). Engine init logs `engine/init_ms/rss_mb`.
- **Dockerfile**: no torch — the OCR stack installs purely from `backend/requirements.txt`. **CI**: no model download/cache/warm steps (models ship inside the rapidocr wheel).

### Rule 8.2e — Adaptive relational grammars (concept maps + comparison matrices)

The visual engine chooses a representation from the material's INFORMATION STRUCTURE, not its subject name. Two relational grammars exist so chemistry, biology and CS material does not collapse into `generic` cards:

- **`concept_map`** (`visual_renderer.py`): `layout` is `tree` | `cycle` | `timeline`.
  - Planner supplies semantics only: node `id`, short `label`, one-clause `detail`, `parent` (tree), `marker` (timeline), optional cross-link `edges` and `relation`. Code owns every coordinate, wrapping decision, connector, arrowhead and font size.
  - Trees are tidy hierarchies with exactly one root, depth ≤ 4, and same-kind siblings. Shallow trees (root + 2–5 alternatives, no grandchildren, no cross-links) render left-to-right with leaf details inside cards; deeper trees render top-down. Top-down connectors are per-child cubic curves (never a shared elbow spine); non-hierarchy cross-links render as dashed arrows.
  - Detail printing is explicit: horizontal/timeline heroes print details in the node card; a tall tree prints header details in the hero and leaf details as callout cards (capped); a cycle draws labels only, so every stage detail becomes a card.
  - Cycles require 3–6 stages and close the loop in code; timelines require 3–6 events already in chronological order.
- **`comparison.table`**: a feature/condition matrix with 2–3 columns × 2–6 rows. Orientation is free: columns may be the alternatives (rows = shared features) or rows may be the alternatives (columns = shared metrics). More than three alternatives MUST transpose so alternatives are the rows — never drop one. `corner` names the row axis truthfully ("FEATURE", "DISTRIBUTION", …). Every column has exactly one non-empty value per row. `verdict` is a synthesis/decision rule, not a glossary definition of one cell.
- **Validation is visibility-first**: the task-link haystack includes only text the composed page actually renders. Composition chips suppressed as duplicates (generic/comparison/horizontal-or-timeline maps) do not satisfy required elements; promoted equation claims always render, so they always count. Exact-claim matching ignores presentation-only label punctuation (`:`) so "Worst-case: O(n)" matches a visible "Worst-case" label plus "O(n)" detail. A taxonomy heading synonym is tolerated only when every exact claim already passes verbatim — visible formulas are the stronger contract.
- **Focused repair**: when the planner selects `comparison` but emits no payload, one focused extraction call returns ONLY the table/panels object and merges it into the spec; a full nested re-emit proved unreliable on dense multi-column boards. Invalid/empty focused output returns None so the route refuses honestly.
- **No generative geometry**: these grammars are deterministic. Generative imagery remains limited to non-exact conceptual illustration; it never computes answers, coordinates, labels or equations.

### Rule 8.3 — Pricing model
- `/extract/text`: 1 credit if OCR succeeds standalone; **5 credits if escalation to Gemini fires** (credit check before Gemini call, line 55-57 of extract.py)
- `/extract/revision`: 1 credit (separate Gemini call for the learning layer)
- `/extract/diagram`: 5 credits always
- **Explain Visually** (`POST /api/extract/visual`): **0 additional student credits** — a bundled benefit of the 5-credit Diagram product, NOT a separate charge.
- **Entitlement is tied to a specific completed Diagram result**: a successful 5-credit Diagram extraction records one Explain Visually grant (via `record_diagram_grant()` in `credits_store.py`, storing `diagram_id` + `study_notes_json` in the `visual_explanations` table). The grant is created AFTER the diagram succeeds, never at request start, and is scoped to that device + `diagram_id`.
- **Lazy, on-click, generated exactly once**: Explain Visually is never generated automatically at extraction time — the student clicks the button. One successful visual generation per granted `diagram_id`; the stored result (`visual_url` for generative, `visual_svg` for deterministic, plus `render_mode`) is immutable once set (`set_visual_result()`). A second click on the same result returns the existing visual with `status: "already_generated"` — no second Gemini/image-model call, no regeneration button, no re-generation endpoint.
- **Hybrid pipeline — generative AI for visual creativity, never for exact typography**: Stage 1 = Gemini (`build_visual_spec()` in `vision_service.py`) reads the screenshot + study notes and emits a structured VisualSpec that includes `render_mode` ("deterministic" | "generative") and `text_required`. Gemini is the semantic/understanding layer only; it never renders images.
  - **`render_mode` decision is based on INFORMATION PRECISION, not subject names**: choose "deterministic" whenever correctness of text, symbols, equations, relationships, or exact structure is central to understanding (formulas, derivations, definitions, labeled flows, physics/math/engineering, chemistry mechanisms). Choose "generative" ONLY when the value is a rich conceptual illustration where exact text is NOT the main payload (conceptual scenes, biology/cell processes, analogy-like visuals). If in doubt, "deterministic".
  - **Deterministic branch** (`visual_renderer.py`): code renders a clean, flat, sanitized SVG from a bounded `DeterministicVisual` payload. Two presentations:
    - **Scene engine (preferred)** — universal educational primitives: `VisualScene` (`force_diagram` | `process_flow`). Gemini supplies ONLY semantics (which objects/vectors, `angle_deg` in degrees with 0=right/90=up, relative `length` 0.4–1.6, vector `tail` chains, angles between vectors, rotation arcs, relations as exact Unicode equations). Code owns ALL coordinates, arrowheads (`<polygon>` — never `<marker>`), arcs, labels, and typography. `force_diagram` = pivot/disk/point/block object + labeled vectors + angle arcs + rotation arc + relation card + "WHAT THE VISUAL SHOWS" caption. `process_flow` = labeled boxes + straight connectors + dashed feedback loop.
    - **Card fallback** — title / equations+meanings / ordered steps / key points (used when the concept fits neither scene kind, or as supplementary sections under a scene).
    Byte-deterministic (same spec → same SVG), no randomness, no LLM at render time. Never Gemini-emitted SVG. **Rule: arrowheads must be standalone `<polygon>` elements — never double-wrapped inside another element's attribute (breaks Chromium).**
  - **Generative branch** (`generate_visual()` in `visual_service.py`): Pollinations renders a raster illustration from the spec. Exact math/labels are NOT the payload here.
- **Quality gate + one hidden retry**: generative renders pass `_quality_pass()` (brightness/size/emptiness — never blind white-flattening) AND, when `text_required=true`, the OCR legibility gate `_legibility_pass()` (rejects renders with essentially no readable tokens at sufficient confidence). `text_required=false` (pure illustration) skips the OCR gate so a perfectly good textless visual is never rejected. On gate failure the pipeline retries ONCE with an emphatic white-background hint, then returns an honest "Visual explanation unavailable for this material." — never a fake success state.
- **Trust labeling**: the generated visual is always labeled "AI-generated visual explanation" (frontend + product copy) and must NOT be presented as a verified reconstruction. Deterministic SVG reconstruction (`StudyNotes.diagram`) remains the accuracy/trust path where supported; unsupported reconstruction types are marked `best_effort` and never presented as verified.
- **Storage**: generative visuals are uploaded to ImgBB for the MVP (not R2); deterministic visuals are stored inline as sanitized SVG in the entitlement row (`visual_svg`).
- 50 free credits/month = **50 OCR-only requests**, **50 revision enhancements**, or **10 Gemini-escalated requests**
- When Gemini returns 429 (rate limit), text endpoint falls back to OCR + a note, charged 1 credit. Diagram endpoint returns a clean "high demand" message.
- **Provider status**: Pollinations is the current MVP image-generation provider (config: `POLLINATIONS_API_KEY` optional for the registered free tier, else anonymous; `POLLINATIONS_MODEL=sana`). Gemini image generation (`gemini-*-flash-image`) is NOT used — free-tier accounts return 429 for every image model.

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

*Last updated: October 2026 (worked mathematics is verified or refused — number_line family; safe AST parser with exact-decimal literals; expression-domain preservation; transformed steps verified algebraically; excluded values must be visible; render-time re-verification; permanent hosted CI; OCR engine = RapidOCR with fail-safe degradation + /health observability incl. cgroup memory; gate mixed digit-letter heuristic is word-level; preprocess resolution cap + image-window serialization close the measured 3-way-concurrency OOM; adaptive relational grammars add concept-map tree/cycle/timeline and comparison feature matrices with visibility-first validation and focused table repair)
