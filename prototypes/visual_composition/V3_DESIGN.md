# VisualSpec v3 Composition — Design Proposal

## 1. Problem (from milestone report)
`render_visual_lesson()` works, but its input (`VisualSpec` v2) carries geometry
truth (`scene`) plus prose derivation only implicitly:
`extract_lesson_content()` *infers* callouts from vector labels, reasoning from
angle/arc captions, result from `scene.relation`. The rich Torque lesson
(O / r / F / θ chips, reasoning chain, highlighted `τ = r × F`, takeaway) and
the Argand lesson (A/B/C/D strip, conjugate+side+modulus reasoning,
highlighted `Area = 4`) could not be expressed — only derived. Derivation by
inference is what produced the thin "2 chips + Axes" output vs the approved
rich renders.

## 2. Smallest compatible extension
Add ONE optional block, `DeterministicVisual.composition: LessonComposition |
None = None`. Absent (`None`) = v2 behavior exactly (derived content). Present
= explicit lesson, preferred over derivation. No existing field changes, no
v2 fixture breakage, old `visual_service.generate_visual` path untouched.

## 3. Schema (typed, bounded, `extra="forbid"`)
```python
class CompositionCallout(BaseModel, extra="forbid"):
    id: str      # ^[a-z0-9_-]{1,32}$ — stable anchor key, e.g. "vec-r", "pt-A"
    label: str   # <= 24 chars, e.g. "r" / "A"
    value: str   # <= 80 chars, e.g. "55° lever" / "z = 1+i → (1,1)"

class CompositionReasoningStep(BaseModel, extra="forbid"):
    id: str            # ^[a-z0-9_-]{1,32}$
    expression: str    # <= 120 chars, exact math, byte-identical passthrough
    explanation: str   # <= 200 chars

class CompositionResult(BaseModel, extra="forbid"):
    expression: str    # <= 120 chars, e.g. "τ = r × F" / "Area = 4"
    emphasis: bool = True

class LessonComposition(BaseModel, extra="forbid"):
    title: str = ""      # <= 80, overrides deterministic.title in composition
    framing: str = ""    # <= 200, one-line concept framing (subtitle)
    callouts: list[CompositionCallout] = []   # <= 6, explicit order kept
    reasoning: list[CompositionReasoningStep] = []  # <= 4, explicit order kept
    result: CompositionResult | None = None
    takeaway: str = ""   # <= 200
```
Bounds protect layout (chips grid, reasoning band, result highlight). Unknown
fields fail validation (never guessed). All optional except when the block is
present, at least one of callouts/reasoning/result must be non-empty.

## 4. Separation of concerns
```
VisualSpec v3
 ├─ semantic/geometry truth  (scene, vectors, plot ranges, points)
 ├─ visual-family info        (scene_kind → family; unchanged)
 └─ composition info          (NEW: callouts/reasoning/result/takeaway)
```
Composition references geometry by `id`/label (e.g. callout `id: "vec-r"`
describes vector `r`) but never carries coordinates. Renderer resolves
anchors; Gemini never outputs pixels (unchanged rule).

## 5. Validation (fail closed)
- v2 geometry validation unchanged.
- If `composition` present: ids match `^[a-z0-9_-]{1,32}$`, unique within
  their list; lengths enforced; counts enforced (callouts ≤ 6, reasoning ≤ 4);
  result expression non-empty when result present; at least one non-empty
  section; `extra="forbid"` rejects unknown keys.
- Mathematical values are NOT re-derived or rewritten — validator checks
  shape only; exact strings pass through byte-identical.

## 6. Extraction (prefer explicit, fallback to derived)
`extract_lesson_content()`: if `deterministic.composition` is present and
valid → build `LessonContent` directly from it (title/framing→title/subtitle,
callouts→chips, reasoning steps joined, result→highlight, takeaway).
Else → current derivation from scene (unchanged fallback). No markdown
parsing, no LLM rewriting.

## 7. Torque / Argand ground truth mapping
- Torque: callouts O("pivot") / r("55° lever") / F("90° force") / θ("θ = 35°,
  between r, F"); reasoning θ=90−55=35, τ=r×F, right-hand rule; result
  τ = r × F; takeaway door-handle sentence. All values from
  ground_truth_torque.json.
- Argand: callouts A/B/C/D with exact z and coords from
  ground_truth_argand.json points; reasoning conj/side/modulus; result
  Area = 4; takeaway side-squared sentence.

## 8. What this does NOT do
- No new visual families, no subject-specific fields, no renderer forks.
- No prompt changes in this step (prompt update is a separate, later step;
  v3 specs in tests/bench are constructed in code from ground truth).
- No wiring into live Explain Visually route; old path untouched.
- Flow/process families reuse the same block (nodes → callouts) with no
  extra fields.
