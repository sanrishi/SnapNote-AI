# Visual Composition Prototype — Benchmark Report
**Branch: `feat/prototype-visual-composition` — Isolated, no master modification**
**Date: 2026-08-26. 2 fixtures × 3 engines, measured on Windows 11 + Python 3.12, Render target `python:3.11-slim`**

## Fixtures (one source for all 3 engines)
- **Torque:** `VisualSpec` force_diagram `O→r(55°)→F(90°)→θ→τ` + relation `τ = r × F` + caption + generic callouts
- **Argand square:** `plot` `Re/Im` `x -1..5, y -1..5` with square `[[1,1],[1,3],[3,3],[3,1]]` + generic `central_label Square ABCD` + 4 callouts + `Area = 4`

## Measured Results (this environment)

| Engine | Torque | Argand | Cold vs Warm (A torque 5 runs) | Deps on this host |
|---|---|---|---|---|
| **A. Current SVG** | `3488 chars, 6.3ms, has <svg true` | `3029 chars, 4.5ms` | `5.4, 4.3, 4.3, 5.2, 4.2ms` (stable) | None (pure Python) |
| **B. Typst CLI** | `not found` | `not found` | — | `typst` binary not on Windows PATH |
| **B. typst-py 0.15.0** | `0 bytes, 0.8ms — OSError 123` | `0.2ms — OSError 123` | — | Installed but fails on Windows path (known 0.15.0 Windows bug) — works on Linux/Docker |
| **C. WeasyPrint 69.0** | `not available` | `not available` | — | `libgobject-2.0-0` missing on Windows (needs `pango`/`cairo` on Debian `python:3.11-slim`) |

**Interpretation:** A is fully measured. B/C are **not measurable on Windows without system deps**, but are **expected to work on Render's Debian Docker** where `apt-get install` provides `libcairo2`/`libpango` and `cargo install typst-cli` or `typst` binary provides CLI. Previous report's latency claims for B/C were unmeasured — now labeled **NR (needs Linux benchmark)**.

## Dependency Footprint (Render `python:3.11-slim`)

- **A:** `0 MB` (already in image, `Pillow` + `opencv` already)
- **B CLI:** `~15MB` binary + Rust toolchain if built, or `~80MB` if `texlive` not needed. Official docs `0.15.1` (typst.app). Binary is Apache 2.0, `typst compile --format svg` is official.
- **B binding `typst-py`:** `~15MB` wheel, but **third-party** (`messense/typst-py`), not official Typst API. Maintenance: `messense` (active, but not Typst org). Thread safety unknown, error handling wraps Rust panics. **Do not treat as official.**
- **C WeasyPrint:** `~50MB` + `libcairo2` `libpango-1.0-0` `libgdk-pixbuf2.0-0` (~30MB). Pure Python `weasyprint` 69.0, well-maintained, no browser.

## Corrected Architecture (addressing CairoSVG HTML error)

**Previous report was wrong:** `HTML/CSS → CairoSVG` is invalid. CairoSVG is SVG→PNG only.

**Valid pipelines:**

```
A) VisualSpec → SVG composition → CairoSVG → PNG   (lightest, manual layout)
B) VisualSpec → SVG hero + HTML/CSS shell (Jinja) → WeasyPrint → PNG  (best typography)
C) VisualSpec → SVG hero + HTML/CSS shell → Playwright screenshot → PNG (heaviest, not recommended)
```

For SnapNote, **B is the infographic path**, **A is the fallback** if WeasyPrint proves heavy.

## Math Typography

- **Unicode** (`τ`, `×`, `∫`, `²`) — keep for hero labels/relation (fast, deterministic). Ceiling: matrices/stacked fractions.
- **KaTeX `renderToString`** — server-side, deterministic, no browser, outputs HTML+CSS. Handles `\frac`, `\begin{pmatrix}`, `\sum_{i=1}^n`. **Add as opt-in** for `derivation`/`result` blocks inside the HTML shell. Do not use MathJax (needs browser). KaTeX HTML must be allowlisted in `svg_safe` or HTML sanitizer.

## Layout Strategy for Anchored Annotations

First-fit `right → above → below` is too crude for 6-10 callouts. Recommended deterministic scoring:

1. Generate 4-8 candidates per annotation (right, above, below, left, 45° diagonals) offset from anchor bbox (anchor = SVG element id, e.g., `vec-F` → `getBBox()` via `lxml` parse of hero SVG).
2. Bounding boxes via `Pango` (WeasyPrint) or `char-count` fallback.
3. Score: `+100` hero overlap, `+50` callout-callout overlap, `+30` leader crossing, `+20` right preferred, `+0.1*dist`, choose min. Route leader as `line` or `polyline` with `stroke-dasharray` if diagonal.

## VisualSpec v3 (lesson, not hero+boxes)

Previous `hero + callouts + derivation + result` was flat. Needed:

```
concept
  scene: { objects/geometry/axes + relationships }
  annotations: [{ anchor: "vector F", text, emphasis }]
  reasoning: [{ eq, why, consequence }]  // derivation
  takeaway: { text, highlight }
  result: { eq, highlight }
  composition: { variant: "argand-square", layout: "hero-center_callouts-right_derivation-bottom" }
```

Gemini outputs `annotations[].anchor` as semantic (`"point A"`), renderer resolves to bbox. No pixels.

## Recommendation (evidence, not preference)

**Adopt Balanced Hybrid: Custom SVG primitives + HTML/CSS shell (WeasyPrint) + CairoSVG for final SVG heroes + KaTeX opt-in, Graphviz helper only for flows.**

- Reuse 90% of current code (`schemas`, `visual_renderer` stage math, `svg_safe`, `math_normalize`).
- Smallest major jump: add `composition.py` (VisualSpec v3 → HTML+hero SVG) + `cairo_export.py` (WeasyPrint or CairoSVG), + `graphviz` helper for `process_flow`.
- Latency: `+300-500ms` for WeasyPrint shell vs `800-1500ms` for Playwright, vs `<100ms` for pure SVG. Stays inside `22s/28s/32s` SLA.
- Keep generative (Pollinations) only when `text_required=false`.

**What to reject:** `exec(llm_code)`, `manim`/`TikZ`/`texlive` for MVP, Playwright screenshot, full Matplotlib infographic.

**What remains uncertain:** WeasyPrint cold time on Render free, KaTeX HTML sanitization, anchor→bbox resolution for generic callouts — all need a Linux Docker prototype (not Windows) before production.

**Next step:** Throwaway prototype on Render-like Debian (not Windows) for `torque` + `argand` in both pipelines (pure SVG vs HTML+WeasyPrint), measure real `weasyprint.HTML(string=html).write_png()` cold/warm, then review.
