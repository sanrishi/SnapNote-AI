# Generative visual providers — measured capability review

Status: research note, October 2026. This document records what was verified
from provider documentation and one live probe of the current production
provider. It deliberately does **not** recommend switching providers on the
basis of marketing claims; any switch must pass the benchmark protocol below.

## Why this exists

SnapNote's Explain Visually pipeline is hybrid:

- **Deterministic SVG/Typst** renders every exact equation, table cell, label,
  arrow, coordinate and verified interval. This is the trust path.
- **Generative raster** is used only when the teaching payload is a conceptual
  illustration and exact text is not the point (`text_required=false`).

The question was whether a better image model could carry more of the visual
load. The answer from the evidence is: not for exact content, and not yet for
conceptual illustration without a paid tier. Deterministic rendering stays the
contract for anything a student could be examined on.

## Current production provider: Pollinations (`sana`)

Configured in `backend/app/config.py`:

| Setting | Value |
|---|---|
| Endpoint | `https://image.pollinations.ai/prompt/` (GET) |
| Model | `sana` |
| Size | 768 × 768 |
| Timeout | 25 s |
| Auth | optional `POLLINATIONS_API_KEY`, anonymous otherwise |

Live probe on 2026-10-31 (concept: photosynthesis; no exact text required):

- First call: HTTP 200, **9.96 s**, 47,589 bytes raw, 254,381 bytes after PNG
  normalization. The quality gate **failed** it.
- Second call within the same minute: HTTP **402**,
  `FREE_LIMIT — 1 image every 45 seconds per IP`.
- The render itself was a decorative radial leaf pattern with a grey gradient
  background, a `pollinations.ai` watermark, and pseudo-text (`Eclvt`) in the
  centre. It contained none of the requested teaching relationships.

This is consistent with the existing product rule: a clean-looking image with
garbled labels is the failure mode, not a success. The current quality +
legibility gates correctly reject it. It also shows the free tier cannot
support even modest concurrency: one image per 45 s per IP.

## Candidate providers (documented facts only)

### OpenAI Images API

- Generation **and editing** endpoints; the Responses API accepts image inputs
  and returns image outputs, so reference-image workflows are supported.
- Models: `gpt-image-2.5-sunburst` (editing precision) and
  `gpt-image-2.5-flare` (speed).
- Standard pricing: image input **$8 / 1M tokens**, image output
  **$30 / 1M tokens**, cached image input **$2 / 1M tokens**; text input
  **$5 / 1M tokens**. Per-image cost depends on size/quality and must be
  computed with OpenAI's calculator, not assumed.
- Rate limits are account-tier based; not measured here (no paid key in this
  environment).

Source: <https://developers.openai.com/api/docs/guides/image-generation> and
<https://developers.openai.com/api/docs/pricing> (accessed 2026-10-31).

### Google Gemini image models ("Nano Banana")

- `gemini-nano-banana-2.1` is documented for accurate text rendering,
  conversational editing, multi-turn character consistency and 1K/2K/4K
  output.
- Pricing (paid tier): **$0.0336 / 1K**, **$0.0504 / 2K**, **$0.113 / 4K**
  image; image input **$1.50 / 1M tokens**.
- `gemini-3.1-flash-image`: **$0.067 / 1K**, **$0.101 / 2K**,
  **$0.151 / 4K**.
- `gemini-3.1-flash-lite-image`: **$0.0336 / 1K**.
- **Free tier for image output is documented as "Not available"** on these
  models. This matches the production experience that free-tier accounts
  returned 429 for every image model, which is why Pollinations is the current
  MVP provider.

Sources: <https://ai.google.dev/gemini-api/docs/pricing> and
<https://ai.google.dev/gemini-api/docs/image-generation> (accessed 2026-10-31).

### Black Forest Labs (FLUX)

- `FLUX 3 Image`: **$0.041 / 768²**, **$0.048 / 1K**, **$0.10 / 2K**,
  **$0.607 / 4K**.
- `FLUX 3 Image Klein`: **$0.009 / 1K**, **$0.014 / 2K** (cheapest documented
  option found).
- Editing/reference-image support was not verified from the pricing page, so
  it is not claimed here.

Source: <https://bfl.ai/pricing> (accessed 2026-10-31).

### Ideogram

Known for strong text rendering, but the pricing page returned HTTP 403 to the
automated fetch and no API key is available. **Not evaluated.** Do not list it
as a supported option until its API, pricing and licensing are verified with a
real account.

## Decision for this increment

1. **No provider integration.** The trust work (traceable table verdicts,
   visible-only claims, real cycle/timeline fixtures) is independent of image
   generation. Adding an unmeasured paid dependency would make the release
   less verifiable, not more.
2. **Keep Pollinations for illustration only**, behind the existing
   `generate_visual()` seam and the existing quality/legibility gates. Its
   measured free-tier limit means it must never be on the automatic extraction
   path; it is already lazy and on-click.
3. **Exact content never moves to a raster model.** Equations, table cells,
   intervals, graph data and labels stay in deterministic SVG/Typst.
4. If a paid key becomes available, swap providers only after the benchmark
   below passes at a rate that justifies the cost.

## Benchmark protocol for a future provider

Run 12 prompts per subject group (maths, chemistry, biology, CS, physics,
engineering, statistics), each with a ground-truth label set. For every render
record:

- OCR token recall/precision against the ground-truth labels, plus whether the
  existing `_legibility_pass()` would accept it.
- `_quality_pass()` result and visual inspection (background, watermark,
  anatomy/structure correctness, absence of invented numbers).
- p50/p95 latency and failure rate over 20 calls.
- Measured cost per accepted image (not per API call; rejected retries count).
- Rate-limit behaviour under two concurrent users, with the same 25 s budget
  the product uses today.
- Whether the provider supports reference-image edits, and whether those edits
  preserve deterministic content supplied by SnapNote (for layered workflows:
  generated scene background + SVG overlay).

Acceptance bar for replacing Pollinations for **illustration**:
≥ 80% quality-gate pass rate, no watermark, p95 ≤ 20 s, cost ≤ $0.02 per
accepted image, and no readable-text reliance.

A provider may replace deterministic SVG **only** if it achieves exact symbol
and label fidelity across all 84 prompts with zero invented values — a bar no
current model documents.

## Operational notes

- All candidates are remote APIs, so the 512 MB Render instance only pays for
  the downloaded image and the existing bounded preprocessing pipeline; no
  local model weights.
- Keys stay in `backend/.env` / environment variables, never in code.
- Generative output is always labelled "AI-generated visual explanation" and
  is never presented as a verified reconstruction.
