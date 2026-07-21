# SnapNote AI — Product Plan

## 1. Product Overview

**Problem:** Students watching online lectures (YouTube, IITM BS, NPTEL, Coursera) take continuous screenshots of slides/diagrams. Gallery becomes a chaotic mess of `Screenshot_20260713_193022.png`. Exam time — 500 photos, no context, no searchability.

**Solution:** A Chrome Extension + FastAPI backend that:
- Captures video frames with one click (keyboard shortcut)
- Auto-tags them based on video title/week/context
- Extracts text (free OCR), tables (Markdown), diagrams (SVG/Mermaid)
- Exports everything as clean `.md` files → paste into Notion/Obsidian/GoodNotes

**Target Audience:** Online learners — IITM BS students, engineering students, UPSC aspirants, Coursera/NPTEL users.

**Revenue Model:** Credits-based freemium. ₹0 for basic OCR, ₹99-₹299/month for Vision LLM features.

---

## 2. Phase 1 Scope (Strict)

### IN SCOPE — Only Online (Chrome Extension)
- [ ] Two shortcuts: `Ctrl+Shift+T` (text/table, free OCR route) and `Ctrl+Shift+D` (diagram, Vision LLM route with credits)
- [ ] Auto-context tagging from browser tab metadata
- [ ] User chooses route — no auto-classification on backend
- [ ] Markdown export + Copy to Clipboard
- [ ] Google Auth (Firebase) — required for Chrome Web Store policy
- [ ] Daily credit limit system (anti-abuse + cost control)

### OUT OF SCOPE — Phase 2+
- [ ] Mobile app / offline blackboard scanner
- [ ] Real-time Notion/OneNote API sync
- [ ] Cloud storage and sync across devices
- [ ] Team/collaboration features

---

## 3. Architecture Overview

```
┌──────────────────────┐
│   Chrome Extension    │  ← Vanilla JS, Manifest V3
│   (Content Script)    │
└─────────┬────────────┘
          │ screenshot (base64)
          │ mode: "text" | "diagram"  ← USER CHOOSES
          ▼
┌──────────────────────┐
│   FastAPI Backend     │  ← Python, async, lightweight
│   (Vercel Serverless) │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│  Two Separate Routes (User-Choice)       │
│                                          │
│  ─── Ctrl+Shift+T (Text/Table) ──────    │
│  Step 1: Image Preprocessing             │
│  Step 2: EasyOCR                         │
│  Step 3: Format as Markdown              │
│  Cost = ₹0                               │
│                                          │
│  ─── Ctrl+Shift+D (Diagram) ────────     │
│  Step 1: Image Preprocessing             │
│  Step 2: Vision LLM (GPT-4o-mini)        │
│    - Clean image (remove noise/glare)    │
│    - Extract text labels                 │
│    - Preserve diagram as-is (NO SVG)     │
│  Step 3: Upload cleaned image to R2      │
│  Step 4: Return markdown with ![diag]    │
│  Cost = ₹0.03-0.10 + storage             │
└──────────────────────────────────────────┘
```

---

## 4. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | Vanilla JS, HTML, CSS | Chrome Extension V3. No framework bloat |
| Backend | FastAPI (Python) | Async, lightweight, blazing fast |
| Auth | Firebase Auth (Google login) | Required by Chrome Web Store. Free tier. |
| OCR | EasyOCR / PaddleOCR | 100% free, open-source, good accuracy |
| Vision LLM | GPT-4o-mini / Claude 3.5 Haiku | Cheap ($0.15/M tokens). Only for complex content |
| Image Storage | Cloudflare R2 / Firebase Storage | Cheap, no egress fees. Store cleaned diagrams |
| Hosting | Vercel Serverless Functions | 1-2s cold start. 100GB-hours/mo free tier |
| DB (minimal) | SQLite → PostgreSQL later | Just users + credits |
| Rate Limiting | Redis (Upstash free tier) | 10k req/day free |

---

## 5. API Routes

```
POST /api/auth/google
  Body: { idToken: string }
  Response: { accessToken, creditsRemaining }

POST /api/extract/text
  Headers: Authorization: Bearer <token>
  Body: FormData { image: File, context: { title, url, timestamp? } }
  Response: {
    type: "text" | "table",
    markdown: string,
    tags: string[],
    creditsUsed: 1
  }

POST /api/extract/diagram
  Headers: Authorization: Bearer <token>
  Body: FormData { image: File, context: { title, url, timestamp? } }
  Response: {
    type: "diagram",
    markdown: string,     // cleaned text labels + context
    imageUrl: string,      // link to cleaned image on R2
    imageWidth: number,
    imageHeight: number,
    tags: string[],
    creditsUsed: 5
  }

GET /api/user/credits
  Response: { creditsRemaining: number, creditsUsed: number }

POST /api/feedback
  Body: { extractionId, rating, comment? }
```

---

## 6. Chrome Extension Structure

```
extension/
├── manifest.json          # Manifest V3
├── background.js          # Service worker, keyboard shortcut handler
├── content.js             # Injected into video page, capture logic
├── popup/
│   ├── index.html         # Popup UI
│   ├── popup.js           # Popup logic
│   └── popup.css
├── lib/
│   ├── auth.js            # Firebase auth wrapper
│   ├── api.js             # Backend API client
│   └── storage.js         # Local storage helpers
├── assets/
│   └── icon-{128,48,16}.png
└── _locales/              # i18n if needed
```

### User Flow:
1. User watches video on YouTube/IITM portal
2. **Text/Table:** Presses `Ctrl+Shift+T` → free OCR route
   **Diagram:** Presses `Ctrl+Shift+D` → Vision LLM route (consumes credits)
3. Content script captures current frame as base64
4. Sends to background worker with tab title/url + mode
5. Background worker calls `/api/extract/text` or `/api/extract/diagram`
6. Result shows in a toast/side panel
7. User clicks "Copy Markdown" or "Download .md"
8. Pastes into Notion/Obsidian

---

## 7. Cost-Saver Architecture (Secret Weapon)

### Without this architecture → Bankruptcy ☠️
Every screenshot → Vision LLM API call → ₹0.10 each
20 screenshots/lecture × 30 lectures/month = ₹60/student
100 students = ₹6000/month API bill at ₹99 pricing = **NEGATIVE MARGIN**

### With this architecture → Profit ✅

| Shortcut | Route | Pipeline | Cost/user |
|----------|-------|----------|-----------|
| `Ctrl+Shift+T` | `/api/extract/text` | EasyOCR → Markdown | ₹0 |
| `Ctrl+Shift+D` | `/api/extract/diagram` | Vision LLM → clean image + R2 storage | ₹0.03-0.10 |

**Key Decision: User chooses, NOT backend.** No auto-classification needed. Simple, fast, accurate.

### Text/Table Pipeline Logic:
```python
async def extract_text(image: bytes) -> ExtractionResult:
    processed = preprocess_image(image)
    ocr_result = easyocr_reader.readtext(processed)

    # Check if table (aligned rows/columns of text)
    if is_table_layout(ocr_result):
        markdown = format_as_table(ocr_result)
    else:
        markdown = format_as_text(ocr_result)

    return ExtractionResult(type="table" if is_table else "text", markdown=markdown)
```

### Diagram Pipeline Logic:
```python
async def extract_diagram(image: bytes, context: dict) -> ExtractionResult:
    processed = preprocess_image(image)

    # Step 1: Vision LLM cleans image + extracts text labels
    llm_result = await call_vision_llm(
        image=processed,
        prompt="1. Remove background noise and glare. "
               "2. Extract all text labels as markdown. "
               "3. Describe diagram type. Do NOT generate SVG."
    )

    # Step 2: Upload cleaned image to R2 storage
    cleaned_image = llm_result.cleaned_image
    image_url = await upload_to_r2(cleaned_image, context)

    # Step 3: Return markdown with embedded image + labels
    markdown = f"## Diagram: {llm_result.title}\n\n"
    markdown += f"![{llm_result.title}]({image_url})\n\n"
    markdown += "### Labels & Text\n" + llm_result.text_labels

    return ExtractionResult(type="diagram", markdown=markdown, imageUrl=image_url)
```

---

## 8. Database Schema (Phase 1 — Minimal)

```sql
-- Users table (Firebase Auth + local DB)
CREATE TABLE users (
    id TEXT PRIMARY KEY,          -- Firebase UID
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    credits_remaining INTEGER DEFAULT 100,  -- Monthly free tier
    credits_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extractions log (for analytics + abuse prevention)
CREATE TABLE extractions (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    layout_type TEXT,              -- text | table | diagram | mixed
    pipeline_used TEXT,            -- ocr_only | vision_llm
    credits_consumed INTEGER DEFAULT 0,
    context_title TEXT,            -- video title
    context_url TEXT,              -- video URL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Credit plans
CREATE TABLE plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,            -- "Free", "Pro", "Unlimited"
    monthly_credits INTEGER NOT NULL,
    price_inr INTEGER NOT NULL,
    features TEXT                  -- JSON
);
```

---

## 9. Monetization Strategy

### Pricing (Indian Market — ₹)

| Plan | Price | Credits/Month | Features |
|------|-------|---------------|----------|
| Free | ₹0 | 50 | OCR only (text + tables). No diagrams |
| Pro | ₹99/mo | 300 | OCR + Vision LLM (diagrams). Priority queue |
| Pro Annual | ₹999/yr | 3600 | Same as Pro. 2 months free |
| Pay-as-you-go | ₹5 per 10 credits | — | No subscription. Buy credits anytime |

**Credit System:**
- Text extraction: 1 credit
- Table extraction: 2 credits
- Diagram extraction: 5 credits
- 1 credit = ₹0.33 (Pro plan) → API cost per diagram ~₹0.10 → **70% margin**

### Anti-Abuse:
- Daily cap: 50 requests/day max
- Rate limit: 10 requests/minute
- Suspicious patterns → flag + manual review

---

## 10. Development Roadmap

### Week 1-2: Backend MVP
```
Day 1:  Project setup, FastAPI boilerplate, directory structure
Day 2-3: Image preprocessing module (contrast, binarize, deskew)
Day 4-5: EasyOCR integration + text extraction route
Day 6-7: Table detection logic (grid alignment check)
Day 8-9: Vision LLM integration + diagram cleaning route
Day 10:  Cloudflare R2 upload + Markdown with embedded image
Day 11-12: Firebase Auth integration, user/credits API
Day 13:  Rate limiting, error handling, logging
Day 14:  Testing with real screenshots from stats lectures
```

### Week 3-4: Chrome Extension
```
Day 15-16: Manifest V3 setup, content script, keyboard shortcut
Day 17-18: Screenshot capture logic (HTML canvas → base64)
Day 19-20: Background worker, API communication
Day 21:  Popup UI, clipboard copy, .md download
Day 22:  Firebase Auth integration in extension
Day 23-24: Toast feedback, error states, loading UX
Day 25:  Testing with real YouTube + IITM portal
Day 26:  Documentation, demo video recording
Day 27-28: Buffer for bugs, edge cases, polish
```

### Week 5: Launch
```
Day 29:  Deploy backend to Render/Railway
Day 30:  Submit to Chrome Web Store
Day 31:  Share demo video in IITM BS groups, Telegram channels
Day 32:  Collect feedback, fix critical bugs
```

---

## 11. Success Metrics (Phase 1)

- **200+** Chrome Web Store installs (organic)
- **50+** daily active users
- **<10%** API cost-to-revenue ratio
- **>80%** extraction accuracy on real lecture screenshots
- **Zero** negative feedback on Chrome Web Store

---

## 12. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Google rejects extension | High | Use Firebase Auth from day 1. Minimal permissions |
| API costs explode | High | Tiered pipeline. Free OCR first. Hard daily caps |
| Low accuracy on complex diagrams | Medium | Collect user feedback. Fine-tune prompts iteratively |
| Competitors copy idea | Medium | Execution moat: Indian pricing + regional language support |
| No paying users | High | Validate with ₹99 pre-sale BEFORE full development |

---

*Plan v1.0 — Last updated: July 2026*
