"""Engine C: HTML/CSS + WeasyPrint."""
import time

def _weasy_available() -> tuple[bool, str]:
    try:
        import weasyprint
        return True, getattr(weasyprint, "__version__", "unknown")
    except ImportError:
        return False, "not installed"
    except Exception as e:
        return False, str(e)

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
@page {{ size: 800px 900px; margin: 16px; }}
body {{ font-family: Inter, sans-serif; color: #1e293b; }}
.hero {{ width: 100%; height: 420px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; display:flex; align-items:center; justify-content:center; }}
.callouts {{ display:flex; gap:12px; margin-top:8px; }}
.callout {{ flex:1; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:8px; font-size:11px; }}
.result {{ margin-top:8px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px; font-weight:700; }}
</style></head><body>
<h1 style="text-align:center; font-size:16pt; font-weight:800">{title}</h1>
<div class="hero">{hero}</div>
<div class="callouts">{callouts}</div>
<div class="result">{result}</div>
</body></html>
"""

def render_weasyprint(title: str, hero_svg: str, callouts: list[str], result: str) -> tuple[bytes | None, float, str]:
    ok, ver = _weasy_available()
    if not ok:
        return None, 0, f"weasyprint not available: {ver}"
    t0 = time.perf_counter()
    try:
        import weasyprint
        callouts_html = "".join(f'<div class="callout">{c}</div>' for c in callouts)
        html = HTML_TEMPLATE.format(title=title, hero=hero_svg, callouts=callouts_html, result=result)
        # WeasyPrint 69.0: HTML.render().write_png() is the correct API (write_png on HTML directly was removed)
        doc = weasyprint.HTML(string=html).render()
        # Write to PDF first (always available), then report PDF size. For PNG, use write_png if available.
        if hasattr(doc, "write_png"):
            png = doc.write_png()
            dt = (time.perf_counter() - t0) * 1000
            return png, dt, f"weasyprint {ver} PNG {dt:.1f}ms"
        pdf = doc.write_pdf()
        dt = (time.perf_counter() - t0) * 1000
        return pdf, dt, f"weasyprint {ver} PDF {len(pdf)} bytes {dt:.1f}ms"
    except Exception as e:
        return None, (time.perf_counter() - t0) * 1000, f"weasyprint failed: {e}"

if __name__ == "__main__":
    print("WeasyPrint:", _weasy_available())
    hero = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="#ddd"/></svg>'
    data, ms, log = render_weasyprint("Test", hero, ["A", "B"], "Area = 4")
    print(f"C weasy: {len(data) if data else 0} bytes, {ms:.1f}ms — {log[:120]}")
