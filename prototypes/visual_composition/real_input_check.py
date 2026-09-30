"""Gate 3: real lecture boards through the REAL understanding pipeline.

For each image: enhance -> extract_study_notes -> build_visual_spec (real
Gemini, same calls the route makes, contradiction repair included) ->
validate_lesson_spec -> render_v3_visual (composed PNG when Typst is on
PATH) -> ImgBB upload when configured. Records the 12 Gate 3 fields.
No credits touched (service functions directly, no routes).
"""
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from app.services.preprocessor import enhance_for_vision
from app.services.vision_service import build_visual_spec, extract_study_notes
from app.utils.visual_lesson import render_v3_visual, select_family, validate_lesson_spec

IMAGES = [
    # Gate 3 corpus: 12 genuinely real boards (3 lecture samples + 9 stress
    # fixtures). Covers physics, math, engineering/control, coaching slides,
    # formula-heavy boards, dark theme, and thin-evidence prose.
    ("physics-torque-board", "../../website/samples/physics-demo.jpg"),
    ("math-complex-board", "../../website/samples/math-demo.jpg"),
    ("engineering-third", "../../website/samples/engineering-demo.jpg"),
    ("stats-table", "../../stress_test_fixtures/1_stats_table.png"),
    ("pid-flowchart", "../../stress_test_fixtures/2_flowchart.png"),
    ("eng-math-formulas", "../../stress_test_fixtures/3_math_formulas.png"),
    ("hindi-coaching", "../../stress_test_fixtures/4_hindi_coaching.png"),
    ("youtube-coaching", "../../stress_test_fixtures/5_youtube_coaching.png"),
    ("messy-prose-plain", "../../stress_test_fixtures/6_plain_prose.png"),
    ("biology-prose", "../../stress_test_fixtures/7_biology_prose.png"),
    ("chemistry-equilibrium", "../../stress_test_fixtures/8_chemistry_equilibrium.png"),
    ("dark-cs-bst", "../../stress_test_fixtures/9_dark_bst.png"),
]


class _RepairWatcher(logging.Handler):
    """Counts contradiction-repair attempts (Gate 3 field 5)."""

    def __init__(self) -> None:
        super().__init__()
        self.repairs = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "attempting one repair" in record.getMessage():
            self.repairs += 1

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


async def one(name: str, path: str) -> dict:
    t0 = time.perf_counter()
    report: dict = {"image": name}
    try:
        with open(path, "rb") as f:
            raw = f.read()
        report["bytes"] = len(raw)
        enhanced = await asyncio.to_thread(enhance_for_vision, raw)
        t_notes = time.perf_counter()
        notes = await extract_study_notes(enhanced)
        report["notes_ms"] = round((time.perf_counter() - t_notes) * 1000, 1)
        report["topic"] = (notes.topic.title if notes.topic else "")
        report["formulas"] = len(notes.key_formulas or [])
        t_spec = time.perf_counter()
        watcher = _RepairWatcher()
        logging.getLogger("app.services.vision_service").addHandler(watcher)
        try:
            spec = await build_visual_spec(enhanced, notes)
        finally:
            logging.getLogger("app.services.vision_service").removeHandler(watcher)
        report["spec_ms"] = round((time.perf_counter() - t_spec) * 1000, 1)
        report["repair_triggered"] = watcher.repairs > 0
        report["repair_attempts"] = watcher.repairs
        report["render_mode"] = str(spec.render_mode)
        det = spec.deterministic
        report["spec_title"] = det.title
        report["has_scene"] = det.scene is not None
        report["scene_kind"] = det.scene.scene_kind if det.scene else None
        report["has_composition"] = det.composition is not None
        if det.composition is not None:
            comp = det.composition
            report["composition"] = {
                "title": comp.title,
                "framing": comp.framing,
                "callouts": [{"id": c.id, "label": c.label, "value": c.value} for c in comp.callouts],
                "reasoning": [
                    {"id": s.id, "expression": s.expression, "explanation": s.explanation}
                    for s in comp.reasoning
                ],
                "result": comp.result.expression if comp.result else None,
                "takeaway": comp.takeaway,
            }
            report["composition_source"] = "explicit"
        else:
            report["composition_source"] = "derived-fallback"
            report["fallback_reason"] = "Gemini omitted composition (insufficient structured evidence)"
        errors = validate_lesson_spec(spec)
        report["validation"] = errors or "clean"
        try:
            family = select_family(spec)
        except Exception as e:
            family = f"REFUSED: {e}"
        report["family"] = family
        t_render = time.perf_counter()
        result = await render_v3_visual(spec)
        report["render_ms"] = round((time.perf_counter() - t_render) * 1000, 1)
        if result is None:
            report["render"] = "None (honest unavailable)"
        else:
            mode, payload = result
            raw_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
            report["render"] = f"mode={mode} bytes={len(raw_bytes)}"
            open(os.path.join(OUT, f"realinput_{name}.{ 'svg' if mode == 'svg' else 'png' }"), "wb").write(raw_bytes)
            if mode == "png":
                t_up = time.perf_counter()
                try:
                    from app.services.storage_service import upload_image

                    url = await asyncio.to_thread(upload_image, raw_bytes, {"title": f"gate3-{name}"})
                except Exception as e:
                    url = f"UPLOAD_ERROR: {type(e).__name__}"
                report["upload_ms"] = round((time.perf_counter() - t_up) * 1000, 1)
                report["upload_url"] = url or "skipped (no IMGBB key)"
            else:
                report["upload_ms"] = 0
                report["upload_url"] = "not attempted (no composed PNG)"
        # spec content quality signals (does it capture the right things?)
        report["equations_in_spec"] = [e.expression for e in det.equations][:4]
        report["points_in_spec"] = det.points[:3]
    except Exception as e:
        report["FAILED"] = f"{type(e).__name__}: {str(e)[:200]}"
    report["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return report


async def main():
    base = os.path.dirname(__file__)
    results = []
    for name, rel in IMAGES:
        print(f"--- {name} ---", flush=True)
        rep = await one(name, os.path.join(base, rel))
        print(json.dumps(rep, indent=1, ensure_ascii=True)[:1500], flush=True)
        results.append(rep)
    open(os.path.join(OUT, "gate3_report.json"), "w", encoding="utf-8").write(
        json.dumps(results, indent=1, ensure_ascii=False)
    )
    print("wrote gate3_report.json")


if __name__ == "__main__":
    asyncio.run(main())
