"""Gate 5: real messy screenshots through the REAL understanding pipeline.

For each image: enhance -> extract_study_notes -> build_visual_spec (real
Gemini, same calls the route makes) -> validate_lesson_spec ->
render_v3_visual. Records spec quality + validation + render outcome.
No credits touched (service functions directly, no routes). No Typst on
Windows -> composed PNGs are proven on Ubuntu; here we verify the spec
generation + validation + honest fallback path.
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from app.services.preprocessor import enhance_for_vision
from app.services.vision_service import build_visual_spec, extract_study_notes
from app.utils.visual_lesson import render_v3_visual, select_family, validate_lesson_spec

IMAGES = [
    ("physics-torque-board", "../../website/samples/physics-demo.jpg"),
    ("math-complex-board", "../../website/samples/math-demo.jpg"),
    ("engineering-third", "../../website/samples/engineering-demo.jpg"),
]

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
        spec = await build_visual_spec(enhanced, notes)
        report["spec_ms"] = round((time.perf_counter() - t_spec) * 1000, 1)
        report["render_mode"] = str(spec.render_mode)
        det = spec.deterministic
        report["spec_title"] = det.title
        report["has_scene"] = det.scene is not None
        report["scene_kind"] = det.scene.scene_kind if det.scene else None
        report["has_composition"] = det.composition is not None
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
            size = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
            report["render"] = f"mode={mode} bytes={size}"
            open(os.path.join(OUT, f"realinput_{name}.{ 'svg' if mode == 'svg' else 'png' }"), "wb").write(
                payload.encode("utf-8") if isinstance(payload, str) else payload
            )
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
        print(json.dumps(rep, indent=1, ensure_ascii=False)[:1500], flush=True)
        results.append(rep)
    open(os.path.join(OUT, "real_input_report.json"), "w", encoding="utf-8").write(
        json.dumps(results, indent=1, ensure_ascii=False)
    )
    print("wrote real_input_report.json")


if __name__ == "__main__":
    asyncio.run(main())
