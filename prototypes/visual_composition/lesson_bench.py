"""Lesson-level bench: BEFORE (old render_deterministic_visual) vs AFTER
(render_visual_lesson production contract) on the SAME ground-truth specs.

Outputs lesson_before_{torque,argand}.svg + lesson_after_{torque,argand}.{pdf,png,svg}
plus timings. No production behavior changed (old path untouched).
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from ground_truth_specs import (
    argand_spec_from_ground_truth,
    argand_v3_spec_from_ground_truth,
    torque_spec_from_ground_truth,
    torque_v3_spec_from_ground_truth,
)
from app.utils.visual_renderer import render_deterministic_visual
from app.utils.visual_lesson import LessonValidationError, render_visual_lesson, select_family, validate_lesson_spec

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def bench():
    print("=== Lesson Contract — BEFORE vs AFTER (same ground truth) ===")
    for name, builder in [("torque", torque_spec_from_ground_truth), ("argand", argand_spec_from_ground_truth)]:
        spec = builder()
        errors = validate_lesson_spec(spec)
        print(f"{name}: validation errors={errors or 'none'}, family={select_family(spec)}")
        # BEFORE: old production path
        t0 = time.perf_counter()
        before = render_deterministic_visual(spec.deterministic)
        ms_before = (time.perf_counter() - t0) * 1000
        open(os.path.join(OUT, f"lesson_before_{name}.svg"), "w", encoding="utf-8").write(before)
        print(f"  BEFORE {name}: {len(before)} chars, {ms_before:.1f}ms")
        # AFTER: new production contract
        try:
            lesson = render_visual_lesson(spec, out_format="pdf")
            open(os.path.join(OUT, f"lesson_after_{name}.{lesson.out_format}"), "wb").write(lesson.data)
            print(
                f"  AFTER {name}: {len(lesson.data)} bytes {lesson.out_format}, "
                f"total={lesson.ms_total:.1f}ms (hero={lesson.ms_hero:.1f}, compose={lesson.ms_compose:.1f}), "
                f"fallback={lesson.fallback_used}, warnings={lesson.warnings}"
            )
        except LessonValidationError as e:
            print(f"  AFTER {name} REFUSED: {e}")
        # AFTER png for viewing
        try:
            lesson_png = render_visual_lesson(spec, out_format="png")
            open(os.path.join(OUT, f"lesson_after_{name}.{lesson_png.out_format}"), "wb").write(lesson_png.data)
            print(f"  AFTER PNG {name}: {len(lesson_png.data)} bytes, total={lesson_png.ms_total:.1f}ms")
        except LessonValidationError as e:
            print(f"  AFTER PNG {name} REFUSED: {e}")

    print("=== Lesson Contract — V3 explicit composition (same ground truth) ===")
    for name, builder in [("torque", torque_v3_spec_from_ground_truth), ("argand", argand_v3_spec_from_ground_truth)]:
        spec = builder()
        errors = validate_lesson_spec(spec)
        assert spec.deterministic.composition is not None
        print(f"V3 {name}: validation errors={errors or 'none'}, family={select_family(spec)}")
        assert not errors, f"v3 {name} must validate clean"
        try:
            lesson = render_visual_lesson(spec, out_format="pdf")
            open(os.path.join(OUT, f"lesson_v3_{name}.{lesson.out_format}"), "wb").write(lesson.data)
            print(
                f"  V3 {name}: {len(lesson.data)} bytes {lesson.out_format}, "
                f"total={lesson.ms_total:.1f}ms (hero={lesson.ms_hero:.1f}, compose={lesson.ms_compose:.1f}), "
                f"fallback={lesson.fallback_used}, warnings={lesson.warnings}"
            )
        except LessonValidationError as e:
            print(f"  V3 {name} REFUSED: {e}")
        try:
            lesson_png = render_visual_lesson(spec, out_format="png")
            open(os.path.join(OUT, f"lesson_v3_{name}.{lesson_png.out_format}"), "wb").write(lesson_png.data)
            print(f"  V3 PNG {name}: {len(lesson_png.data)} bytes, total={lesson_png.ms_total:.1f}ms")
        except LessonValidationError as e:
            print(f"  V3 PNG {name} REFUSED: {e}")

    print("=== Route contract — render_v3_visual (exact function the route calls) ===")
    import asyncio as _asyncio

    from app.utils.visual_lesson import render_v3_visual, should_use_v3

    for name, builder in [("torque", torque_v3_spec_from_ground_truth), ("argand", argand_v3_spec_from_ground_truth)]:
        spec = builder()
        print(f"ROUTE {name}: should_use_v3(flag off default)={should_use_v3(spec)}")
        t0 = time.perf_counter()
        result = _asyncio.run(render_v3_visual(spec))
        ms = (time.perf_counter() - t0) * 1000
        if result is None:
            print(f"  ROUTE {name}: None (honest unavailable)")
        else:
            mode, payload = result
            size = len(payload) if isinstance(payload, (bytes, bytearray)) else len(payload.encode("utf-8"))
            open(os.path.join(OUT, f"lesson_route_{name}.{ 'svg' if mode == 'svg' else 'png' }"), "wb").write(
                payload.encode("utf-8") if isinstance(payload, str) else payload
            )
            print(f"  ROUTE {name}: mode={mode}, {size} bytes, {ms:.1f}ms")


if __name__ == "__main__":
    bench()
