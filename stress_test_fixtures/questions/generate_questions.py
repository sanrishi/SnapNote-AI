"""Generate the five Hinglish-explanation acceptance fixtures.

The images are clean, typed practice questions (white background) so the
vision model can read every symbol without OCR noise. They are deliberately
NOT lecture slides: the feature under test is "explain the question first,
then the solution".

Run from the repo root:
    python stress_test_fixtures/questions/generate_questions.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent
WIDTH = 1180
PAD_X = 78
PAD_Y = 62
BODY_SIZE = 31
HEADER_SIZE = 18
LINE_GAP = 14
TEXT_COLOR = (24, 28, 38)
MUTED = (110, 118, 134)
ACCENT = (76, 74, 190)


def _font(size: int, *, symbols: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/ARIALUNI.TTF" if symbols else "",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render(filename: str, header: str, body: list[str]) -> None:
    body_font = _font(BODY_SIZE, symbols=True)
    header_font = _font(HEADER_SIZE)

    # Measure first so the canvas fits the content exactly.
    probe = Image.new("RGB", (WIDTH, 100), "white")
    probe_draw = ImageDraw.Draw(probe)
    rendered: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int], int]] = []
    for index, paragraph in enumerate(body):
        font = body_font
        color = TEXT_COLOR
        gap = LINE_GAP
        if paragraph.startswith("    "):
            # Indented equation lines (kept verbatim, no reflow).
            rendered.append((paragraph.strip(), font, ACCENT, gap))
            continue
        for line in _wrap(probe_draw, paragraph, font, WIDTH - 2 * PAD_X):
            rendered.append((line, font, color, gap))

    line_height = BODY_SIZE + 10
    height = PAD_Y + HEADER_SIZE + 34 + len(rendered) * line_height + PAD_Y

    image = Image.new("RGB", (WIDTH, height), "white")
    draw = ImageDraw.Draw(image)
    y = PAD_Y
    draw.text((PAD_X, y), header.upper(), font=header_font, fill=MUTED)
    y += HEADER_SIZE + 34
    draw.line((PAD_X, y - 18, WIDTH - PAD_X, y - 18), fill=(226, 230, 240), width=1)

    for text, font, color, gap in rendered:
        draw.text((PAD_X, y), text, font=font, fill=color)
        y += line_height + (gap if gap > LINE_GAP else 0)

    image.save(OUT_DIR / filename, format="PNG", optimize=True)
    print(f"wrote {filename} ({WIDTH}x{height})")


def main() -> None:
    _render(
        "q_math_log_inequality.png",
        "Practice question 1 — Mathematics",
        [
            "Determine the set of all real numbers x for which the following logarithmic inequality is satisfied.",
            "    log base (x + 3) of (x\u00b2 \u2212 x) < 1",
            "Express your answer as a union of intervals, and justify every restriction on x.",
        ],
    )
    _render(
        "q_chemistry_equilibrium.png",
        "Practice question 2 — Chemistry",
        [
            "At 500 K, the equilibrium constant Kc for the reaction",
            "    H\u2082(g) + I\u2082(g) \u21cc 2 HI(g)",
            "is 49.0. A sealed vessel is initially charged with 0.200 M H\u2082 and 0.200 M I\u2082 at 500 K.",
            "Calculate the equilibrium concentration of HI. State any assumption you use and give the answer with correct units.",
        ],
    )
    _render(
        "q_biology_bohr_effect.png",
        "Practice question 3 — Biology",
        [
            "When the CO\u2082 concentration in the blood increases, the oxygen dissociation curve of haemoglobin shifts to the right.",
            "Explain why this shift happens, using the relevant properties of haemoglobin.",
            "State one physiological advantage of this shift for a metabolically active tissue.",
        ],
    )
    _render(
        "q_physics_rolling_cylinder.png",
        "Practice question 4 — Physics",
        [
            "A solid cylinder of mass 2.0 kg and radius 0.20 m rolls without slipping down a plane inclined at \u03b8 = 30\u00b0.",
            "Calculate the acceleration of its centre of mass and the minimum coefficient of static friction \u03bcs required for rolling without slipping.",
            "Take g = 9.8 m/s\u00b2 and show the direction of the friction force.",
        ],
    )
    _render(
        "q_cs_bst_complexity.png",
        "Practice question 5 — Computer Science",
        [
            "A binary search tree (BST) stores n distinct keys. Searching for a key takes O(h) time, where h is the height of the tree.",
            "Explain why the bound is O(h), and describe the worst-case height of a BST with n nodes.",
            "When does a BST degrade to O(n) search, and how do balanced trees prevent that degradation?",
        ],
    )


if __name__ == "__main__":
    main()
