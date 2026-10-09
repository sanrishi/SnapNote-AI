"""Mirror topic-label -> chapter concept map builder.

The mirror (third-party transcription) tags each question with its own topic
labels. This table maps every OBSERVED label to a systematic chapter concept
derived from the NTA syllabus units. Special cases are explicit and reviewed;
anything unmapped fails loudly instead of guessing.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTERS = os.path.join(HERE, "data", "jee-chapters.json")
OUT = os.path.join(HERE, "data", "mirror_topic_map.json")

# label -> chapter slug suffix (jee-<subject>-<suffix>-chapter); subject comes
# from the index record, so the map only resolves the chapter part.
SPECIAL = {
    # Physics: mirror splits, syllabus merges.
    "Ray Optics": "optics",
    "Wave Optics": "optics",
    "Oscillations": "oscillations-and-waves",
    "Waves": "oscillations-and-waves",
    # Chemistry: hydrocarbons live inside the organic-principles unit.
    "Hydrocarbons": "some-basic-principles-of-organic-chemistry",
    # Mathematics: mirror splits, syllabus merges.
    "Complex Numbers": "complex-numbers-and-quadratic-equations",
    "Quadratic Equations": "complex-numbers-and-quadratic-equations",
    "Statistics": "statistics-and-probability",
    "Probability": "statistics-and-probability",
    "Trigonometric Ratios and Equations": "trigonometry",
    "Inverse Trigonometric Functions": "trigonometry",
    "Straight Lines": "co-ordinate-geometry",
    "Circles": "co-ordinate-geometry",
    "Conic Sections": "co-ordinate-geometry",
    "Limits, Continuity and Differentiability": "limit-continuity-and-differentiability",
    "Differentiation and Applications of Derivatives": "limit-continuity-and-differentiability",
    # Syllabus-unit wording differs from the mirror label.
    "Binomial Theorem": "binomial-theorem-and-its-simple-applications",
    "Classification of Elements and Periodicity":
        "classification-of-elements-and-periodicity-in-properties",
    # Official syllabus PDF spells it "CALCULAS"; map to the same chapter.
    "Integral Calculus": "integral-calculas",
    # Official syllabus PDF spells it "DIFFRENTIAL"; map to the same chapter.
    "Differential Equations": "diffrential-equations",
    "Sequences and Series": "sequence-and-series",
}


def slugify(unit: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", unit.lower()).strip("-"))


def build(observed_labels: list[str]) -> dict[str, dict]:
    chapters = json.load(open(CHAPTERS, encoding="utf-8"))
    by_subject_slug = {(c["subject"], slugify(c["syllabus_unit"])): c for c in chapters}
    # Title-case display names also resolve (mirror labels are title-cased).
    everything = dict(by_subject_slug)
    for c in chapters:
        everything[(c["subject"], slugify(c["display_name"]))] = c
    mapping = {}
    for label in sorted(set(observed_labels)):
        # Subject is resolved per record at queue time; find candidate chapters.
        cands = [c for (s, slug), c in everything.items()
                 if slug == slugify(label) or SPECIAL.get(label) == slug]
        if not cands:
            raise SystemExit(f"unmapped mirror label (refusing to guess): {label!r}")
        if len({c["chapter_id"] for c in cands}) > 1:
            raise SystemExit(f"ambiguous mirror label: {label!r} -> {[c['chapter_id'] for c in cands]}")
        mapping[label] = {
            "chapter_id": cands[0]["chapter_id"],
            "subject": cands[0]["subject"],
            "via": "special-case" if label in SPECIAL else "exact-unit-match",
        }
    return mapping


def main() -> int:
    import sys
    labels = json.load(open(sys.argv[1], encoding="utf-8")) if len(sys.argv) > 1 else []
    mapping = build(labels)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=1)
    print(json.dumps({"labels": len(mapping), "out": OUT}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
