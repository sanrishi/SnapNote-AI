from app.models.schemas import StudyNotes


def uncertainty_sentence(symbol: str) -> str:
    return f"The meaning of {symbol} cannot be confirmed from this screenshot because the surrounding lecture context is missing."


def render_study_notes(notes: StudyNotes) -> str:
    lines: list[str] = []

    if notes.topic.title:
        prefix = "Probable topic" if notes.topic.is_probable else "Topic"
        lines.append(f"## {prefix}: {notes.topic.title}")

    if notes.simple_explanation:
        lines.append("\n## Simple Explanation")
        lines.append(notes.simple_explanation)

    if notes.formula_box:
        lines.append("\n## Formula Box")
        for f in notes.formula_box:
            line = f"`{f.formula}`"
            if f.explanation:
                line += f"\n{f.explanation}"
            if f.uncertain_symbols:
                line += "\n" + "\n".join(
                    f"*{uncertainty_sentence(s)}*" for s in f.uncertain_symbols
                )
            lines.append(line)

    if notes.study_notes:
        lines.append("\n## Study Notes")
        for note in notes.study_notes:
            if note.strip():
                lines.append(f"- {note.strip()}")

    if notes.diagram_interpretation.present:
        lines.append("\n## Diagram")
        for el in notes.diagram_interpretation.visible_elements:
            if el.strip():
                lines.append(f"- {el.strip()}")
        if notes.diagram_interpretation.likely_interpretation:
            lines.append("\nLikely interpretation:")
            for interp in notes.diagram_interpretation.likely_interpretation:
                if interp.strip():
                    lines.append(f"- {interp.strip()}")

    if notes.uncertainties:
        lines.append("\n## Unclear or Uncertain")
        for u in notes.uncertainties:
            if u.strip():
                lines.append(f"- {u.strip()}")

    if notes.key_takeaway:
        lines.append(f"\n**Key takeaway:** {notes.key_takeaway}")

    return "\n".join(lines).strip()
