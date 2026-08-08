from app.models.schemas import StudyNotes
from app.utils.svg_safe import sanitize_svg, svg_data_uri


def uncertainty_sentence(symbol: str) -> str:
    return f"The meaning of {symbol} cannot be confirmed from this screenshot because the surrounding lecture context is missing."


def _confidence_note(confidence: str) -> str:
    if confidence == "context_needed":
        return "*Context needed: the formula is readable, but the surrounding lecture context is missing.*"
    if confidence == "possible_extraction_issue":
        return "*Possible extraction issue: the mathematical symbols are ambiguous. See Verify Before Studying below.*"
    return ""


def render_study_notes(notes: StudyNotes) -> str:
    lines: list[str] = []

    if notes.topic.title:
        title = notes.topic.title.upper()
        if notes.topic.is_probable:
            lines.append(f"# {title}\n*Topic inferred from screenshot*")
        else:
            lines.append(f"# {title}")

    if notes.what_you_should_remember:
        lines.append("\n## 🎯 What You Should Remember")
        lines.append(notes.what_you_should_remember)

    if notes.key_formulas:
        lines.append("\n## 📦 Key Formulas")
        for f in notes.key_formulas:
            line = f"`{f.formula}`"
            if f.explanation:
                line += f"\n{f.explanation}"
            for sym in f.uncertain_symbols:
                line += f"\n*{uncertainty_sentence(sym)}*"
            note = _confidence_note(f.confidence)
            if note:
                line += f"\n{note}"
            lines.append(line)

    if notes.understand_it:
        lines.append("\n## 🧠 Understand It")
        for para in notes.understand_it:
            if para.strip():
                lines.append(para.strip())

    if notes.common_mistakes:
        lines.append("\n## ⚠️ Common Mistakes")
        for m in notes.common_mistakes:
            if m.strip():
                lines.append(f"- {m.strip()}")

    if notes.thirty_second_revision:
        lines.append("\n## ⏱️ 30-Second Revision")
        for b in notes.thirty_second_revision:
            if b.strip():
                lines.append(f"- {b.strip()}")

    if notes.visual_context.present and notes.visual_context.summary:
        lines.append("\n## 🔎 Visual Context")
        lines.append(notes.visual_context.summary)

    if notes.diagram.present and notes.diagram.svg:
        svg = sanitize_svg(notes.diagram.svg)
        if svg:
            lines.append("\n## 📐 Diagram")
            lines.append(f"![Clean diagram]({svg_data_uri(svg)})")
            lines.append("*Rebuilt from your screenshot as a clean diagram.*")

    if notes.verify_before_studying:
        lines.append("\n## 🛡️ Verify Before Studying")
        lines.append(
            "Some symbols or equations may be ambiguous because the handwriting or image is unclear. "
            "Verify these against the original lecture before memorizing them:"
        )
        for item in notes.verify_before_studying:
            if item.strip():
                lines.append(f"- {item.strip()}")

    if notes.uncertainties:
        lines.append("\n## ⚠️ Unclear")
        for u in notes.uncertainties:
            if u.strip():
                lines.append(f"- {u.strip()}")

    if notes.analogy:
        lines.append("\n## 💡 Easy Analogy")
        lines.append(notes.analogy)

    return "\n".join(lines).strip()
