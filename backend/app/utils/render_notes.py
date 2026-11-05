from app.models.schemas import ExplanationLanguage, JEEConceptContext, StudyNotes
from app.utils.svg_safe import sanitize_svg, svg_data_uri


def uncertainty_sentence(symbol: str) -> str:
    return f"The meaning of {symbol} cannot be confirmed from this screenshot because the surrounding lecture context is missing."


def _confidence_note(confidence: str) -> str:
    if confidence == "context_needed":
        return "*Context needed: the formula is readable, but the surrounding lecture context is missing.*"
    if confidence == "possible_extraction_issue":
        return "*Possible extraction issue: the mathematical symbols are ambiguous. See Verify Before Studying below.*"
    return ""


def _render_explanation(notes: StudyNotes) -> list[str]:
    """Render the question-first Hinglish/English explanation as markdown.

    The formal exam answer stays a separate English block even in Hinglish
    mode: that is the text a student writes in the exam. The source question
    is printed before any reasoning so it is never silently rewritten.
    """
    explanation = notes.explanation
    if explanation is None:
        return []

    hinglish = explanation.language == ExplanationLanguage.HINGLISH
    labels = {
        "meaning": "Question ka matlab" if hinglish else "What the Question Means",
        "asked": (
            "Question humse kya maang raha hai?"
            if hinglish
            else "What the Question Asks"
        ),
        "concept": "Concept samjho" if hinglish else "The Concept",
        "steps": "Step-by-step solution",
        "exam": (
            "Exam mein kaise likhna hai? (formal English answer)"
            if hinglish
            else "How to Write It in the Exam (formal answer)"
        ),
        "recap": "Quick recap" if hinglish else "Quick Recap",
        "review": (
            "Solution needs a quick check — question ya symbol ambiguous ho sakta hai. "
            "Verify against the original screenshot."
            if hinglish
            else "Solution needs a quick check — the question or a symbol may be ambiguous. "
            "Verify against the original screenshot."
        ),
    }

    lines: list[str] = []
    if explanation.question_text.strip():
        lines.append("\n## 📖 Original Question")
        lines.append(explanation.question_text.strip())
    if explanation.question_meaning.strip():
        lines.append(f"\n## 🧠 {labels['meaning']}")
        lines.append(explanation.question_meaning.strip())
    if explanation.what_is_asked:
        lines.append(f"\n## 🎯 {labels['asked']}")
        for item in explanation.what_is_asked:
            if item.strip():
                lines.append(f"- {item.strip()}")
    if explanation.concept:
        lines.append(f"\n## 📚 {labels['concept']}")
        for item in explanation.concept:
            if item.strip():
                lines.append(item.strip())
    if explanation.steps:
        lines.append(f"\n## 🪜 {labels['steps']}")
        for index, step in enumerate(explanation.steps, start=1):
            title = step.title.strip() or f"Step {index}"
            lines.append(f"\n### {index}. {title}")
            if step.equation.strip():
                lines.append(f"`{step.equation.strip()}`")
            if step.explanation.strip():
                lines.append(step.explanation.strip())
    if explanation.exam_answer.strip():
        lines.append(f"\n## ✍️ {labels['exam']}")
        lines.append(explanation.exam_answer.strip())
    if explanation.verified:
        lines.append("*✓ Final result verified by SnapNote's exact solver.*")
    if explanation.verification_note.strip():
        lines.append(f"*{explanation.verification_note.strip()}*")
    if explanation.recap:
        lines.append(f"\n## ⏱️ {labels['recap']}")
        for item in explanation.recap:
            if item.strip():
                lines.append(f"- {item.strip()}")
    if explanation.confidence == "review":
        lines.append(f"\n*{labels['review']}*")
    return lines


def render_study_notes(notes: StudyNotes) -> str:
    lines: list[str] = []

    if notes.topic.title:
        title = notes.topic.title.upper()
        if notes.topic.is_probable:
            lines.append(f"# {title}\n*Topic inferred from screenshot*")
        else:
            lines.append(f"# {title}")

    lines.extend(_render_explanation(notes))
    if notes.explanation is not None:
        lines.append("\n---\n## 📝 Revision Sheet")

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
            if notes.diagram.best_effort:
                lines.append("*Best-effort reconstruction — this diagram type isn't fully supported yet; verify it against the original screenshot.*")
            else:
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

    if notes.analogy:
        lines.append("\n## 💡 Easy Analogy")
        lines.append(notes.analogy)

    return "\n".join(lines).strip()


def render_jee_section(ctx: JEEConceptContext | None) -> str:
    """Render the student-facing JEE Context markdown section.

    Only matched or ambiguous concepts render anything — unresolved input
    adds no JEE content, so generic screenshots stay JEE-free.
    """
    if ctx is None or ctx.match_status not in ("matched", "ambiguous"):
        return ""
    lines: list[str] = ["\n## 🎓 JEE Context"]
    if ctx.match_status == "ambiguous":
        lines.append("**JEE topic identified — chapter not yet certain**")
        lines.append(
            "*This lecture may belong to any of these chapters. Related stored "
            "questions from each are shown below; exact-subtopic relevance has "
            "not been confirmed.*"
        )
        for cand in ctx.candidates:
            lines.append(f"- {cand.name}")
        for item in ctx.evidence:
            lines.append(f"*{item}*")
        for group in ctx.candidate_results:
            lines.append(f"\n**Possible: {group.name}** ({group.pyq_count} related record(s))")
            lines.append(f"*{group.coverage_note}*")
            for pyq in group.pyqs:
                label = f"{pyq.exam} {pyq.year}"
                if pyq.session:
                    label += f" {pyq.session}"
                if pyq.shift:
                    label += f" {pyq.shift}"
                answer = pyq.answer_key if pyq.answer_status == "confirmed" else "answer unconfirmed"
                provenance = "official NTA paper" if pyq.source_class == "OFFICIAL" else "third-party transcription"
                lines.append(
                    f"- {label}, Q{pyq.paper_question_number} ({pyq.question_type}): "
                    f"{pyq.question_summary or 'see source'} — {answer} [{provenance}]({pyq.source_url})"
                )
            if group.pyqs_truncated:
                lines.append(f"*Showing {len(group.pyqs)} of {group.pyq_count} related records.*")
        return "\n".join(lines)
    lines.append(f"**{ctx.canonical_name}** ({ctx.subject} · {ctx.chapter})")
    if ctx.syllabus is not None:
        lines.append(f"Syllabus: {ctx.syllabus.subtopic}")
    for item in ctx.evidence:
        lines.append(f"*{item}*")
    if ctx.pyqs:
        lines.append(f"\nVerified PYQs ({ctx.pyq_count}):")
        for pyq in ctx.pyqs:
            label = f"{pyq.exam} {pyq.year}"
            if pyq.session:
                label += f" {pyq.session}"
            if pyq.shift:
                label += f" {pyq.shift}"
            answer = pyq.answer_key if pyq.answer_status == "confirmed" else "answer unconfirmed"
            provenance = "official NTA paper" if pyq.source_class == "OFFICIAL" else "third-party transcription"
            lines.append(
                f"- {label}, Q{pyq.paper_question_number} ({pyq.question_type}): "
                f"{pyq.question_summary or 'see source'} — {answer} [{provenance}]({pyq.source_url})"
            )
        if ctx.pyqs_truncated:
            lines.append(f"*Showing {len(ctx.pyqs)} of {ctx.pyq_count} verified PYQs.*")
    for pattern in ctx.patterns:
        lines.append(f"\n**Pattern: {pattern.name}** ({pattern.sub_concept})")
        lines.append(pattern.method)
        lines.append(f"Examples: {', '.join(pattern.question_ids)}")
        lines.append(f"*{pattern.coverage_note}*")
    lines.append(f"\n*{ctx.coverage_note}*")
    return "\n".join(lines)
