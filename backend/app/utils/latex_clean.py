import re


_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ",
    "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}

_SYMBOL = {
    "times": "×", "cdot": "·", "ast": "*", "pm": "±", "mp": "∓",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
    "approx": "≈", "propto": "∝", "rightarrow": "→", "to": "→",
    "implies": "→", "Leftarrow": "⇐", "Rightarrow": "⇒", "leftarrow": "←",
    "infty": "∞", "sum": "Σ", "int": "∫", "partial": "∂", "nabla": "∇",
    "forall": "∀", "exists": "∃", "in": "∈", "notin": "∉",
    "subset": "⊂", "supset": "⊃", "cup": "∪", "cap": "∩",
    "sqrt": "√", "sin": "sin", "cos": "cos", "tan": "tan",
    "log": "log", "ln": "ln", "exp": "exp", "det": "det",
    "lim": "lim", "max": "max", "min": "min", "mod": "mod",
    "qquad": "  ", "quad": "  ", ",": " ",
}

_TEXT_CMDS = {"text", "mathrm", "textrm", "operatorname", "mbox", "rm", "bf", "it", "em", "bm"}

# Superscript letters were individually render-verified through the real
# Typst pipeline (glyph probe PNG: no tofu, legible). f/g/q/v/z are
# deliberately absent — unverified coverage, so ^f etc. stay ASCII rather
# than risk tofu boxes. Partial groups never translate (all-or-nothing).
_SUPERSCRIPTS = str.maketrans(
    "0123456789+-=()niabcdehjklmoprstuwxy",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉʰʲᵏˡᵐᵒᵖʳˢᵗᵘʷˣʸ",
)
_SUBSCRIPTS = str.maketrans("0123456789+-=()ae", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑ")

_CMD_RE = re.compile(r"\\([a-zA-Z]+)")
_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_HAT_RE = re.compile(r"\\hat\s*\{([^{}]*)\}")
_VEC_RE = re.compile(r"\\vec\s*\{([^{}]*)\}")
_OVERLINE_RE = re.compile(r"\\(?:bar|overline)\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_LEFT_RE = re.compile(r"\\(?:left|right)")
_BRACE_CMD_RE = re.compile(r"\\([a-zA-Z]+)\s*\{([^{}]*)\}")


def _translate_script(text: str, script_type: str) -> str:
    if script_type == "^":
        table = _SUPERSCRIPTS
    else:
        table = _SUBSCRIPTS
    digits = text.translate(table)
    if any(ord(c) > 127 for c in digits):
        return digits
    return f"{script_type}{text}"


def _replace_group(match: re.Match, script_type: str) -> str:
    return _translate_script(match.group(1).strip(), script_type)


def _clean_braced_commands(text: str) -> str:
    def repl(match: re.Match) -> str:
        name, body = match.group(1), match.group(2)
        if name in _TEXT_CMDS:
            return body
        if name in _GREEK:
            return _GREEK[name] + body
        if name in _SYMBOL:
            return _SYMBOL[name] + body
        return body
    return _BRACE_CMD_RE.sub(repl, text)


_ASCII_MATH_RES = [
    (re.compile(r"<->"), "\u2194"),
    (re.compile(r"(?<!-)->(?!>)"), "\u2192"),
    (re.compile(r"(?<!<)<-(?!-)"), "\u2190"),
    (re.compile(r"<="), "\u2264"),
    (re.compile(r">="), "\u2265"),
    (re.compile(r"!="), "\u2260"),
    (re.compile(r"\bsqrt\s*\("), "\u221a("),
    (re.compile(r"\bcbrt\s*\("), "\u221b("),
    (re.compile(r"\bSum\b"), "\u03a3"),
]

# Bare-word Greek: exactly the LaTeX converter's vocabulary minus the
# backslash (theta, lambda, mu, ...). Letter-boundary guarded so "spin",
# "pie", "menu" are untouched ("alpha-beta" keeps its hyphen: "-" is not a
# letter, so each side still converts — correct, both are Greek there).
_BARE_GREEK_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted(_GREEK, key=len, reverse=True)) + r")(?![A-Za-z])"
)


def normalize_ascii_math(text: str) -> str:
    """Normalize unambiguous ASCII math spellings to Unicode.

    MATH SLOTS ONLY (expressions, relations, curve labels, callout values) —
    never prose (titles, captions, explanations) and never code (curve exprs
    fed to the safe evaluator, which needs ASCII "sqrt("). Each pattern is
    an explicit allowlist entry with a regression test; this is not a
    general find/replace. Things it deliberately does NOT touch: single
    Latin letters that could be variables ("u", "l" stay — indistinguishable
    from μ/λ without context), parenthesized exponents ("e^(-z^2/2)" —
    partial superscripting would look worse), English phrasing inside
    expressions ("(from 0 to 2pi)" — needs understanding, not rewriting).
    """
    if not text:
        return text
    for pattern, replacement in _ASCII_MATH_RES:
        text = pattern.sub(replacement, text)
    text = _BARE_GREEK_RE.sub(lambda m: _GREEK[m.group(1)], text)
    return text


def normalize_spec_math(spec: object) -> object:
    """Apply normalize_ascii_math to math-designated spec slots, in place.

    Covered: equation expressions, force/flow relation expressions,
    composition reasoning/result expressions, callout labels+values, plot
    curve labels. Deliberately NOT covered: titles, captions, framings,
    takeaways, explanations, meanings (prose), node/connector labels
    (natural language), and curve EXPRS (evaluator code — "sqrt(" must stay
    ASCII there). Returns the same object for chaining.
    """
    det = getattr(spec, "deterministic", None)
    if det is None:
        return spec
    for eq in getattr(det, "equations", None) or []:
        if getattr(eq, "expression", None):
            eq.expression = normalize_ascii_math(eq.expression)
    scene = getattr(det, "scene", None)
    if scene is not None:
        for payload_name in ("force", "flow"):
            payload = getattr(scene, payload_name, None)
            rel = getattr(payload, "relation", None) if payload is not None else None
            if rel is not None and getattr(rel, "expression", None):
                rel.expression = normalize_ascii_math(rel.expression)
        plot = getattr(scene, "plot", None)
        for curve in getattr(plot, "curves", None) or []:
            if getattr(curve, "label", None):
                curve.label = normalize_ascii_math(curve.label)
    comp = getattr(det, "composition", None)
    if comp is not None:
        for step in getattr(comp, "reasoning", None) or []:
            if getattr(step, "expression", None):
                step.expression = normalize_ascii_math(step.expression)
        result = getattr(comp, "result", None)
        if result is not None and getattr(result, "expression", None):
            result.expression = normalize_ascii_math(result.expression)
        for callout in getattr(comp, "callouts", None) or []:
            if getattr(callout, "label", None):
                callout.label = normalize_ascii_math(callout.label)
            if getattr(callout, "value", None):
                callout.value = normalize_ascii_math(callout.value)
    return spec


def latex_to_unicode(text: str) -> str:
    if not text:
        return text
    text = text.replace(r"\ ", " ")
    text = _FRAC_RE.sub(lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", text)
    text = _HAT_RE.sub(lambda m: f"{_GREEK.get(m.group(1).strip(), m.group(1).strip())}\u0302", text)
    text = _VEC_RE.sub(lambda m: f"{_GREEK.get(m.group(1).strip(), m.group(1).strip())}\u20D7", text)
    text = _OVERLINE_RE.sub(lambda m: f"{_GREEK.get(m.group(1).strip(), m.group(1).strip())}\u0304", text)
    text = _SQRT_RE.sub(lambda m: f"\u221a({m.group(1).strip()})", text)
    text = _LEFT_RE.sub("", text)
    text = _clean_braced_commands(text)

    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name in _GREEK:
            return _GREEK[name]
        if name in _SYMBOL:
            return _SYMBOL[name]
        # Unknown command (e.g. a Gemini misspelling like \integ): keep it
        # verbatim instead of silently eating the backslash — visible and
        # debuggable beats silent mangling ("integ" for "∫").
        return "\\" + name
    text = _CMD_RE.sub(repl, text)

    text = re.sub(r"\^{([^{}]*)}", lambda m: _replace_group(m, "^"), text)
    text = re.sub(r"_{([^{}]*)}", lambda m: _replace_group(m, "_"), text)
    text = re.sub(r"\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^([0-9a-zA-Z])", lambda m: _translate_script(m.group(1), "^"), text)
    text = re.sub(r"_([0-9a-zA-Z])", lambda m: _translate_script(m.group(1), "_"), text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    # NOTE: no blanket backslash strip — unknown \commands are preserved
    # verbatim by repl() above, and stripping here would silently mangle them.
    text = re.sub(r"[{}]", "", text)
    return text
