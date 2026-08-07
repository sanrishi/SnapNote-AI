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

_SUPERSCRIPTS = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
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
        return name
    text = _CMD_RE.sub(repl, text)

    text = re.sub(r"\^{([^{}]*)}", lambda m: _replace_group(m, "^"), text)
    text = re.sub(r"_{([^{}]*)}", lambda m: _replace_group(m, "_"), text)
    text = re.sub(r"\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^([0-9a-zA-Z])", lambda m: _translate_script(m.group(1), "^"), text)
    text = re.sub(r"_([0-9a-zA-Z])", lambda m: _translate_script(m.group(1), "_"), text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"[{}]", "", text)
    return text
