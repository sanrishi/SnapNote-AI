import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_reader_lock = threading.Lock()
OCRResult = list[tuple[list[list[float]], str, float]]
ROW_TOLERANCE = 15.0
CONF_THRESHOLD = 0.3

_reader = None


def _get_reader():
    global _reader
    if _reader is not None:
        return _reader
    try:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader initialized")
    except ImportError:
        logger.warning("EasyOCR not available — will escalate all extraction to Gemini")
        _reader = None
    return _reader


def read_raw(image_np: Any) -> OCRResult:
    reader = _get_reader()
    if reader is None:
        return []
    with _reader_lock:
        return reader.readtext(image_np)


def ocr_available() -> bool:
    """True when the EasyOCR reader is importable (may lazily initialize)."""
    return _get_reader() is not None


def _is_english_like(word: str) -> bool:
    vowels = set("aeiou")
    chars = word.lower()
    has_vowel = any(c in vowels for c in chars)
    consonant_run = 0
    for c in chars:
        if c.isalpha() and c not in vowels:
            consonant_run += 1
            if consonant_run >= 5:
                return False
        else:
            consonant_run = 0
    return has_vowel and len(chars) >= 3


def low_quality_result(ocr_results: OCRResult) -> bool:
    if not ocr_results:
        return True

    tokens = [text.strip() for _, text, _ in ocr_results]

    total_chars = sum(len(t) for t in tokens)
    if total_chars < 10:
        return True

    confs = [conf for _, _, conf in ocr_results]
    avg_conf = sum(confs) / len(confs)
    if avg_conf < 0.4:
        return True

    low_conf_ratio = sum(1 for c in confs if c < 0.3) / len(confs)
    if low_conf_ratio > 0.3:
        return True

    all_text = " ".join(tokens)

    # Garbled text: high ratio of non-alphanumeric chars
    alpha_chars = sum(c.isalpha() or c.isdigit() or c.isspace() for c in all_text)
    if len(all_text) > 0 and alpha_chars / len(all_text) < 0.5:
        return True

    # Structural: high ratio of single-char tokens (OCR fragmentation)
    single_char = sum(1 for t in tokens if len(t) == 1 and t.isalnum())
    if len(tokens) > 0 and single_char / len(tokens) > 0.3:
        return True

    # English-likeness: check tokens that look like real English words
    alpha_tokens = [t.lower().strip(".,;:!?)}]>\"'-") for t in tokens]
    alpha_tokens = [t for t in alpha_tokens if len(t) >= 3 and t.isalpha()]
    if alpha_tokens:
        eng_like = sum(1 for t in alpha_tokens if _is_english_like(t))
        if eng_like / len(alpha_tokens) < 0.4:
            return True

    # Structural: average alpha token length — garbled OCR fragments are short
    if alpha_tokens:
        avg_len = sum(len(t) for t in alpha_tokens) / len(alpha_tokens)
        if avg_len < 4.5:
            return True

    # Structural: high ratio of tokens ending with closing symbols
    end_sym = sum(1 for t in tokens if t and t[-1] in "})]>")
    if len(tokens) > 0 and end_sym / len(tokens) > 0.15:
        return True

    # Structural: high ratio of tokens containing special symbols
    symbol_tokens = sum(1 for t in tokens if any(c in t for c in "}{()[]=+/\\|<>*^"))
    if len(tokens) > 0 and symbol_tokens / len(tokens) > 0.3:
        return True

    # Mixed digit-letter ratio: garbled formulas mix digits and letters oddly
    mixed = sum(1 for t in tokens if any(c.isdigit() for c in t) and any(c.isalpha() for c in t))
    if len(tokens) > 0 and mixed / len(tokens) > 0.2:
        return True

    return False


def raw_to_lines(ocr_results: OCRResult) -> list[str]:
    lines: list[tuple[float, str]] = []
    for bbox, text, conf in ocr_results:
        if conf > CONF_THRESHOLD:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            lines.append((y_center, text.strip()))
    lines.sort(key=lambda x: x[0])
    return [text for _, text in lines]


def extract_text(image_np: Any) -> list[str]:
    results = read_raw(image_np)
    return raw_to_lines(results)


def _cluster_columns(ocr_results: OCRResult, tolerance: float = 20.0) -> list[list[float]]:
    filtered = [r for r in ocr_results if r[2] > CONF_THRESHOLD]
    if not filtered:
        return []
    x_centers = sorted([(bbox[0][0] + bbox[2][0]) / 2 for bbox, _, _ in filtered])
    clusters: list[list[float]] = [[x_centers[0]]]
    for x in x_centers[1:]:
        if abs(x - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return clusters


def is_table_layout(ocr_results: OCRResult) -> bool:
    filtered = [r for r in ocr_results if r[2] > CONF_THRESHOLD]
    if len(filtered) < 4:
        return False

    col_clusters = _cluster_columns(ocr_results)
    if len(col_clusters) < 2:
        return False

    col_centers = [sum(c) / len(c) for c in col_clusters]
    gaps = [col_centers[i + 1] - col_centers[i] for i in range(len(col_centers) - 1)]
    if not gaps:
        return False

    mean_gap = sum(gaps) / len(gaps)
    aligned = sum(1 for g in gaps if abs(g - mean_gap) < mean_gap * 0.3)
    return (aligned / len(gaps)) > 0.5


def format_as_markdown(text_lines: list[str]) -> str:
    if not text_lines:
        return ""

    lines: list[str] = []
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        lines.append(line)

    return "\n\n".join(lines)


def _classify_line(text: str) -> str:
    math_symbols = "ωθφαπλμ∑∫=±∞×÷→≤≥√Σ∏≠≈Δγβδ°²³₁₂₀"
    sym_count = sum(1 for c in text if c in math_symbols or c in "+-*/^=")
    if sym_count >= 2 and len(text) <= 60:
        return "formula"
    if len(text.split()) == 1 and len(text) <= 12:
        return "label"
    return "text"


def format_structured_text(text_lines: list[str]) -> str:
    if not text_lines:
        return ""

    cleaned = [ln.strip() for ln in text_lines if ln.strip()]
    if not cleaned:
        return ""

    formulas: list[str] = []
    labels: list[str] = []
    statements: list[str] = []
    seen: set[str] = set()

    for line in cleaned:
        if line in seen:
            continue
        seen.add(line)
        cls = _classify_line(line)
        if cls == "formula":
            formulas.append(line)
        elif cls == "label":
            labels.append(line)
        else:
            statements.append(line)

    parts: list[str] = []
    if formulas:
        parts.append("## Visible Formulas\n\n" + "\n\n".join(f"`{f}`" for f in formulas))
    if labels:
        parts.append("## Visible Labels\n\n" + "\n".join(f"- {lbl}" for lbl in labels))
    if statements:
        parts.append("## Extracted Text\n\n" + "\n\n".join(statements))

    return "\n\n".join(parts)


def _cluster_rows(
    ocr_results: OCRResult,
) -> list[list[tuple[float, str]]]:
    filtered = [(bbox, text, conf) for bbox, text, conf in ocr_results if conf > CONF_THRESHOLD]
    if not filtered:
        return []

    indexed: list[tuple[float, float, str]] = []
    for bbox, text, _ in filtered:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        indexed.append((y_center, x_center, text.strip()))

    indexed.sort(key=lambda x: x[0])

    clusters: list[list[tuple[float, float, str]]] = []
    current_cluster: list[tuple[float, float, str]] = [indexed[0]]

    for i in range(1, len(indexed)):
        prev_y = indexed[i - 1][0]
        curr_y = indexed[i][0]
        if abs(curr_y - prev_y) <= ROW_TOLERANCE:
            current_cluster.append(indexed[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [indexed[i]]
    clusters.append(current_cluster)

    result: list[list[tuple[float, str]]] = []
    for cluster in clusters:
        sorted_cells = sorted(cluster, key=lambda x: x[1])
        result.append([(x, t) for _, x, t in sorted_cells])
    return result


def format_as_table(ocr_results: OCRResult) -> str:
    rows = _cluster_rows(ocr_results)
    if not rows:
        return ""

    md_rows: list[str] = []
    for cells in rows:
        md_rows.append("| " + " | ".join(c[1] for c in cells) + " |")

    if not md_rows:
        return ""

    header = md_rows[0]
    col_count = header.count("|") - 1
    separator = "| " + " | ".join(["---"] * col_count) + " |"

    result = [header, separator] + md_rows[1:]
    return "\n".join(result)
