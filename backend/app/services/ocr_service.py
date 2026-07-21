import os
import threading
from functools import lru_cache
from typing import Any

os.environ["EASYOCR_VERBOSE"] = "0"

import easyocr

_reader_lock = threading.Lock()
OCRResult = list[tuple[list[list[float]], str, float]]
ROW_TOLERANCE = 15.0
CONF_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def _get_reader() -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def read_raw(image_np: Any) -> OCRResult:
    reader = _get_reader()
    with _reader_lock:
        return reader.readtext(image_np)


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
