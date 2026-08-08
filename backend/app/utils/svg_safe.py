import re
from xml.etree import ElementTree

ALLOWED_TAGS = {
    "svg", "g", "defs", "marker", "path", "rect", "circle", "ellipse",
    "line", "polyline", "polygon", "text", "tspan", "title", "desc",
    "linearGradient", "radialGradient", "stop", "use", "symbol",
}

ALLOWED_ATTRS = {
    "xmlns", "viewBox", "width", "height", "fill", "stroke", "stroke-width",
    "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "opacity",
    "font-size", "font-family", "font-weight", "text-anchor", "x", "y",
    "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "points",
    "transform", "d", "id", "offset", "stop-color", "stop-opacity",
    "dx", "dy", "href", "dominant-baseline",
}

DANGEROUS = re.compile(
    r"<script|script>|onerror\s*=|onload\s*=|onclick\s*=|on[A-Za-z]+\s*=|javascript:|expression\(|foreignObject|embed|iframe",
    re.IGNORECASE,
)


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _clean_tree(elem: ElementTree.Element) -> ElementTree.Element | None:
    """Return a sanitized copy of elem, or None if it must be dropped."""
    name = _local_name(elem.tag)
    if name not in ALLOWED_TAGS:
        return None

    clone = ElementTree.Element(name)
    if elem.text and (elem.text or "").strip():
        clone.text = elem.text
    for attr, value in elem.attrib.items():
        local = _local_name(attr)
        if local not in ALLOWED_ATTRS:
            continue
        if local == "href" and not value.startswith("#"):
            continue
        clone.set(local, value)

    for child in elem:
        cleaned = _clean_tree(child)
        if cleaned is not None:
            clone.append(cleaned)
    return clone


def sanitize_svg(svg_text: str) -> str:
    """Strip anything dangerous from an LLM-generated SVG. Returns '' if nothing safe survives."""
    if not svg_text or DANGEROUS.search(svg_text):
        return ""
    try:
        root = ElementTree.fromstring(svg_text)
    except ElementTree.ParseError:
        return ""
    if _local_name(root.tag) != "svg":
        return ""

    cleaned = _clean_tree(root)
    if cleaned is None:
        return ""
    if not cleaned.findall(".//*") and not (cleaned.text or "").strip():
        return ""

    svg_body = ElementTree.tostring(cleaned, encoding="unicode", method="xml")
    if DANGEROUS.search(svg_body):
        return ""
    return svg_body


def svg_data_uri(svg_text: str) -> str:
    """URL-encoded data URI for use in markdown so plain markdown viewers show the diagram."""
    encoded = svg_text.replace("%", "%25").replace("#", "%23")
    encoded = encoded.replace('"', "%22").replace("<", "%3C").replace(">", "%3E")
    return f"data:image/svg+xml;charset=utf-8,{encoded}"
