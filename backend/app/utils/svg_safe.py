import re
from html.parser import HTMLParser
from xml.etree import ElementTree

ALLOWED_TAGS = {
    "svg", "g", "defs", "marker", "path", "rect", "circle", "ellipse",
    "line", "polyline", "polygon", "text", "tspan", "title", "desc",
    "lineargradient", "radialgradient", "stop", "use", "symbol",
}

ALLOWED_ATTRS = {
    "xmlns", "viewbox", "width", "height", "fill", "stroke", "stroke-width",
    "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "opacity",
    "font-size", "font-family", "font-weight", "text-anchor", "x", "y",
    "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "points",
    "transform", "d", "id", "offset", "stop-color", "stop-opacity",
    "dx", "dy", "href", "dominant-baseline",
    "marker-width", "marker-height", "marker-start", "marker-end",
    "refx", "refy", "orient",
    "markerwidth", "markerheight", "markerend", "markerstart",
}

ATTR_RENAME = {
    "viewbox": "viewBox",
    "marker-width": "markerWidth",
    "marker-height": "markerHeight",
    "marker-start": "markerStart",
    "marker-end": "markerEnd",
    "refx": "refX",
    "refy": "refY",
    "lineargradient": "linearGradient",
    "radialgradient": "radialGradient",
    "markerwidth": "markerWidth",
    "markerheight": "markerHeight",
    "markerstart": "markerStart",
    "markerend": "markerEnd",
    "stopcolor": "stop-color",
    "stopopacity": "stop-opacity",
    "strokewidth": "stroke-width",
    "strokelinecap": "stroke-linecap",
    "strokelinejoin": "stroke-linejoin",
    "strokedasharray": "stroke-dasharray",
    "fontsize": "font-size",
    "fontfamily": "font-family",
    "fontweight": "font-weight",
    "textanchor": "text-anchor",
    "dominantbaseline": "dominant-baseline",
}

DANGEROUS = re.compile(
    r"<script|script>|onerror\s*=|onload\s*=|onclick\s*=|on[A-Za-z]+\s*=|javascript:|expression\(|foreignObject|foreignobject|embed|iframe|&lt;script",
    re.IGNORECASE,
)

_ATTR_ESCAPE_RE = re.compile(r"[&<>\"]")
_ATTR_ESCAPE_MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}
_TEXT_ESCAPE_RE = re.compile(r"[&<>]")
_TEXT_ESCAPE_MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def _attr_value(value: str | None) -> str:
    return _ATTR_ESCAPE_RE.sub(lambda m: _ATTR_ESCAPE_MAP[m.group(0)], value or "")


def _text_escape(text: str) -> str:
    return _TEXT_ESCAPE_RE.sub(lambda m: _TEXT_ESCAPE_MAP[m.group(0)], text)


class _SvgRebuilder(HTMLParser):
    """Rebuild a clean, well-formed SVG from (possibly malformed) LLM output.

    HTMLParser is tolerant of unclosed/mismatched tags, so Gemini's sloppy
    markup still yields a valid SVG instead of being dropped entirely.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._stack: list[str] = []
        self._skip = 0

    def _clean_attrs(self, attrs: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
        cleaned = []
        for name, value in attrs:
            local = name.lower()
            if local not in ALLOWED_ATTRS:
                continue
            if local == "href" and not (value or "").startswith("#"):
                continue
            cleaned.append((ATTR_RENAME.get(local, local), _attr_value(value)))
        return cleaned

    def _attrs_str(self, attrs: list[tuple[str, str]]) -> str:
        return "".join(f' {k}="{v}"' for k, v in attrs)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in ALLOWED_TAGS:
            self._skip += 1
            return
        if self._skip:
            self._skip += 1
            return
        self._stack.append(tag)
        self.parts.append(f"<{tag}{self._attrs_str(self._clean_attrs(attrs))}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in ALLOWED_TAGS or self._skip:
            return
        self.parts.append(f"<{tag}{self._attrs_str(self._clean_attrs(attrs))}/>")

    def handle_endtag(self, tag: str) -> None:
        if self._skip:
            self._skip -= 1
            return
        tag = tag.lower()
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i] == tag:
                for open_tag in reversed(self._stack[i:]):
                    self.parts.append(f"</{open_tag}>")
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if self._skip or not data:
            return
        for open_tag in reversed(self._stack):
            if open_tag in ("text", "tspan", "title", "desc"):
                self.parts.append(_text_escape(data))
                return

    def result(self) -> str:
        for open_tag in reversed(self._stack):
            self.parts.append(f"</{open_tag}>")
        return "".join(self.parts)


def sanitize_svg(svg_text: str) -> str:
    """Strip anything dangerous from an LLM-generated SVG. Returns '' if nothing safe survives."""
    if not svg_text:
        return ""
    pre = re.sub(r"<!DOCTYPE[\s\S]*?>", "", svg_text, flags=re.IGNORECASE)
    pre = re.sub(r"<\?xml[^>]*\?>", "", pre, flags=re.IGNORECASE)
    pre = re.sub(r"<style[\s\S]*?</style>", "", pre, flags=re.IGNORECASE)
    pre = re.sub(r"<!--[\s\S]*?-->", "", pre)
    if not pre or DANGEROUS.search(pre):
        return ""

    parser = _SvgRebuilder()
    try:
        parser.feed(pre)
        parser.close()
    except Exception:
        return ""
    svg_body = parser.result()
    if not svg_body.strip() or "<svg" not in svg_body or DANGEROUS.search(svg_body):
        return ""
    try:
        root = ElementTree.fromstring(svg_body)
    except ElementTree.ParseError:
        return ""
    if _local_name(root.tag) != "svg":
        return ""

    cleaned = _clean_tree(root)
    if cleaned is None:
        return ""
    if not cleaned.findall(".//*") and not (cleaned.text or "").strip():
        return ""

    final = ElementTree.tostring(cleaned, encoding="unicode", method="xml")
    if DANGEROUS.search(final):
        return ""
    return final


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
        local = _local_name(attr).lower()
        if local not in ALLOWED_ATTRS:
            continue
        if local == "href" and not value.startswith("#"):
            continue
        clone.set(ATTR_RENAME.get(local, local), value)

    for child in elem:
        cleaned = _clean_tree(child)
        if cleaned is not None:
            clone.append(cleaned)
    return clone


def svg_data_uri(svg_text: str) -> str:
    """URL-encoded data URI for use in markdown so plain markdown viewers show the diagram."""
    encoded = svg_text.replace("%", "%25").replace("#", "%23")
    encoded = encoded.replace('"', "%22").replace("<", "%3C").replace(">", "%3E")
    return f"data:image/svg+xml;charset=utf-8,{encoded}"
