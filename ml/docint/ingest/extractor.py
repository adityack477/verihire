# ml/docint/ingest/extractor.py
"""PyMuPDF page-wise PDF text extraction (VeriHire Part 1).

Conventions
-----------
* ``page_no`` is 1-based and equals the page's position in ``ExtractedDoc.pages``.
* ``is_scanned``: True when the mean number of non-whitespace characters per
  page is below ``MIN_CHARS_PER_PAGE`` (50). No OCR is performed.
* ``suspicious_hidden_text``: True when any non-blank span is (a) near-white and
  not centred inside a non-white filled rectangle, (b) smaller than 2pt, or
  (c) outside the page rectangle. Limitations: images, clipping paths and
  text hidden behind other objects are not detected. Flagged text is kept in
  the page text (flag only, no removal).
* Reading order: blocks are y-sorted; if a vertical gutter separating two
  substantial columns is found, each band between full-width blocks is read
  left column first, then right column.
* Normalisation: ligatures expanded, NBSP/zero-width/soft-hyphen handled,
  glyph bullets mapped to "• ", whitespace collapsed, "lower-\\nlower" line-break
  hyphenation joined (this can also join genuine hyphenated compounds).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

MIN_CHARS_PER_PAGE = 50
MAX_PDF_BYTES = 25 * 1024 * 1024
MIN_SPAN_FONT_SIZE = 2.0
OFFPAGE_TOLERANCE = 2.0
WHITE_THRESHOLD = 245
_TEXT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE  # deliberately no MEDIABOX_CLIP

Block = tuple[tuple[float, float, float, float], str]


class Page(TypedDict):
    """One extracted page."""

    page_no: int
    text: str


@dataclass(frozen=True)
class ExtractedDoc:
    """Frozen Part 1 extraction contract."""

    pages: list[Page]
    page_count: int
    is_scanned: bool
    suspicious_hidden_text: bool


class ExtractionError(Exception):
    """Raised for corrupt, unreadable, encrypted or invalid PDFs."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}
_TRANSLATE: dict[int, str | None] = {ord(k): v for k, v in _LIGATURES.items()}
_TRANSLATE.update({
    0x00A0: " ", 0x2002: " ", 0x2003: " ", 0x2007: " ", 0x2009: " ",
    0x200A: " ", 0x202F: " ", 0x200B: None, 0x200C: None, 0x200D: None,
    0x2060: None, 0xFEFF: None, 0x00AD: None, 0x2018: "'", 0x2019: "'",
    0x201C: '"', 0x201D: '"', 0x2028: "\n", 0x2029: "\n",
})
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LEADING_BULLET = re.compile(
    "^[\u2022\u25cf\u25cb\u25e6\u25aa\u25ab\u25a0\u25a1\u25c6\u25c7\u25b8\u25b9"
    "\u25ba\u25bb\u2023\u2043\u2219\u00b7\u27a2\u27a4\u2794\u2713\u2714"
    "\uf0b7\uf0a7\uf0d8\uf076\uf0fc]\\s*"
)
_DEHYPHEN_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


def extract_pdf(pdf_bytes: bytes) -> ExtractedDoc:
    """Extract page-wise text from PDF bytes.

    Raises:
        ExtractionError: empty/oversized/non-PDF/corrupt/encrypted input.
    """
    if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
        raise ExtractionError("invalid_input", "pdf_bytes must be bytes-like")
    data = bytes(pdf_bytes)
    if not data:
        raise ExtractionError("empty", "no PDF data supplied")
    if len(data) > MAX_PDF_BYTES:
        raise ExtractionError("too_large", "PDF exceeds size limit")
    if b"%PDF-" not in data[:1024]:
        raise ExtractionError("not_pdf", "data is not a PDF")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ExtractionError("corrupt", "PDF could not be opened") from exc

    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ExtractionError("encrypted", "encrypted PDFs are not supported")
        if doc.page_count < 1:
            raise ExtractionError("no_pages", "PDF contains no pages")
        pages: list[Page] = []
        hidden = False
        for index in range(doc.page_count):
            try:
                text, page_hidden = _extract_page(doc.load_page(index))
            except ExtractionError:
                raise
            except Exception as exc:
                raise ExtractionError("corrupt", f"page {index + 1} unreadable") from exc
            hidden = hidden or page_hidden
            pages.append({"page_no": index + 1, "text": text})
    finally:
        doc.close()

    chars = sum(len(re.sub(r"\s", "", p["text"])) for p in pages)
    is_scanned = (chars / len(pages)) < MIN_CHARS_PER_PAGE
    return ExtractedDoc(
        pages=pages,
        page_count=len(pages),
        is_scanned=is_scanned,
        suspicious_hidden_text=hidden,
    )


def _extract_page(page: Any) -> tuple[str, bool]:
    rect = page.rect
    data = page.get_text("dict", flags=_TEXT_FLAGS)
    fills = _filled_rects(page)
    blocks: list[Block] = []
    hidden = False
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        for line in block.get("lines", []):
            parts: list[str] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip() and _is_suspicious(span, rect, fills):
                    hidden = True
                parts.append(text)
            raw = "".join(parts)
            if raw.strip():
                lines.append(raw)
        if lines:
            bbox = tuple(float(v) for v in block["bbox"])
            blocks.append((bbox, "\n".join(lines)))  # type: ignore[arg-type]
    ordered = _order_blocks(blocks, float(rect.width))
    return _normalize("\n\n".join(text for _, text in ordered)), hidden


def _filled_rects(page: Any) -> list[Any]:
    rects = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill is None:
            continue
        if len(fill) == 3 and min(fill) >= 0.96:
            continue
        rects.append(fitz.Rect(drawing["rect"]))
    return rects


def _is_suspicious(span: dict[str, Any], page_rect: Any, fills: list[Any]) -> bool:
    if float(span.get("size", 0.0)) < MIN_SPAN_FONT_SIZE:
        return True
    bbox = fitz.Rect(span["bbox"])
    tol = OFFPAGE_TOLERANCE
    if (bbox.x1 < page_rect.x0 - tol or bbox.x0 > page_rect.x1 + tol
            or bbox.y1 < page_rect.y0 - tol or bbox.y0 > page_rect.y1 + tol):
        return True
    color = int(span.get("color", 0))
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    if min(r, g, b) >= WHITE_THRESHOLD:
        centre = fitz.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
        return not any(fill.contains(centre) for fill in fills)
    return False


def _find_gutter(blocks: list[Block], page_width: float) -> float | None:
    if len(blocks) < 4 or page_width <= 0:
        return None
    narrow_limit = 0.75 * page_width
    total_chars = sum(len(t) for _, t in blocks) or 1
    step = max(2.0, page_width / 150.0)
    x = 0.15 * page_width
    while x <= 0.85 * page_width:
        ok = True
        left_n = right_n = left_c = right_c = 0
        for (x0, _y0, x1, _y1), text in blocks:
            if x0 < x < x1:
                if (x1 - x0) <= narrow_limit:
                    ok = False
                    break
            elif x1 <= x:
                left_n += 1
                left_c += len(text)
            else:
                right_n += 1
                right_c += len(text)
        if (ok and left_n >= 2 and right_n >= 2
                and left_c >= 0.2 * total_chars and right_c >= 0.2 * total_chars):
            return x
        x += step
    return None


def _order_blocks(blocks: list[Block], page_width: float) -> list[Block]:
    if not blocks:
        return []
    key = lambda b: (round(b[0][1], 1), b[0][0])  # noqa: E731
    gutter = _find_gutter(blocks, page_width)
    if gutter is None:
        return sorted(blocks, key=key)
    out: list[Block] = []
    left: list[Block] = []
    right: list[Block] = []
    for block in sorted(blocks, key=key):
        x0, _, x1, _ = block[0]
        if x0 < gutter < x1:
            out.extend(left + right)
            left, right = [], []
            out.append(block)
        elif x1 <= gutter:
            left.append(block)
        else:
            right.append(block)
    out.extend(left + right)
    return out


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_TRANSLATE)
    text = _CONTROL_RE.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(_LEADING_BULLET.sub("\u2022 ", line).strip())
    merged: list[str] = []
    i = 0
    while i < len(lines):
        if (lines[i] == "\u2022" and i + 1 < len(lines) and lines[i + 1]
                and not lines[i + 1].startswith("\u2022")):
            merged.append("\u2022 " + lines[i + 1])
            i += 2
            continue
        merged.append(lines[i])
        i += 1
    text = "\n".join(merged)
    text = _DEHYPHEN_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()