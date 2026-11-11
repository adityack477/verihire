# ml/docint/ingest/chunker.py
"""Section-aware chunking (VeriHire Part 1).

Offsets: ``char_start/char_end`` index the page's extracted text, so
``text_original == page_text[char_start:char_end]``. ``text_scrubbed`` is
``scrub_slice`` of the same range; chunk boundaries fall on line boundaries
(or whitespace, snapped away from PII spans), so no span is cut.
``token_count`` is an approximate word/punctuation count, not a model tokenizer.
Chunks never span pages. Text before the first heading has section "header";
unrecognised headings give "other".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .extractor import Page
from .pii_scrubber import PiiSpan, scrub_slice

TARGET_MIN = 800
TARGET_MAX = 1200
HARD_MAX = 1500


@dataclass(frozen=True)
class Chunk:
    """Frozen Part 1 chunk contract."""

    chunk_index: int
    section: str
    page_no: int
    char_start: int
    char_end: int
    text_original: str
    text_scrubbed: str
    token_count: int

    def embedding_text(self) -> str:
        """Exact scrubbed string to embed."""
        return self.text_scrubbed


_SECTION_HEADINGS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "career summary", "executive summary",
                "profile", "professional profile", "objective", "career objective",
                "about me", "about"),
    "experience": ("experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "career history", "internships",
                   "internship", "internship experience", "relevant experience"),
    "skills": ("skills", "technical skills", "key skills", "core competencies",
               "skills and tools", "technologies", "tech stack", "skills summary",
               "areas of expertise", "tools and technologies", "technical proficiency"),
    "education": ("education", "academic background", "academics", "educational qualifications",
                  "education and training", "academic qualifications", "qualifications"),
    "projects": ("projects", "personal projects", "academic projects", "key projects",
                 "selected projects", "project experience"),
    "certifications": ("certifications", "certificates", "licenses and certifications",
                       "certifications and courses", "courses", "licenses",
                       "training and certifications"),
    "other": ("awards", "achievements", "honors", "publications", "research",
              "extracurricular", "extracurricular activities", "volunteer", "volunteering",
              "languages", "interests", "hobbies", "references", "leadership", "activities",
              "positions of responsibility"),
}
_HEADING_LOOKUP = {v: s for s, variants in _SECTION_HEADINGS.items() for v in variants}

_BULLET_RE = re.compile(r"^(?:[\u2022\-\*\u2013\u2014\u25aa]|\d{1,2}[.)])\s+")
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE_RE = re.compile(
    r"(?i)\b(?:" + _MONTH + r"\s+)?(?:19|20)\d{2}\b\s*(?:[-\u2013\u2014]|to)\s*"
    r"(?:(?:" + _MONTH + r"\s+)?(?:19|20)\d{2}|\d{2}|present|current|ongoing|now)\b"
)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class _Line:
    start: int
    end: int
    text: str


@dataclass
class _Item:
    start_line: int
    end_line: int
    section: str
    has_bullet: bool
    header_only: bool


def chunk_document(pages: Sequence[Page],
                   spans_by_page: Mapping[int, Sequence[PiiSpan]]) -> list[Chunk]:
    """Create section-aware chunks for all pages."""
    chunks: list[Chunk] = []
    section = "header"
    for page in pages:
        text = page["text"]
        page_no = page["page_no"]
        spans = list(spans_by_page.get(page_no, ()))
        lines = _split_lines(text)
        items, section = _page_items(lines, section)
        for sec, a, b in _pack(_units(items, lines, text, spans), text, spans):
            scrubbed = scrub_slice(text, spans, a, b)
            if b <= a or not scrubbed.strip():
                continue
            chunks.append(Chunk(
                chunk_index=len(chunks), section=sec, page_no=page_no,
                char_start=a, char_end=b, text_original=text[a:b],
                text_scrubbed=scrubbed, token_count=len(_TOKEN_RE.findall(scrubbed)),
            ))
    return chunks


def _split_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    pos = 0
    for raw in text.split("\n"):
        lines.append(_Line(pos, pos + len(raw), raw))
        pos += len(raw) + 1
    return lines


def _trim(line: _Line) -> tuple[int, int]:
    lead = len(line.text) - len(line.text.lstrip())
    trail = len(line.text) - len(line.text.rstrip())
    return line.start + lead, line.end - trail


def _norm_heading(text: str) -> str:
    t = text.lower().replace("&", " and ").replace("/", " and ")
    return " ".join(re.sub(r"[^a-z ]", " ", t).split())


def _heading_section(text: str) -> Optional[str]:
    if len(text) > 50 or _BULLET_RE.match(text):
        return None
    return _HEADING_LOOKUP.get(_norm_heading(text))


def _is_header_like(text: str) -> bool:
    return (len(text) <= 80 and (text[0].isupper() or text[0].isdigit())
            and not text.endswith((".", ",", ";")))


def _next_is_bullet(lines: list[_Line], i: int) -> bool:
    j = i + 1
    while j < len(lines) and not lines[j].text.strip():
        j += 1
    return j < len(lines) and bool(_BULLET_RE.match(lines[j].text.strip()))


def _page_items(lines: list[_Line], section: str) -> tuple[list[_Item], str]:
    items: list[_Item] = []
    cur: Optional[_Item] = None
    prev_blank = True
    for i, line in enumerate(lines):
        text = line.text.strip()
        if not text:
            prev_blank = True
            continue
        heading = _heading_section(text)
        if heading is not None:
            section = heading
            if cur is not None:
                items.append(cur)
            cur = _Item(i, i, section, False, True)
            prev_blank = False
            continue
        if _BULLET_RE.match(text):
            if cur is not None and not cur.has_bullet:
                cur.has_bullet = True
                cur.header_only = False
                cur.end_line = i
            else:
                if cur is not None:
                    items.append(cur)
                cur = _Item(i, i, section, True, False)
        else:
            header_like = _is_header_like(text)
            new_header = header_like and (
                prev_blank or bool(_DATE_RE.search(text)) or _next_is_bullet(lines, i))
            if cur is None:
                cur = _Item(i, i, section, False, header_like)
            elif cur.header_only:
                cur.end_line = i
                cur.header_only = header_like
            elif new_header or prev_blank:
                items.append(cur)
                cur = _Item(i, i, section, False, header_like and new_header)
            else:
                cur.end_line = i
        prev_blank = False
    if cur is not None:
        items.append(cur)
    return items, section


def _measure(text: str, spans: Sequence[PiiSpan], a: int, b: int) -> int:
    return max(b - a, len(scrub_slice(text, spans, a, b)))


def _find_cut(text: str, spans: Sequence[PiiSpan], pos: int, limit: int) -> int:
    idx = text.rfind(" ", pos + 1, limit)
    cut = idx if idx > pos else limit
    for s in spans:
        if s.start < cut < s.end:
            cut = s.start if s.start > pos else s.end
            break
    return cut


def _split_long_line(text: str, spans: Sequence[PiiSpan], a: int, b: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = a
    while _measure(text, spans, pos, b) > TARGET_MAX:
        cut = _find_cut(text, spans, pos, min(pos + TARGET_MAX, b))
        out.append((pos, cut))
        pos = cut
        while pos < b and text[pos].isspace():
            pos += 1
    if pos < b:
        out.append((pos, b))
    return out


def _split_range(lines: list[_Line], i0: int, i1: int, text: str,
                 spans: Sequence[PiiSpan]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur_a: Optional[int] = None
    cur_b = 0
    for i in range(i0, i1 + 1):
        ls, le = _trim(lines[i])
        if ls >= le:
            continue
        if le - ls > HARD_MAX:
            if cur_a is not None:
                out.append((cur_a, cur_b))
                cur_a = None
            out.extend(_split_long_line(text, spans, ls, le))
        elif cur_a is None:
            cur_a, cur_b = ls, le
        elif _measure(text, spans, cur_a, le) <= TARGET_MAX:
            cur_b = le
        else:
            out.append((cur_a, cur_b))
            cur_a, cur_b = ls, le
    if cur_a is not None:
        out.append((cur_a, cur_b))
    return out


def _units(items: list[_Item], lines: list[_Line], text: str,
           spans: Sequence[PiiSpan]) -> list[tuple[str, int, int]]:
    units: list[tuple[str, int, int]] = []
    for item in items:
        a = _trim(lines[item.start_line])[0]
        b = _trim(lines[item.end_line])[1]
        if _measure(text, spans, a, b) <= HARD_MAX:
            units.append((item.section, a, b))
        else:
            for pa, pb in _split_range(lines, item.start_line, item.end_line, text, spans):
                units.append((item.section, pa, pb))
    return units


def _pack(units: list[tuple[str, int, int]], text: str,
          spans: Sequence[PiiSpan]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    cur: Optional[list] = None
    for sec, a, b in units:
        if cur is None:
            cur = [sec, a, b]
            continue
        if sec == cur[0]:
            merged = _measure(text, spans, cur[1], b)
            cur_len = _measure(text, spans, cur[1], cur[2])
            if merged <= TARGET_MAX or (cur_len < TARGET_MIN and merged <= HARD_MAX):
                cur[2] = b
                continue
        out.append((cur[0], cur[1], cur[2]))
        cur = [sec, a, b]
    if cur is not None:
        out.append((cur[0], cur[1], cur[2]))
    return out