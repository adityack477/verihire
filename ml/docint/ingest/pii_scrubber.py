# ml/docint/ingest/pii_scrubber.py
"""Regex/heuristic PII detection and redaction (VeriHire Part 1).

Offset mapping: detection yields ``PiiSpan(start, end, label)`` over the
ORIGINAL page text. Scrubbed text for any original slice ``[a, b)`` is derived
by ``scrub_slice`` (spans replaced by placeholders), so chunk
``text_original`` stays an exact source slice. Spans never contain newlines.
spaCy is optional and only used as a last-resort name hint (local model only,
never downloaded). Raw PII is never logged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, NamedTuple, Optional, Sequence

from .extractor import Page

PLACEHOLDERS = {
    "NAME": "[NAME]", "EMAIL": "[EMAIL]", "PHONE": "[PHONE]",
    "URL": "[URL]", "ADDRESS": "[ADDRESS]",
}
_PRIORITY = {"EMAIL": 0, "URL": 1, "PHONE": 2, "ADDRESS": 3, "NAME": 4}


class PiiSpan(NamedTuple):
    """Half-open span ``[start, end)`` in original page text."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class PiiEntities:
    """Frozen Part 1 PII metadata contract."""

    full_name: Optional[str] = None
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScrubResult:
    """Entities plus per-page spans (keyed by page_no) for chunking."""

    entities: PiiEntities
    spans: dict[int, list[PiiSpan]]


_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
_HOSTS = (
    r"(?:linkedin\.com|github\.com|gitlab\.com|bitbucket\.org|kaggle\.com|medium\.com|"
    r"huggingface\.co|leetcode\.com|codeforces\.com|hackerrank\.com|stackoverflow\.com|"
    r"twitter\.com|behance\.net|dribbble\.com|github\.io|vercel\.app|netlify\.app|"
    r"herokuapp\.com|streamlit\.app|hf\.space)"
)
_URL_TAIL = r"[^\s<>\"'\[\]{}|]"
_URL_RE = re.compile(
    r"(?i)(?<![\w@.\-])(?:(?:https?://|www\.)" + _URL_TAIL + r"+|(?:[a-z0-9\-]+\.)*"
    + _HOSTS + r"(?![a-z0-9\-])(?:/" + _URL_TAIL + r"*)?)"
)
_PHONE_PATTERNS: list[tuple[re.Pattern[str], bool]] = [
    (re.compile(r"(?<![\w+])(?:\(?\+?91\)?[ \t.\-]?|0)?[6-9]\d{4}[ \t.\-]?\d{5}(?!\d)"), False),
    (re.compile(r"(?<![\w+])\+\d{1,3}(?:[ \t.\-]?\(?\d{1,5}\)?){2,5}(?!\d)"), True),
    (re.compile(r"(?<![\w.])(?:\(\d{3}\)[ \t.\-]?|\d{3}[ \t.\-])\d{3}[ \t.\-]\d{4}(?!\d)"), False),
    (re.compile(r"(?<![\w+])0\d{2,4}[ \t\-]\d{6,8}(?!\d)"), False),
]
_ADDR_LABEL_RE = re.compile(
    r"(?im)^[ \t]*(?:permanent[ \t]+|current[ \t]+|present[ \t]+|home[ \t]+)?"
    r"address[ \t]*[:\-\u2013][ \t]*(?P<v>[^\n]+)"
)
_STREET_RE = re.compile(
    r"\b\d{1,5}[A-Za-z]?(?:[ \t]+[A-Za-z][\w.'-]*){1,4}[ \t]+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr|Marg|Nagar|Colony)\b\.?"
    r"(?:,[^\n|\u2022]{0,60})?"
)
_PIN_ADDR_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z.'-]*(?: [A-Za-z][A-Za-z.'-]*)*,[ \t]*){1,3}"
    r"[A-Za-z][A-Za-z.'-]*(?: [A-Za-z][A-Za-z.'-]*)*[ ,\-\u2013]*(?<!\d)[1-9]\d{2}[ ]?\d{3}(?!\d)"
)

_NAME_STOP = frozenset(
    "resume cv curriculum vitae profile summary objective experience education skills "
    "projects contact details developer engineer analyst scientist manager intern student "
    "software data machine learning artificial intelligence full stack web backend frontend "
    "research university institute college school technologies technology page email phone "
    "mobile address linkedin github portfolio".split()
)
_SEGMENT_SPLIT = re.compile(r"\s*[|\u2022\u00b7,/;]\s*|\s+[-\u2013\u2014]\s+")
_NAME_LABEL = re.compile(r"(?i)^(?:full\s+)?name\s*[:\-]\s*")


def scrub_text(text: str, spans: Sequence[PiiSpan]) -> str:
    """Return ``text`` with every span replaced by its placeholder."""
    return scrub_slice(text, spans, 0, len(text))


def scrub_slice(text: str, spans: Sequence[PiiSpan], start: int = 0,
                end: Optional[int] = None) -> str:
    """Scrubbed version of ``text[start:end]``; partially covered spans are fully masked."""
    if end is None:
        end = len(text)
    out: list[str] = []
    pos = start
    for span in sorted(spans):
        if span.end <= start or span.start >= end:
            continue
        a = max(span.start, start)
        if a > pos:
            out.append(text[pos:a])
        out.append(PLACEHOLDERS[span.label])
        pos = min(span.end, end)
    out.append(text[pos:end])
    return "".join(out)


def scrub_pages(pages: Sequence[Page], use_spacy: bool = True) -> ScrubResult:
    """Detect PII across pages; returns entities and per-page spans."""
    structured: dict[int, list[PiiSpan]] = {}
    for position, page in enumerate(pages):
        structured[page["page_no"]] = _detect_structured(page["text"], position == 0)
    full_name = _detect_name(pages, structured, use_spacy) if pages else None

    emails: list[str] = []
    phones: list[str] = []
    urls: list[str] = []
    other: list[str] = []
    final: dict[int, list[PiiSpan]] = {}
    for page in pages:
        text = page["text"]
        candidates = list(structured[page["page_no"]])
        if full_name:
            candidates.extend(_name_spans(text, full_name))
        spans = _resolve(candidates)
        final[page["page_no"]] = spans
        for span in spans:
            raw = text[span.start:span.end]
            if span.label == "EMAIL":
                emails.append(raw)
            elif span.label == "PHONE":
                phones.append(raw)
            elif span.label == "URL":
                urls.append(raw)
            elif span.label == "ADDRESS":
                other.append(raw)
    entities = PiiEntities(
        full_name=full_name,
        emails=list(dict.fromkeys(emails)),
        phones=list(dict.fromkeys(phones)),
        urls=list(dict.fromkeys(urls)),
        other=list(dict.fromkeys(other)),
    )
    return ScrubResult(entities=entities, spans=final)


def _detect_structured(text: str, first_page: bool) -> list[PiiSpan]:
    found: list[PiiSpan] = []
    for m in _EMAIL_RE.finditer(text):
        found.append(PiiSpan(m.start(), m.end(), "EMAIL"))
    for m in _URL_RE.finditer(text):
        s, e = m.start(), m.end()
        while e > s and text[e - 1] in ".,;:!?)":
            e -= 1
        if e > s:
            found.append(PiiSpan(s, e, "URL"))
    for pattern, check_digits in _PHONE_PATTERNS:
        for m in pattern.finditer(text):
            if check_digits and not 7 <= sum(c.isdigit() for c in m.group()) <= 15:
                continue
            found.append(PiiSpan(m.start(), m.end(), "PHONE"))
    for m in _ADDR_LABEL_RE.finditer(text):
        value = m.group("v")
        found.append(PiiSpan(m.start("v"), m.start("v") + len(value.rstrip()), "ADDRESS"))
    for m in _STREET_RE.finditer(text):
        found.append(PiiSpan(m.start(), m.end(), "ADDRESS"))
    if first_page:
        top = _top_region_end(text, 8)
        for m in _PIN_ADDR_RE.finditer(text, 0, top):
            found.append(PiiSpan(m.start(), m.end(), "ADDRESS"))
    return found


def _resolve(candidates: Sequence[PiiSpan]) -> list[PiiSpan]:
    accepted: list[PiiSpan] = []
    for cand in sorted(candidates, key=lambda s: (_PRIORITY[s.label], -(s.end - s.start), s.start)):
        if cand.end <= cand.start:
            continue
        if any(cand.start < a.end and a.start < cand.end for a in accepted):
            continue
        accepted.append(cand)
    return sorted(accepted)


def _top_region_end(text: str, n_lines: int) -> int:
    count = 0
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        if line.strip():
            count += 1
            if count >= n_lines:
                return end
        pos = end + 1
    return len(text)


def _blank(text: str, spans: Sequence[PiiSpan]) -> str:
    out: list[str] = []
    pos = 0
    for span in sorted(spans):
        out.append(text[pos:span.start])
        out.append(" " * (span.end - span.start))
        pos = span.end
    out.append(text[pos:])
    return "".join(out)


def _nonblank_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            out.append(stripped)
            if len(out) >= limit:
                break
    return out


def _is_name_token(token: str) -> bool:
    core = token.replace(".", "").replace("'", "").replace("-", "")
    if not core or not core.isalpha() or len(token) > 20:
        return False
    return token[0].isupper() and core.lower() not in _NAME_STOP


def _looks_like_name(segment: str) -> bool:
    parts = segment.split()
    return 2 <= len(parts) <= 4 and all(_is_name_token(p) for p in parts)


def _segments(line: str) -> list[str]:
    out = []
    for seg in _SEGMENT_SPLIT.split(line):
        seg = " ".join(_NAME_LABEL.sub("", seg.strip()).split())
        if seg:
            out.append(seg)
    return out


def _name_from_lines(lines: Sequence[str]) -> Optional[str]:
    for line in lines:
        for seg in _segments(line):
            if _looks_like_name(seg):
                return seg
    return None


def _name_from_email(lines: Sequence[str], emails: Sequence[str]) -> Optional[str]:
    for email in emails:
        local = re.sub(r"[^a-z]", "", email.split("@")[0].lower())
        if len(local) < 4:
            continue
        for line in lines:
            for seg in _segments(line):
                if not _looks_like_name(seg):
                    continue
                tokens = [re.sub(r"[^a-z]", "", p.lower()) for p in seg.split()]
                if any(len(t) >= 3 and t in local for t in tokens):
                    return seg
    return None


@lru_cache(maxsize=1)
def _load_spacy() -> Any:
    try:
        import spacy  # type: ignore

        return spacy.load("en_core_web_sm")
    except (ImportError, OSError, ValueError):
        return None


def _name_from_spacy(text: str) -> Optional[str]:
    nlp = _load_spacy()
    if nlp is None or not text.strip():
        return None
    for ent in nlp(text[:1000]).ents:
        candidate = " ".join(ent.text.split())
        if ent.label_ == "PERSON" and _looks_like_name(candidate):
            return candidate
    return None


def _detect_name(pages: Sequence[Page], structured: dict[int, list[PiiSpan]],
                 use_spacy: bool) -> Optional[str]:
    first = pages[0]
    text = first["text"]
    spans = _resolve(structured[first["page_no"]])
    lines = _nonblank_lines(_blank(text, spans), 20)
    name = _name_from_lines(lines[:4])
    if name is None:
        emails = [text[s.start:s.end] for s in spans if s.label == "EMAIL"]
        name = _name_from_email(lines, emails)
    if name is None and use_spacy:
        name = _name_from_spacy("\n".join(lines[:10]))
    return name


def _name_spans(text: str, name: str) -> list[PiiSpan]:
    pattern = re.compile(
        r"(?<!\w)" + r"[ \t]+".join(re.escape(t) for t in name.split()) + r"(?!\w)",
        re.IGNORECASE,
    )
    return [PiiSpan(m.start(), m.end(), "NAME") for m in pattern.finditer(text)]