# ml/docint/ingest/profile_parser.py
"""Structured resume profile from PII-scrubbed chunks (VeriHire Part 2).

Everything is parsed from ``Chunk.text_scrubbed`` only, so no raw PII can reach
the profile. ResumeProfile lives here (re-exported by ingest.py) to avoid a
circular import.

Documented rules
* years_experience: date ranges found ONLY in "experience"-section chunks.
  Formats: "Jan 2024", "January 2024", "01/2024", "2024-01", "2024", and
  "May - June 2026" (month span in one year). End may be present/current/
  ongoing/now/to date. Year-only dates are placed mid-year (July) so "2021 - 2023"
  counts as 2.0 years. Overlaps are merged. Ranges that are reversed, have invalid
  months, start before 1960 or in the future, or exceed 50 years are ignored; end
  dates are clamped to ``as_of`` (default: today, so "present" is time dependent;
  pass ``as_of`` for reproducibility). Total capped at 45 years. None if no valid range.
* education_level: highest of Diploma=1; B.Tech/BE/BSc/Bachelor=2;
  M.Tech/ME/MSc/MBA/Master=3; PhD=4, from "education" chunks only (in-progress
  degrees count). 0 if nothing matches.
* education_fields: from a fixed list of fields, order of first appearance.
* titles: heuristic role phrases from non-bullet lines of experience chunks.
* summary: "summary"-section text, <= 300 chars; None if there is no such section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from .chunker import Chunk, _heading_section
from .skills import extract_skills

PARSER_VERSION = "part2-1.0"
MAX_YEARS_EXPERIENCE = 45.0
MAX_PERIOD_YEARS = 50
MIN_YEAR = 1960
MAX_TITLES = 10
MAX_CERTIFICATIONS = 15
SUMMARY_MAX_CHARS = 300


@dataclass(frozen=True)
class ResumeProfile:
    """Frozen profile contract."""

    skills: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    years_experience: Optional[float] = None
    education_level: int = 0
    education_fields: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    parser_version: str = PARSER_VERSION


_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_TOKEN = (r"(?:" + _MON + r"\.?[ \t]*,?[ \t]*'?\d{4}"
          r"|\d{1,2}[ \t]*[/.][ \t]*\d{4}"
          r"|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])"
          r"|(?:19|20)\d{2})")
_SEP = r"[ \t]*(?:[-\u2013\u2014]+|\bto\b|\buntil\b|\btill\b)[ \t]*"
_RANGE_RE = re.compile(
    r"(?<![\w/.])(?P<s>" + _TOKEN + r")" + _SEP
    + r"(?P<e>" + _TOKEN + r"|present|current|ongoing|now|date)(?![\w/])", re.IGNORECASE)
_SPAN_RE = re.compile(
    r"(?<![\w/.])(?P<m1>" + _MON + r")\.?[ \t]*(?:[-\u2013\u2014]+|\bto\b)[ \t]*(?P<m2>" + _MON
    + r")\.?[ \t]*,?[ \t]*'?(?P<y>\d{4})(?![\w/])", re.IGNORECASE)
_PRESENT = frozenset({"present", "current", "ongoing", "now", "date"})


def _parse_token(token: str) -> Optional[tuple[int, bool]]:
    """Return (month_index, is_month_precise) or None."""
    t = token.strip().lower()
    m = re.fullmatch(r"([a-z]+)\.?[ \t]*,?[ \t]*'?(\d{4})", t)
    if m:
        month = _MONTHS.get(m.group(1)[:3])
        return (int(m.group(2)) * 12 + month - 1, True) if month else None
    m = re.fullmatch(r"(\d{1,2})[ \t]*[/.][ \t]*(\d{4})", t)
    if m:
        month = int(m.group(1))
        return (int(m.group(2)) * 12 + month - 1, True) if 1 <= month <= 12 else None
    m = re.fullmatch(r"(\d{4})-(\d{2})", t)
    if m:
        month = int(m.group(2))
        return (int(m.group(1)) * 12 + month - 1, True) if 1 <= month <= 12 else None
    if re.fullmatch(r"\d{4}", t):
        return (int(t) * 12 + 6, False)
    return None


def _years_experience(text: str, as_of: date) -> Optional[float]:
    now_excl = as_of.year * 12 + as_of.month  # exclusive end of the current month
    periods: list[tuple[int, int]] = []

    def add(start: int, end: int) -> None:
        end = min(end, now_excl)
        if start < MIN_YEAR * 12 or start >= end or end - start > MAX_PERIOD_YEARS * 12:
            return
        periods.append((start, end))

    for m in _RANGE_RE.finditer(text):
        start = _parse_token(m.group("s"))
        if start is None:
            continue
        end_raw = m.group("e").lower()
        if end_raw in _PRESENT:
            end_excl = now_excl
        else:
            end = _parse_token(end_raw)
            if end is None:
                continue
            end_excl = end[0] + 1 if end[1] else end[0]
        add(start[0], end_excl)
    for m in _SPAN_RE.finditer(text):
        m1 = _MONTHS[m.group("m1").lower()[:3]]
        m2 = _MONTHS[m.group("m2").lower()[:3]]
        if m2 >= m1:
            year = int(m.group("y"))
            add(year * 12 + m1 - 1, year * 12 + m2)
    if not periods:
        return None
    periods.sort()
    merged: list[list[int]] = []
    for start, end in periods:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    months = sum(e - s for s, e in merged)
    return min(round(months / 12, 1), MAX_YEARS_EXPERIENCE)


_TITLE_RE = re.compile(
    r"\b(?:intern|trainee|engineer|developer|analyst|scientist|manager|consultant|architect|"
    r"researcher|associate|specialist|administrator|designer|officer|director|programmer|"
    r"coordinator|technician|lead|head|assistant|apprentice|fellow|founder|co-founder|executive)\b",
    re.IGNORECASE)
_TITLE_SPLIT = re.compile(r"\s*[|\u2022\u00b7;]\s*|\s+[-\u2013\u2014]\s+|\s*,\s*|\s+(?:at|@)\s+|\s*\(")
_BULLET_RE = re.compile(r"^(?:[\u2022\-\*\u2013\u2014\u25aa]|\d{1,2}[.)])\s+")


def _titles(chunks: Sequence[Chunk]) -> list[str]:
    found: dict[str, str] = {}
    for chunk in chunks:
        if chunk.section != "experience":
            continue
        for raw in chunk.text_scrubbed.split("\n"):
            line = raw.strip()
            if not line or _BULLET_RE.match(line) or len(line.split()) > 12 or line.endswith("."):
                continue
            for seg in _TITLE_SPLIT.split(line):
                seg = " ".join(seg.strip().rstrip(")").split())
                words = seg.split()
                if (not seg or len(words) > 6 or not seg[0].isupper() or "[" in seg
                        or any(ch.isdigit() for ch in seg) or seg.endswith((".", ",", ":", ";"))
                        or not _TITLE_RE.search(seg)):
                    continue
                found.setdefault(seg.lower(), seg)
    return list(found.values())[:MAX_TITLES]


_EDU_PATTERNS: tuple[tuple[int, "re.Pattern[str]"], ...] = (
    (4, re.compile(r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctor\s+of\s+philosophy\b", re.I)),
    (3, re.compile(r"\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmba\b|\bmca\b|\bm\.?\s?com\b|\bmaster(?:'s|s)?\b", re.I)),
    (3, re.compile(r"\bM\.E\b\.?|\bME\s+(?:in|\()")),
    (2, re.compile(r"\bb\.?\s?tech\b|\bb\.?\s?sc\b|\bbca\b|\bbba\b|\bb\.?\s?com\b"
                   r"|\bbachelor(?:'s|s)?\b|\bundergraduate\b", re.I)),
    (2, re.compile(r"\bB\.E\b\.?|\bBE\s+(?:in|\()")),
    (2, re.compile(r"\bB\.S\b\.?|\bBS\s+(?:in|\()")),
    (1, re.compile(r"\bdiploma\b", re.I)),
)
_FIELD_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (name, re.compile(pat, re.I)) for name, pat in (
        ("computer science", r"computer\s+science|\bcse\b"),
        ("computer engineering", r"computer\s+engineering"),
        ("information technology", r"information\s+technology"),
        ("information systems", r"information\s+systems"),
        ("software engineering", r"software\s+engineering"),
        ("data science", r"data\s+science"),
        ("artificial intelligence", r"artificial\s+intelligence"),
        ("cybersecurity", r"cyber\s*security"),
        ("electronics and communication",
         r"electronics\s*(?:&|and)\s*(?:tele)?communications?(?:\s+engineering)?"),
        ("electrical engineering", r"electrical(?:\s*(?:&|and)\s*electronics)?\s+engineering"),
        ("mechanical engineering", r"mechanical\s+engineering"),
        ("civil engineering", r"civil\s+engineering"),
        ("chemical engineering", r"chemical\s+engineering"),
        ("robotics", r"\brobotics\b"),
        ("biotechnology", r"biotechnology"),
        ("statistics", r"\bstatistics\b"),
        ("mathematics", r"\bmathematics\b|\bmaths?\b"),
        ("physics", r"\bphysics\b"),
        ("chemistry", r"\bchemistry\b"),
        ("business administration", r"business\s+administration"),
        ("finance", r"\bfinance\b"),
        ("economics", r"\beconomics\b"),
        ("commerce", r"\bcommerce\b"),
    ))


def _education_level(text: str) -> int:
    level = 0
    for value, pattern in _EDU_PATTERNS:
        if pattern.search(text):
            level = max(level, value)
    return level


def _education_fields(text: str) -> list[str]:
    hits: list[tuple[int, str]] = []
    for name, pattern in _FIELD_PATTERNS:
        m = pattern.search(text)
        if m:
            hits.append((m.start(), name))
    return [name for _, name in sorted(hits)]


_PLACEHOLDER_ONLY = re.compile(r"(?:\[[A-Z]+\][\s|,]*)+")
_DATE_ONLY = re.compile(r"[\d\s/\-\u2013\u2014.,]+")


def _content_lines(chunks: Sequence[Chunk], section: str) -> list[str]:
    lines: list[str] = []
    for chunk in chunks:
        if chunk.section != section:
            continue
        for raw in chunk.text_scrubbed.split("\n"):
            line = _BULLET_RE.sub("", raw.strip()).strip()
            if (not line or _heading_section(line) is not None
                    or _PLACEHOLDER_ONLY.fullmatch(line)):
                continue
            lines.append(line)
    return lines


def _certifications(chunks: Sequence[Chunk]) -> list[str]:
    found: dict[str, str] = {}
    for line in _content_lines(chunks, "certifications"):
        if not 4 <= len(line) <= 120 or _DATE_ONLY.fullmatch(line):
            continue
        found.setdefault(line.lower(), line)
    return list(found.values())[:MAX_CERTIFICATIONS]


def _summary(chunks: Sequence[Chunk]) -> Optional[str]:
    text = " ".join(" ".join(_content_lines(chunks, "summary")).split())
    if not text:
        return None
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    cut = text[:SUMMARY_MAX_CHARS]
    idx = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if idx >= 100:
        return cut[:idx + 1]
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "..."


def parse_profile(chunks: Sequence[Chunk], as_of: Optional[date] = None) -> ResumeProfile:
    """Build a ResumeProfile from scrubbed chunks without guessing missing data."""
    items = list(chunks)
    today = as_of or date.today()
    experience_text = "\n".join(c.text_scrubbed for c in items if c.section == "experience")
    education_text = "\n".join(c.text_scrubbed for c in items if c.section == "education")
    all_text = "\n".join(c.text_scrubbed for c in items)
    return ResumeProfile(
        skills=extract_skills(all_text),
        titles=_titles(items),
        years_experience=_years_experience(experience_text, today) if experience_text else None,
        education_level=_education_level(education_text),
        education_fields=_education_fields(education_text),
        certifications=_certifications(items),
        summary=_summary(items),
        parser_version=PARSER_VERSION,
    )
