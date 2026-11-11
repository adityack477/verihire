# ml/docint/ingest/skills.py
"""Boundary-safe, deterministic skill extraction (VeriHire Part 2).

* Dictionary: ``skills_dict.json`` next to this file (cwd independent).
  Schema: ``{"schema_version": 1, "skills": {canonical_lowercase: [aliases]}}``.
* Matching: one case-insensitive regex over all terms, longest term first.
  A match may not touch a word character, ``.``, ``+`` or ``#`` on either side
  (so java != javascript, sql != mysql, c != c++/c#, js != node.js).
  Space/hyphen differences inside a term are equivalent (fine-tuning/fine tuning).
* Short/ambiguous terms: ``go``, ``r``, ``c`` match only as exact ``Go``/``R``/``C``
  standing alone between list separators (comma, ;, |, /, bullet, :, parentheses,
  line edges). ``golang`` is an ordinary alias. Common-word terms in
  ``_CASE_SENSITIVE`` (react, excel, spark, ...) must not be all-lowercase.
* Output: unique canonical skills in order of first appearance. No network/LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

SKILLS_DICT_PATH = Path(__file__).resolve().with_name("skills_dict.json")
SCHEMA_VERSION = 1

_STRICT_CONTEXT = {"go": "Go", "r": "R", "c": "C"}
_BEFORE_OK = ",;|/\u2022:(\n"
_AFTER_OK = ",;|/)\u2022\n"
_CASE_SENSITIVE = frozenset({
    "react", "spark", "swift", "excel", "rust", "ruby", "julia", "dart", "oracle", "unity",
    "express", "sas", "sap", "ml", "ai", "ros", "rag", "hive", "athena", "bootstrap",
    "confluence", "ionic", "electron", "angular", "sass", "groovy", "snowflake",
})


class SkillsDictionaryError(Exception):
    """Raised when skills_dict.json is missing or invalid."""


@dataclass(frozen=True)
class _Index:
    pattern: "re.Pattern[str]"
    lookup: dict[str, str]


def _norm(term: str) -> str:
    return re.sub(r"[\s\-]+", "", term.lower())


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise SkillsDictionaryError(f"duplicate keys in skills dictionary: {dup[:5]}")
    return dict(pairs)


def load_skills_dict(path: Optional[Path] = None) -> dict[str, list[str]]:
    """Load and validate the dictionary: canonical -> aliases."""
    target = Path(path) if path else SKILLS_DICT_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as exc:
        raise SkillsDictionaryError(f"cannot load skills dictionary '{target.name}'") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise SkillsDictionaryError("unsupported skills dictionary schema_version")
    skills = raw.get("skills")
    if not isinstance(skills, dict) or not skills:
        raise SkillsDictionaryError("'skills' must be a non-empty object")
    owner: dict[str, str] = {}
    for canon, aliases in skills.items():
        if not canon or canon != canon.strip().lower():
            raise SkillsDictionaryError(f"canonical skill must be lowercase: {canon!r}")
        if not isinstance(aliases, list):
            raise SkillsDictionaryError(f"aliases of {canon!r} must be a list")
        for term in [canon, *aliases]:
            if not isinstance(term, str) or not term or term != term.strip().lower():
                raise SkillsDictionaryError(f"invalid term {term!r} for {canon!r}")
            prior = owner.setdefault(_norm(term), canon)
            if prior != canon:
                raise SkillsDictionaryError(f"ambiguous term {term!r}: {prior!r} vs {canon!r}")
    return {k: list(v) for k, v in skills.items()}


def _term_pattern(term: str) -> str:
    parts: list[str] = []
    for ch in term:
        if ch == " ":
            parts.append(r"[\s\-]+")
        elif ch == "-":
            parts.append(r"[\s\-]?")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


@lru_cache(maxsize=None)
def _get_index(path_str: str) -> _Index:
    skills = load_skills_dict(Path(path_str))
    lookup: dict[str, str] = {}
    terms: set[str] = set()
    for canon, aliases in skills.items():
        for term in [canon, *aliases]:
            lookup[_norm(term)] = canon
            terms.add(term)
    ordered = sorted(terms, key=lambda t: (-len(t), t))
    body = "|".join(_term_pattern(t) for t in ordered)
    pattern = re.compile(r"(?<![\w.+#])(?:" + body + r")(?![\w+#])(?!\.\w)", re.IGNORECASE)
    return _Index(pattern=pattern, lookup=lookup)


def _strict_context_ok(text: str, start: int, end: int) -> bool:
    before = text[:start].rstrip(" \t")
    if before and before[-1] not in _BEFORE_OK:
        return False
    after = text[end:].lstrip(" \t")
    if not after or after[0] in _AFTER_OK:
        return True
    return after[0] == "." and (len(after) == 1 or after[1] == "\n")


_PROSE_BEFORE = re.compile(r"\b(?:in|using|with|and|via|including|plus)[ \t]+$", re.IGNORECASE)
_PROSE_PAIR = re.compile(r"(?:and|or)[ \t]+[A-Z]")


def _prose_go_ok(text: str, start: int, end: int) -> bool:
    """Accept exact 'Go' after a language-context word ("built in Go.", "with Go and Python").

    Lowercase 'go', sentence-initial 'Go to ...' and 'go' without such a lead-in stay rejected.
    """
    if not _PROSE_BEFORE.search(text[max(0, start - 16):start]):
        return False
    after = text[end:end + 12].lstrip(" \t")
    if not after or after[0] in _AFTER_OK:
        return True
    if after[0] == ".":
        return len(after) == 1 or after[1].isspace()
    return _PROSE_PAIR.match(after) is not None


def extract_skills(text: str, skills_path: Optional[Path] = None) -> list[str]:
    """Return unique canonical skills in order of first appearance."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    index = _get_index(str(Path(skills_path) if skills_path else SKILLS_DICT_PATH))
    found: dict[str, None] = {}
    for match in index.pattern.finditer(text):
        surface = match.group(0)
        key = _norm(surface)
        canon = index.lookup.get(key)
        if canon is None:
            continue
        if key in _STRICT_CONTEXT:
            if surface != _STRICT_CONTEXT[key]:
                continue
            if not (_strict_context_ok(text, match.start(), match.end())
                    or (key == "go" and _prose_go_ok(text, match.start(), match.end()))):
                continue
        elif key in _CASE_SENSITIVE and surface == surface.lower():
            continue
        found.setdefault(canon, None)
    return list(found)
