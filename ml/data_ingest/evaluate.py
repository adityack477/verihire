"""Evaluate the VeriHire ingest pipeline on the synthetic gold set (Part 3).

Usage (project root):
    python -m ml.data_ingest.evaluate [--gold PATH] [--expected-count 30]
                                      [--as-of YYYY-MM-DD] [--verbose] [--strict]

Exit status:
    0  evaluation completed
    1  dataset invalid (malformed JSON, schema error, wrong record count)
    2  critical failure (pipeline exception, chunk offset/size invariant broken,
       or a DETECTED email/phone/URL still present in scrubbed text)
    3  only with --strict: a planned target is not met

Every number is computed from gold.jsonl at run time. Schema: docs/data_card.md.
Years of experience are evaluated with a fixed ``as_of`` date (record ``as_of``
or ``--as-of``) so "present" roles are reproducible.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ml.docint.ingest.chunker import HARD_MAX, chunk_document
from ml.docint.ingest.extractor import MIN_CHARS_PER_PAGE
from ml.docint.ingest.pii_scrubber import scrub_pages
from ml.docint.ingest.profile_parser import parse_profile
from ml.docint.ingest.skills import SkillsDictionaryError, load_skills_dict

DEFAULT_GOLD = Path(__file__).resolve().with_name("gold.jsonl")
EXPECTED_RECORDS = 30
DEFAULT_AS_OF = "2026-01-01"
VALID_SECTIONS = frozenset({"summary", "experience", "skills", "education", "projects", "certifications"})
UNLABELLED_SECTIONS = frozenset({"header", "other"})
_REQUIRED = ("id", "pages", "pii", "sections", "years_experience", "education_level", "skills", "sparse")

TARGETS = {  # planned targets only; compared with measured values, never assumed
    "Email recall": (0.99, True),
    "Phone recall": (0.99, True),
    "Name recall": (0.90, True),
    "Years-of-experience MAE": (1.0, False),
    "Education-level accuracy": (0.90, True),
}


# ----------------------------------------------------------------- dataset --
def validate_record(obj: Any, known_skills: set[str]) -> list[str]:
    """Return schema problems for one gold record (never echoes PII values)."""
    if not isinstance(obj, dict):
        return ["record is not a JSON object"]
    errs = [f"missing key '{k}'" for k in _REQUIRED if k not in obj]
    if errs:
        return errs
    if not isinstance(obj["id"], str) or not obj["id"]:
        errs.append("id must be a non-empty string")
    pages = obj["pages"]
    if not (isinstance(pages, list) and pages and all(isinstance(p, str) for p in pages)):
        errs.append("pages must be a non-empty list of strings")
        pages = []
    text = "\n".join(pages)
    pii = obj["pii"]
    if not isinstance(pii, dict):
        errs.append("pii must be an object")
    else:
        name = pii.get("full_name")
        if name is not None and not (isinstance(name, str) and name):
            errs.append("pii.full_name must be a non-empty string or null")
        elif name and name not in text:
            errs.append("pii.full_name not found in pages")
        for key in ("emails", "phones", "urls"):
            vals = pii.get(key)
            if not (isinstance(vals, list) and all(isinstance(v, str) and v for v in vals)):
                errs.append(f"pii.{key} must be a list of non-empty strings")
            else:
                errs.extend(f"pii.{key}[{i}] not found in pages" for i, v in enumerate(vals) if v not in text)
    secs = obj["sections"]
    if not (isinstance(secs, list) and all(s in VALID_SECTIONS for s in secs)):
        errs.append(f"sections must be a list drawn from {sorted(VALID_SECTIONS)}")
    years = obj["years_experience"]
    if years is not None and (isinstance(years, bool) or not isinstance(years, (int, float))
                              or not 0 <= years <= 45):
        errs.append("years_experience must be null or a number in [0, 45]")
    edu = obj["education_level"]
    if isinstance(edu, bool) or not isinstance(edu, int) or not 0 <= edu <= 4:
        errs.append("education_level must be an integer 0-4")
    skills = obj["skills"]
    if not (isinstance(skills, list) and all(isinstance(s, str) for s in skills)):
        errs.append("skills must be a list of strings")
    else:
        errs.extend(f"skills[{i}] is not a canonical skill in skills_dict.json"
                    for i, s in enumerate(skills) if s not in known_skills)
    if not isinstance(obj["sparse"], bool):
        errs.append("sparse must be a boolean")
    if "as_of" in obj:
        try:
            date.fromisoformat(str(obj["as_of"]))
        except ValueError:
            errs.append("as_of must be an ISO date")
    return errs


def load_gold(path: Path, expected_count: int = EXPECTED_RECORDS) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and validate every JSONL line. Returns (valid_records, error_messages)."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"cannot read '{Path(path).name}': {exc.strerror or exc}"]
    try:
        known = set(load_skills_dict())
    except SkillsDictionaryError as exc:
        return [], [f"skills dictionary invalid: {exc}"]
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"line {n}: blank line")
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            errors.append(f"line {n}: invalid JSON ({exc})")
            continue
        problems = validate_record(obj, known)
        if problems:
            errors.extend(f"line {n}: {p}" for p in problems)
            continue
        if obj["id"] in seen:
            errors.append(f"line {n}: duplicate id '{obj['id']}'")
            continue
        seen.add(obj["id"])
        records.append(obj)
    if len(records) != expected_count:
        errors.append(f"expected {expected_count} valid records, found {len(records)}")
    return records, errors


# -------------------------------------------------------------- evaluation --
def _run_pipeline(rec: dict[str, Any], default_as_of: str):
    pages = [{"page_no": i + 1, "text": t} for i, t in enumerate(rec["pages"])]
    scrub = scrub_pages(pages, use_spacy=False)
    chunks = chunk_document(pages, scrub.spans)
    as_of = date.fromisoformat(str(rec.get("as_of") or default_as_of))
    return pages, scrub, chunks, parse_profile(chunks, as_of=as_of)


def _check_chunks(pages: list[dict[str, Any]], chunks: Sequence[Any]) -> list[str]:
    issues: list[str] = []
    for i, ch in enumerate(chunks):
        if ch.chunk_index != i:
            issues.append(f"chunk {i}: chunk_index {ch.chunk_index}")
        if not 1 <= ch.page_no <= len(pages):
            issues.append(f"chunk {i}: invalid page_no")
            continue
        text = pages[ch.page_no - 1]["text"]
        if not 0 <= ch.char_start < ch.char_end <= len(text):
            issues.append(f"chunk {i}: invalid offsets")
        elif text[ch.char_start:ch.char_end] != ch.text_original:
            issues.append(f"chunk {i}: text_original is not the page slice")
        if not ch.text_scrubbed.strip():
            issues.append(f"chunk {i}: empty scrubbed text")
        if len(ch.text_original) > HARD_MAX or len(ch.text_scrubbed) > HARD_MAX:
            issues.append(f"chunk {i}: exceeds {HARD_MAX} chars")
    return issues


_PII_KEYS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("emails", lambda v: v.casefold()),
    ("phones", lambda v: re.sub(r"\D", "", v)),
    ("urls", lambda v: v.casefold().rstrip(".,;:!?)")),
)


def evaluate(records: Sequence[dict[str, Any]], default_as_of: str = DEFAULT_AS_OF) -> dict[str, Any]:
    """Run the pipeline on each record and accumulate raw counts (no thresholds here)."""
    c: Counter[str] = Counter()
    sk_fn: Counter[str] = Counter()
    sk_fp: Counter[str] = Counter()
    abs_err: list[float] = []
    critical: list[str] = []
    mism: list[str] = []
    chunk_counts: list[int] = []
    chunk_lens: list[int] = []

    for rec in records:
        rid = rec["id"]
        try:
            pages, scrub, chunks, profile = _run_pipeline(rec, default_as_of)
        except Exception as exc:  # reported, never swallowed
            critical.append(f"{rid}: pipeline raised {type(exc).__name__}: {exc}")
            continue
        c["records_ok"] += 1
        c["pages"] += len(pages)
        c["chunks"] += len(chunks)
        chunk_counts.append(len(chunks))
        chunk_lens.extend(len(ch.text_original) for ch in chunks)
        critical.extend(f"{rid}: {p}" for p in _check_chunks(pages, chunks))

        ent, gold = scrub.entities, rec["pii"]
        # name
        if gold["full_name"] is not None:
            c["name_den"] += 1
            if ent.full_name and ent.full_name.casefold() == gold["full_name"].casefold():
                c["name_hit"] += 1
            else:
                mism.append(f"{rid}: name not detected")
        elif ent.full_name is not None:
            c["name_spurious"] += 1
        # emails / phones / urls
        for key, norm in _PII_KEYS:
            g = {norm(v) for v in gold[key]}
            d = {norm(v) for v in getattr(ent, key)}
            c[f"{key}_den"] += len(g)
            c[f"{key}_hit"] += len(g & d)
            c[f"{key}_det"] += len(d)
            if g - d:
                mism.append(f"{rid}: {len(g - d)} gold {key} not detected")
            if d - g:
                mism.append(f"{rid}: {len(d - g)} extra {key} detected")
        # leakage into scrubbed chunks
        low = "\n".join(ch.text_scrubbed for ch in chunks).casefold()
        for key, _ in _PII_KEYS:
            c[f"gold_{key}_leak"] += sum(1 for v in gold[key] if v.casefold() in low)
            critical.extend(f"{rid}: detected {key} value present in scrubbed text"
                            for raw in getattr(ent, key) if raw.casefold() in low)
        if gold["full_name"] and gold["full_name"].casefold() in low:
            c["gold_name_leak"] += 1
        # skills
        g_sk, p_sk = set(rec["skills"]), set(profile.skills)
        c["sk_tp"] += len(g_sk & p_sk)
        c["sk_fp"] += len(p_sk - g_sk)
        c["sk_fn"] += len(g_sk - p_sk)
        sk_fn.update(g_sk - p_sk)
        sk_fp.update(p_sk - g_sk)
        c["sk_exact"] += int(g_sk == p_sk)
        if g_sk != p_sk:
            mism.append(f"{rid}: skills missing={sorted(g_sk - p_sk)} extra={sorted(p_sk - g_sk)}")
        # sections
        e_sec = set(rec["sections"])
        d_sec = {ch.section for ch in chunks} - UNLABELLED_SECTIONS
        c["sec_tp"] += len(e_sec & d_sec)
        c["sec_fp"] += len(d_sec - e_sec)
        c["sec_fn"] += len(e_sec - d_sec)
        if e_sec != d_sec:
            mism.append(f"{rid}: sections missing={sorted(e_sec - d_sec)} extra={sorted(d_sec - e_sec)}")
        # years
        g_y, p_y = rec["years_experience"], profile.years_experience
        if g_y is None:
            c["years_none_den"] += 1
            if p_y is None:
                c["years_none_ok"] += 1
            else:
                mism.append(f"{rid}: years expected null, got {p_y}")
        else:
            c["years_den"] += 1
            if p_y is None:
                c["years_missing"] += 1
                mism.append(f"{rid}: years expected {g_y}, got null")
            else:
                abs_err.append(abs(p_y - g_y))
                if abs(p_y - g_y) > 0.05:
                    mism.append(f"{rid}: years expected {g_y}, got {p_y}")
        # education
        c["edu_den"] += 1
        c["edu_hit"] += int(profile.education_level == rec["education_level"])
        if profile.education_level != rec["education_level"]:
            mism.append(f"{rid}: education expected {rec['education_level']}, got {profile.education_level}")
        # sparse-page flag (same rule as extractor: mean non-space chars per page < MIN_CHARS_PER_PAGE)
        chars = sum(len(re.sub(r"\s", "", p)) for p in rec["pages"])
        pred_sparse = (chars / len(rec["pages"])) < MIN_CHARS_PER_PAGE
        c["sparse_den"] += 1
        c["sparse_hit"] += int(pred_sparse == rec["sparse"])
        if pred_sparse != rec["sparse"]:
            mism.append(f"{rid}: sparse expected {rec['sparse']}, got {pred_sparse}")

    return {"counts": c, "sk_fn": sk_fn, "sk_fp": sk_fp, "abs_err": abs_err, "critical": critical,
            "mismatches": mism, "chunk_counts": chunk_counts, "chunk_lens": chunk_lens}


# ----------------------------------------------------------------- report --
def _ratio(num: float, den: float) -> Optional[float]:
    return None if den == 0 else num / den


def _pct(v: Optional[float]) -> str:
    return "undefined" if v is None else f"{v * 100:.1f}%"


def _row(label: str, num: int, den: int) -> str:
    note = "  (undefined: 0 gold items)" if den == 0 else ""
    return f"  {label:<26} {num:>4}/{den:<4} {_pct(_ratio(num, den)):>10}{note}"


def _prf(tp: int, fp: int, fn: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f = None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f


def print_report(rep: dict[str, Any], records: Sequence[dict[str, Any]], verbose: bool) -> dict[str, Optional[bool]]:
    """Print all metrics; returns {target label: True/False/None(undefined)}."""
    c = rep["counts"]
    print("=" * 66)
    print("VeriHire evaluation on synthetic gold set (computed at run time)")
    print("=" * 66)
    print(f"Records: {len(records)} loaded, {c['records_ok']} processed | pages: {c['pages']} | chunks: {c['chunks']}")
    cc, cl = rep["chunk_counts"], rep["chunk_lens"]
    if cc:
        print(f"Chunks/record: min {min(cc)} mean {sum(cc) / len(cc):.1f} max {max(cc)} | "
              f"chunk length: min {min(cl)} mean {sum(cl) / len(cl):.0f} max {max(cl)} (hard max {HARD_MAX})")
    print("\nGold label coverage (records WITHOUT a label of this kind):")
    print(f"  full_name: {sum(1 for r in records if r['pii']['full_name'] is None)} | "
          f"emails: {sum(1 for r in records if not r['pii']['emails'])} | "
          f"phones: {sum(1 for r in records if not r['pii']['phones'])} | "
          f"urls: {sum(1 for r in records if not r['pii']['urls'])} | "
          f"sections: {sum(1 for r in records if not r['sections'])} | "
          f"skills: {sum(1 for r in records if not r['skills'])} | "
          f"years (null): {sum(1 for r in records if r['years_experience'] is None)}")

    print("\nPII detection")
    print(_row("Email recall", c["emails_hit"], c["emails_den"]))
    print(_row("Phone recall", c["phones_hit"], c["phones_den"]))
    print(_row("Name recall", c["name_hit"], c["name_den"]))
    print(_row("URL recall (info)", c["urls_hit"], c["urls_den"]))
    print(_row("Email precision (info)", c["emails_hit"], c["emails_det"]))
    print(_row("Phone precision (info)", c["phones_hit"], c["phones_det"]))
    print(_row("URL precision (info)", c["urls_hit"], c["urls_det"]))
    print(f"  names detected where gold has none: {c['name_spurious']}")
    print("  gold values still present in scrubbed chunks (leakage): "
          f"emails {c['gold_emails_leak']}, phones {c['gold_phones_leak']}, "
          f"urls {c['gold_urls_leak']}, names {c['gold_name_leak']}")

    p, r, f = _prf(c["sk_tp"], c["sk_fp"], c["sk_fn"])
    print("\nSkills (micro-averaged over all records)")
    print(f"  TP {c['sk_tp']}  FP {c['sk_fp']}  FN {c['sk_fn']}")
    print(f"  Precision {_pct(p)} | Recall {_pct(r)} | F1 {_pct(f)}")
    print(_row("Exact-set records", c["sk_exact"], c["records_ok"]))
    if rep["sk_fn"]:
        print("  most missed:", ", ".join(f"{k}({v})" for k, v in rep["sk_fn"].most_common(10)))
    if rep["sk_fp"]:
        print("  most spurious:", ", ".join(f"{k}({v})" for k, v in rep["sk_fp"].most_common(10)))

    p, r, f = _prf(c["sec_tp"], c["sec_fp"], c["sec_fn"])
    print("\nSection detection (labelled sections only; header/other excluded)")
    print(f"  TP {c['sec_tp']}  FP {c['sec_fp']}  FN {c['sec_fn']}")
    print(f"  Precision {_pct(p)} | Recall {_pct(r)} | F1 {_pct(f)}")

    abs_err = rep["abs_err"]
    mae = (sum(abs_err) / len(abs_err)) if abs_err else None
    print("\nProfile")
    print(f"  Years-of-experience MAE   {'undefined (no comparable records)' if mae is None else f'{mae:.3f} years'}"
          f"  (n={len(abs_err)}, predicted null where gold has a value: {c['years_missing']})")
    print(_row("Null-years correct", c["years_none_ok"], c["years_none_den"]))
    print(_row("Education-level accuracy", c["edu_hit"], c["edu_den"]))
    print(_row("Sparse-flag accuracy", c["sparse_hit"], c["sparse_den"]))

    measured = {
        "Email recall": _ratio(c["emails_hit"], c["emails_den"]),
        "Phone recall": _ratio(c["phones_hit"], c["phones_den"]),
        "Name recall": _ratio(c["name_hit"], c["name_den"]),
        "Years-of-experience MAE": mae,
        "Education-level accuracy": _ratio(c["edu_hit"], c["edu_den"]),
    }
    status: dict[str, Optional[bool]] = {}
    print("\nPlanned targets vs measured (30 synthetic records; not a claim about real resumes)")
    for label, (thr, higher) in TARGETS.items():
        v = measured[label]
        ok = None if v is None else (v >= thr if higher else v <= thr)
        status[label] = ok
        shown = "undefined" if v is None else (f"{v:.3f}" if not higher else _pct(v))
        goal = f">= {thr * 100:.0f}%" if higher else f"<= {thr}"
        print(f"  {label:<26} target {goal:<8} measured {shown:<10} -> "
              f"{'undefined' if ok is None else 'met' if ok else 'NOT met'}")

    if verbose and rep["mismatches"]:
        print("\nPer-record mismatches")
        for line in rep["mismatches"]:
            print("  -", line)
    return status


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate VeriHire ingest on gold.jsonl")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--expected-count", type=int, default=EXPECTED_RECORDS)
    ap.add_argument("--as-of", default=DEFAULT_AS_OF)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 3 if any planned target is not met")
    args = ap.parse_args(argv)
    try:
        date.fromisoformat(args.as_of)
    except ValueError:
        print("invalid --as-of (use YYYY-MM-DD)")
        return 1
    records, errors = load_gold(Path(args.gold), args.expected_count)
    if errors:
        print("DATASET INVALID:")
        for e in errors[:50]:
            print("  -", e)
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more")
        return 1
    rep = evaluate(records, args.as_of)
    status = print_report(rep, records, args.verbose)
    if rep["critical"]:
        print("\nCRITICAL FAILURES:")
        for line in rep["critical"][:50]:
            print("  -", line)
        return 2
    if args.strict and any(v is False for v in status.values()):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())