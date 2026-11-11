# Data card: `ml/data_ingest/gold.jsonl`

## Purpose
Measure the Part 1/Part 2 pipeline (PII detection, section detection, skills, years of
experience, education level, sparse-page flag) with `python -m ml.data_ingest.evaluate`.

## Size and source
30 records, one JSON object per line. **100% synthetic**: hand-written resume text with
invented names, `@example.com` emails and fictitious phones. No real resumes or personal data.

## Fields
| Field | Meaning |
|---|---|
| `id` | unique id (`G01`..`G30`) |
| `tags` | free-text edge-case labels |
| `as_of` | date used to resolve "Present" in experience |
| `pages` | list of page texts as the extractor would emit them (page_no = index + 1) |
| `pii` | `full_name` (str/null), `emails`, `phones`, `urls` (surface strings exactly as in `pages`) |
| `sections` | labelled sections expected: summary, experience, skills, education, projects, certifications |
| `years_experience` | merged months of dated experience / 12, 1 decimal; `null` = no dated experience |
| `education_level` | 0-4 (Diploma=1; B.Tech/BE/BSc/Bachelor/BCA/BBA=2; M.Tech/MSc/MBA/MCA=3; PhD=4) |
| `skills` | canonical lowercase names from `skills_dict.json` |
| `sparse` | expected sparse/scanned flag (mean non-space chars per page < 50) |

## Annotation process and conventions
- Annotations were written by hand alongside the text, independently of pipeline output.
- PII is annotated as **values**, not character offsets; each value must occur in `pages` (validated on load).
- `skills`: every canonical skill explicitly named anywhere in the text, aliases included
  (e.g. "Google Data Analytics" -> `data analysis`; degree names such as "Data Science" count).
- Years follow the parser's documented rules (year-only dates mid-year, overlaps merged, present = `as_of`).
- Addresses are not annotated.
- Deliberately hard cases are included (G23: "Go" in prose, `BS` degree) and can lower measured scores.

## Edge cases covered
Multi-page (G07, G30), overlapping/contiguous jobs (G02, G07, G26), missing dates (G03), aliases (G04),
short tokens (G05, G23), multiple education entries (G06, G19), sparse text (G08, G09), heading variants (G29).

## Intended use
Regression checking and a sanity-level estimate during development.

## Limitations and bias
- 30 short, clean, English, single-column text resumes: not representative of real resumes
  (layouts, languages, scanned files, non-Indian/US naming and phone formats are barely covered).
- Names and formats skew to Indian and US conventions; skills skew to software/data roles.
- Written by the same author as the parser, so scores are likely optimistic.
- Metrics from 30 records have wide uncertainty; do not present them as real-world performance.

## Privacy
Contains no real personal data. Do not add real resumes to this file.