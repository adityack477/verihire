# VeriHire: Resume Ingestion and Embedding Pipeline (ML Track B)

Local, offline-at-runtime Python library that turns a resume PDF into clean, PII-scrubbed, section-aware chunks, a structured candidate profile and 384-dimensional embeddings.

**Scope:** no dashboard, frontend, database, web API, LLM, OCR, job-description comparison or candidate ranking is included here. Those belong to other components of the VeriHire project.

---

## 1. What it does

```
PDF bytes + filename
   |
   v
extractor.py      -> ExtractedDoc   (page text, is_scanned, suspicious_hidden_text)
   |
   v
pii_scrubber.py   -> PiiEntities + per-page PII spans
   |
   v
chunker.py        -> list[Chunk]    (section-aware, exact offsets, original + scrubbed text)
   |
   v
skills.py + profile_parser.py -> ResumeProfile (skills, titles, years, education, ...)
   |
   v
IngestResult(pii, chunks, profile)

Chunk.embedding_text() (scrubbed text) -> embedder.py -> 384-dim L2-normalised vectors
```

Public entry point:

```python
run_ingest(pdf_bytes: bytes, filename: str) -> tuple[ExtractedDoc, IngestResult]
```

`filename` is metadata only and is never used as a filesystem path.

---

## 2. Tested environment

| Item | Version |
|---|---|
| OS / editor | Windows, PyCharm |
| Python | 3.13.5 |
| PyMuPDF | 2.1 |
| NumPy | 2.5.4 |
| pytest | 9.2.0 |
| FastEmbed | run `pip show fastembed` and record it here |

Exact pinned versions are in `requirements.txt`. spaCy is optional and not required.

---

## 3. Setup (Windows / PyCharm)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In PyCharm: Settings > Project > Python Interpreter > select `.venv\Scripts\python.exe`, and mark the project folder as Sources Root.

### One-time model cache (internet needed once)

```powershell
python -m ml.docint.embed.embedder
```

This downloads `BAAI/bge-small-en-v1.5` into `.cache/fastembed` (or the folder in `FASTEMBED_CACHE_PATH`). After that, run fully offline:

```powershell
$env:VERIHIRE_EMBED_OFFLINE = "1"
```

If the model is not cached, the embedder raises `ModelUnavailableError`. There is no silent remote fallback, and model weights are never committed.

---

## 4. Usage

```python
from pathlib import Path
from ml.docint.ingest.ingest import run_ingest
from ml.docint.embed.embedder import get_embedder

doc, result = run_ingest(Path("my_synthetic_resume.pdf").read_bytes(), "my_synthetic_resume.pdf")

print(doc.page_count, doc.is_scanned, doc.suspicious_hidden_text)
print(result.pii)       # PiiEntities
print(result.profile)   # ResumeProfile
for c in result.chunks:
    print(c.chunk_index, c.section, c.page_no, c.char_start, c.char_end, c.text_scrubbed[:60])

embedder = get_embedder(offline=True)                                   # 384-dim, L2-normalised
vectors = embedder.embed_documents([c.embedding_text() for c in result.chunks])  # shape (n, 384)
query = embedder.embed_query("python developer")                        # shape (384,), BGE prefix added here only
```

---

## 5. Output objects

| Object | Fields |
|---|---|
| `ExtractedDoc` | `pages[{page_no, text}]`, `page_count`, `is_scanned`, `suspicious_hidden_text` |
| `IngestResult` | `pii`, `chunks`, `profile` |
| `PiiEntities` | `full_name?`, `emails[]`, `phones[]`, `urls[]`, `other[]` |
| `Chunk` | `chunk_index`, `section`, `page_no`, `char_start`, `char_end`, `text_original`, `text_scrubbed`, `token_count`; method `embedding_text()` |
| `ResumeProfile` | `skills[]` (canonical lowercase), `titles[]`, `years_experience`, `education_level` (0-4), `education_fields[]`, `certifications[]`, `summary?`, `parser_version` |
| `Embedder` | `name`, `dim` (384), `embed_documents(list[str])`, `embed_query(str)` |

These interfaces are frozen. Do not rename or remove fields.

Invariants enforced in code and tests:
- `text_original == page_text[char_start:char_end]`
- scrubbed chunk text never contains a detected email, phone or URL
- embedding dimension is exactly 384 and vectors are L2-normalised
- the BGE query prefix is applied only in `embed_query`, never to documents

---

## 6. Project layout and what each file does

```
VeriHire/
├── ml/
│   ├── docint/
│   │   ├── __init__.py
│   │   ├── ingest/
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py
│   │   │   ├── pii_scrubber.py
│   │   │   ├── chunker.py
│   │   │   ├── skills.py
│   │   │   ├── skills_dict.json
│   │   │   ├── profile_parser.py
│   │   │   └── ingest.py
│   │   └── embed/
│   │       ├── __init__.py
│   │       └── embedder.py
│   └── data_ingest/
│       ├── gold.jsonl
│       ├── evaluate.py
│       ├── build_corpus.py
│       ├── corpus.jsonl        (generated)
│       └── eval_output.txt     (generated)
├── tests/                      (7 test modules)
├── sample_resumes/README.md
├── docs/data_card.md
├── docs/architecture.md
├── report/chapters_1_to_5.md
├── requirements.txt
├── README.md
└── .gitignore
```

### 6.1 Ingestion modules (`ml/docint/ingest/`)

**`extractor.py`**: PDF text extraction with PyMuPDF.
- Opens the PDF from bytes (`extract_pdf(pdf_bytes) -> ExtractedDoc`), page by page; `page_no` is 1-based.
- Uses text blocks and spans, sorts them into reading order and detects a simple two-column gutter.
- Normalises ligatures, non-breaking/zero-width characters, bullets, whitespace and hyphenated line breaks.
- `is_scanned` is True when the mean non-whitespace characters per page is below 50. No OCR is performed.
- `suspicious_hidden_text` is True for near-white text (unless on a filled non-white rectangle), fonts smaller than 2 pt, or text outside the page. It cannot detect text hidden in images, clipping paths or behind other objects. Flagged text is kept (flag only).
- Raises `ExtractionError(code, message)` with codes `invalid_input`, `empty`, `too_large` (over 25 MB), `not_pdf`, `corrupt`, `encrypted`, `no_pages`.

**`pii_scrubber.py`**: detects and redacts personal information.
- Regex detection for emails, URLs (including LinkedIn and GitHub), Indian and international phone numbers, and addresses (labelled `Address:`, street-type lines, city + PIN).
- Conservative full-name heuristics: top of page 1, email-local-part match, optional local spaCy PERSON hint (spaCy is never required or downloaded).
- Placeholders: `[NAME]`, `[EMAIL]`, `[PHONE]`, `[URL]`, `[ADDRESS]`.
- Keeps skills, company names, titles and institutions intact.
- Returns `ScrubResult(entities, spans)`; spans are offsets into the original page text so chunking can map scrubbed text back to exact source slices. Helpers: `scrub_pages`, `scrub_text`, `scrub_slice`.
- Raw PII is never logged.

**`chunker.py`**: section-aware chunk creation (`chunk_document(pages, spans_by_page)`).
- Recognises headings for summary, experience, skills, education, projects, certifications (and an "other" group such as awards or languages). Text before the first heading is `header`.
- Targets 800-1200 characters per chunk, hard maximum 1500, splitting at paragraph or bullet boundaries and keeping a role/date header with its bullets.
- Chunks never span pages and are never empty.
- Fills every `Chunk` field, including original and scrubbed text and an approximate `token_count`.
- Short resumes naturally produce small chunks; text is never padded.

**`skills_dict.json`**: the shared skills dictionary (411 canonical lowercase skills with aliases, for example `js` to `javascript`, `k8s` to `kubernetes`). Schema: `{"schema_version": 1, "skills": {canonical: [aliases]}}`. It must be plain JSON (no comment lines). The JD parser of another component imports this same file.

**`skills.py`**: boundary-safe skill extraction (`extract_skills(text) -> list[str]`).
- Loads the dictionary relative to its own location, so it works from any working directory, and validates it (lowercase, no duplicates, no ambiguous aliases).
- Matching is case-insensitive with strict boundaries (`java` does not match `javascript`, `sql` does not match `mysql`, `c` does not match `c++`).
- Short or ambiguous tokens (`go`, `r`, `c`) match only as exact `Go`, `R`, `C` in list-like context. `Go` is also accepted after words such as in, using, with, and, via, including, plus (for example "built services in Go."). Ordinary prose such as "go to market" is not matched.
- Common-word skills (react, excel, spark, swift, ...) must not be all lowercase.
- Returns unique canonical skills in order of first appearance. No network, no LLM.

**`profile_parser.py`**: builds a `ResumeProfile` from scrubbed chunks only (`parse_profile(chunks, as_of=None)`).
- `years_experience`: date ranges found only in experience chunks (formats like `Jan 2024`, `01/2024`, `2024-01`, `2021 - 2023`, `May - June 2026`, present/current); overlaps are merged, invalid ranges are ignored and the total is capped. Pass `as_of` for reproducible results when a role is "Present". Returns `None` when no valid range exists.
- `education_level`: Diploma=1; B.Tech/BE/BSc/BS/Bachelor/BCA/BBA=2; M.Tech/MSc/MBA/MCA/Master=3; PhD=4; otherwise 0.
- Also extracts `education_fields`, `titles`, `certifications` and a short PII-scrubbed `summary` (up to 300 characters).
- Unknown information is left empty or `None`; nothing is guessed.

**`ingest.py`**: orchestration and canonical exports.
- `run_ingest(pdf_bytes, filename)` runs extract, scrub, chunk, profile in that order (scrubbing always happens before profile and embedding).
- Defines `IngestResult` and `PIPELINE_VERSION`; re-exports `ExtractedDoc`, `ExtractionError`, `PiiEntities`, `Chunk`, `ResumeProfile`.
- Raises `ExtractionError` for bad PDFs and `TypeError` if `filename` is not a string.

### 6.2 Embedding module (`ml/docint/embed/`)

**`embedder.py`**: local FastEmbed embedder for `BAAI/bge-small-en-v1.5`.
- `get_embedder(cache_dir=None, offline=None, batch_size=32) -> Embedder` (cached per configuration).
- `Embedder.embed_documents(list[str])` returns an `(n, 384)` float32 array with no prefix; `embed_query(str)` returns a `(384,)` array with the BGE query prefix.
- Validates input text and output shape/finiteness; outputs are L2-normalised.
- `embed_chunks(chunks, embedder=None)` embeds `Chunk.embedding_text()` (scrubbed text only).
- Errors: `ModelUnavailableError`, `EmbeddingInputError`, `EmbeddingOutputError`.
- Running `python -m ml.docint.embed.embedder` downloads and caches the model once.
- Environment variables: `VERIHIRE_EMBED_OFFLINE=1` (never touch the network), `FASTEMBED_CACHE_PATH` (cache location).

### 6.3 Evaluation and corpus (`ml/data_ingest/`)

**`gold.jsonl`**: 30 hand-written synthetic resume records (invented names, `@example.com` emails, fictitious phones), annotated with PII values, sections, years of experience, education level, canonical skills and a sparse-text flag. Covers multi-page, overlapping jobs, missing dates, aliases, short skill tokens, multiple degrees and sparse text. Schema and conventions: `docs/data_card.md`.

**`evaluate.py`**: validates every line of `gold.jsonl`, runs the pipeline and prints metrics computed at run time (never hard-coded): email/phone/name recall, URL recall and precision, leakage of gold PII into scrubbed text, skills precision/recall/F1, section precision/recall/F1, years MAE, education accuracy, sparse-flag accuracy, planned targets versus measured values, and per-record mismatches with `--verbose`.
- Exit codes: `0` completed, `1` invalid dataset, `2` critical failure (pipeline exception, broken chunk invariant, leaked detected PII), `3` with `--strict` if a planned target is missed.

**`build_corpus.py`**: builds `corpus.jsonl`, one JSON object per resume, for the downstream ranking component.
- Schema: `{"doc_id", "profile", "chunks", "embedding"}`. Chunks contain only scrubbed text (`text_original` is omitted unless `--include-original`).
- `--mode chunks` (default): `embedding` is a list of 384-float vectors aligned with `chunks`. `--mode mean`: one L2-normalised mean vector per resume. Confirm the expected form with the consumer.
- Renders synthetic PDFs from `gold.jsonl` by default, or reads PDFs from `--pdf-dir`. Refuses to write if raw PII leaks into scrubbed text.
- Exit codes: `0` ok, `1` invalid input, `2` pipeline/embedding/leakage failure.

**`corpus.jsonl`, `eval_output.txt`**: generated files (corpus and saved evaluation output).

### 6.4 Tests (`tests/`)

| File | Covers |
|---|---|
| `test_extractor.py` | multi-page PDFs, corrupt/encrypted input, scanned flag, two-column order, hidden-text flags |
| `test_pii_scrubber.py` | emails, Indian/international phones, URLs, names, false positives, placeholders, no raw values left |
| `test_chunker.py` | section detection, size limits, bullet boundaries, exact offsets, empty/short resumes, long multi-page resume |
| `test_skills.py` | dictionary size and validity, Java vs JavaScript, aliases, C++/C#/Node.js, `Go` handling and false positives, determinism |
| `test_profile_parser.py` | overlapping/present/malformed dates, education mapping (including BS), titles, certifications, summary PII safety |
| `test_ingest.py` | frozen contract fields, offsets, PII leakage, gold dataset validity, evaluate and build_corpus behaviour, synthetic-PDF smoke tests |
| `test_embedder.py` | dimension, normalisation, query prefix only on queries, input/output validation, offline behaviour, embedding of ingested chunks |

Ordinary tests use a fake embedding model and need no download. Real-model tests are skipped unless `VERIHIRE_RUN_MODEL_TESTS=1` and the model is cached. All test data is synthetic.

### 6.5 Documentation and config

| File | Purpose |
|---|---|
| `README.md` | this overview |
| `docs/architecture.md` | pipeline stages, interfaces, errors, local runtime |
| `docs/data_card.md` | gold dataset description, annotation rules, limitations, measured results |
| `report/chapters_1_to_5.md` | report chapters 1-5 (fill measured values from real output) |
| `sample_resumes/README.md` | explains synthetic fixtures and the rule against committing real resumes |
| `requirements.txt` | pinned dependencies (PyMuPDF, NumPy, FastEmbed, pytest; spaCy optional) |
| `.gitignore` | excludes virtual environments, caches, model files, logs, secrets and `sample_resumes/*.pdf` |

---

## 7. Running tests and evaluation

```powershell
python -m pytest tests -q

# with the real cached model
$env:VERIHIRE_EMBED_OFFLINE = "1"; $env:VERIHIRE_RUN_MODEL_TESTS = "1"
python -m pytest tests -q

# evaluation on the synthetic gold set
python -m ml.data_ingest.evaluate --verbose

# build the processed corpus (model must be cached)
python -m ml.data_ingest.build_corpus --offline
```

Latest local run: 113 tests passed. Gold-set evaluation (30 synthetic records, computed by `evaluate.py`): email recall 100%, phone recall 100%, name recall 100%, skills F1 100%, section F1 100%, years MAE 0.000, education accuracy 100%. These apply only to this small synthetic set written by the pipeline's author and are likely optimistic; they are not a claim about real resumes. Re-run `evaluate.py` and use its actual output when reporting.

---

## 8. Limitations

- No OCR: scanned PDFs produce little or no text and are only flagged via `is_scanned`.
- Hidden-text detection is partial (see `extractor.py`).
- Heuristics are English-centric; names may be missed when not near the top of page 1; city-level locations are not redacted (only labelled, street and city+PIN addresses).
- `Go`, `R` and `C` are matched only in list-like or "in/using/with Go" contexts, so some genuine mentions may be missed.
- The gold set is 30 short, clean, synthetic, single-column resumes, so the 800-1200 character chunk target is only exercised by the long synthetic test resume, not by the gold records (mean chunk length there is small).
- Years of experience come only from dated ranges in experience sections; undated roles give `None`.

## 9. Privacy

- Use synthetic data only in the repository. Never commit real resumes, model weights, API keys or secrets without authorization and consent.
- Only scrubbed text is meant to be embedded or exported. Raw PII is never logged.
- Nothing is sent to external services at runtime; the model is downloaded once, then used offline.
