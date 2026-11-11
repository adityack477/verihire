# VeriHire: Resume Document Intelligence (Part 1-3 scope)

## Chapter 1: Introduction
**1.1 Problem.** Resumes arrive as PDFs with varied layouts and contain personal data. Downstream analysis needs clean text, protected PII and structured fields.
**1.2 Objectives.** From PDF bytes produce (a) page-wise text with quality flags, (b) PII-scrubbed section-aware chunks, (c) a structured `ResumeProfile`, (d) local 384-dimensional embeddings.
**1.3 Scope.** Local/offline Python 3.13 library. Excluded: dashboard/frontend, database, external APIs, LLMs, OCR, job-description comparison, ranking, hiring scores.
**1.4 Report structure.** Ch. 2 background, Ch. 3 requirements/design, Ch. 4 implementation, Ch. 5 testing and evaluation.

## Chapter 2: Background
PDF text extraction (PyMuPDF block/span data), rule-based PII detection (regular expressions and heuristics, optional local spaCy), section-aware chunking, dictionary-based skill extraction, and sentence embeddings with `BAAI/bge-small-en-v1.5` executed through FastEmbed (ONNX, CPU). BGE models use an instruction prefix for queries only. No external literature is cited in this draft; add references from the project plan.

## Chapter 3: Requirements and design
Frozen contract: `run_ingest(pdf_bytes, filename) -> (ExtractedDoc, IngestResult)` with the field lists in `docs/architecture.md`. Design rules: deterministic behaviour, typed exceptions, scrubbing before profile summary and embedding, offsets that keep `text_original` an exact source slice, no raw PII in logs or embeddings input. Pipeline: extraction -> PII scrubbing -> chunking -> skills/profile -> embeddings.

## Chapter 4: Implementation
Modules are summarised in `docs/architecture.md`. Key behaviours: reading-order sorting with gutter detection; sparse-page rule (< 50 chars/page) and hidden-text flags; priority-based span resolution (email > URL > phone > address > name); heading dictionary and size limits (800-1200 target, 1500 max) for chunks; a 411-skill dictionary with alias and boundary rules (short tokens `go`, `r`, `c` need exact case and list-like context); years of experience from merged date ranges in experience chunks only; education level from a fixed mapping; FastEmbed embedder with L2 normalisation and query-only prefix.

## Chapter 5: Testing and evaluation
**5.1 Unit/integration tests.** Seven pytest modules with synthetic inputs; embedder tests use a fake model, real-model tests run only with `VERIHIRE_RUN_MODEL_TESTS=1`. On the author's Windows machine (Python 3.13.5): 94 passed, 3 skipped without the model flag; 97 passed with the cached model and the flag.
**5.2 Gold evaluation.** 30 synthetic records (`docs/data_card.md`), run with `python -m ml.data_ingest.evaluate`. Paste the actual console output below; do not edit values.

| Metric | Planned target | Measured |
|---|---|---|
| Email recall | >= 99% | `<paste>` |
| Phone recall | >= 99% | `<paste>` |
| Name recall | >= 90% | `<paste>` |
| Years-of-experience MAE | <= 1.0 | `<paste>` |
| Education-level accuracy | >= 90% | `<paste>` |
| Skills P / R / F1 | none set | `<paste>` |
| Section P / R / F1 | none set | `<paste>` |

Targets are plan values. Measured values apply only to the synthetic set, which is small, written by the implementer and not representative of real resumes; the project plan's 30-labelled-resume target is therefore not claimed as met on real data.
**5.3 Known limitations.** No OCR; hidden-text detection is partial; `Go` in running prose and `BS` degrees are not recognised; city-level locations are not redacted; English-centric heuristics; names may be missed when not at the top of page 1.
**5.4 Conclusion.** State conclusions only after pasting the measured results.