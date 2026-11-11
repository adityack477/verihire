# Architecture (implemented behaviour only)

```
PDF bytes -> extractor -> pii_scrubber -> chunker -> skills + profile_parser -> (separately) embedder
```
Runs locally; no network at runtime once the embedding model is cached.

## Modules and public interfaces
| Module | Interface |
|---|---|
| `ingest/extractor.py` | `extract_pdf(pdf_bytes) -> ExtractedDoc`; `ExtractionError(code, message)` |
| `ingest/pii_scrubber.py` | `scrub_pages(pages, use_spacy=True) -> ScrubResult(entities, spans)`; `scrub_text`, `scrub_slice` |
| `ingest/chunker.py` | `chunk_document(pages, spans_by_page) -> list[Chunk]` |
| `ingest/skills.py` | `extract_skills(text) -> list[str]`; `load_skills_dict()`; `SkillsDictionaryError` |
| `ingest/profile_parser.py` | `parse_profile(chunks, as_of=None) -> ResumeProfile` |
| `ingest/ingest.py` | `run_ingest(pdf_bytes, filename) -> (ExtractedDoc, IngestResult)` |
| `embed/embedder.py` | `get_embedder() -> Embedder`; `embed_chunks(chunks, embedder=None)` |

## Data objects
- `ExtractedDoc(pages[{page_no, text}], page_count, is_scanned, suspicious_hidden_text)`
- `IngestResult(pii, chunks, profile)`
- `PiiEntities(full_name, emails, phones, urls, other)`
- `Chunk(chunk_index, section, page_no, char_start, char_end, text_original, text_scrubbed, token_count)` + `embedding_text()` (returns `text_scrubbed`)
- `ResumeProfile(skills, titles, years_experience, education_level, education_fields, certifications, summary, parser_version)`
- `Embedder`: `name`, `dim=384`, `embed_documents(list[str])`, `embed_query(str)`

## Stages
1. **Extraction** (PyMuPDF, from bytes): page_no is 1-based; blocks sorted in reading order with simple two-column gutter detection; ligatures, bullets, whitespace and hyphenated line breaks normalised.
2. **PII scrubbing**: regex/heuristic detection of emails, URLs, phones, labelled/street/city+PIN addresses and a conservative full name (top of page 1, email local part, optional local spaCy). Placeholders `[NAME] [EMAIL] [PHONE] [URL] [ADDRESS]`. Spans are offsets into the original page text.
3. **Chunking**: heading-based sections (header, summary, experience, skills, education, projects, certifications, other); target 800-1200 chars, hard max 1500; `text_original == page_text[char_start:char_end]`; `text_scrubbed` is the same range with spans replaced. Chunks never span pages.
4. **Skills/profile**: dictionary (`skills_dict.json`) matching with boundary rules; profile built only from scrubbed chunk text.
5. **Embedding** (separate call): `BAAI/bge-small-en-v1.5` via FastEmbed, 384 dims, float32, L2-normalised; the BGE query prefix is added only in `embed_query`; documents are embedded unprefixed.

## Local runtime and model cache
Run `python -m ml.docint.embed.embedder` once with internet to cache the model in `.cache/fastembed` (or `FASTEMBED_CACHE_PATH`). Afterwards set `VERIHIRE_EMBED_OFFLINE=1` or call `get_embedder(offline=True)`. A missing model raises `ModelUnavailableError`; there is no remote fallback.

## Errors
`ExtractionError` codes: `invalid_input`, `empty`, `too_large` (>25 MB), `not_pdf`, `corrupt`, `encrypted`, `no_pages`. `run_ingest` raises `TypeError` for non-str filename (filename is metadata only). Embedder: `ModelUnavailableError`, `EmbeddingInputError`, `EmbeddingOutputError`. `SkillsDictionaryError` for a bad dictionary.

## Scanned PDFs and hidden text
- `is_scanned` is True when mean non-whitespace characters per page < 50. **No OCR is performed**; scanned PDFs yield little or no text.
- `suspicious_hidden_text` flags near-white text (unless centred on a filled non-white rectangle), fonts < 2 pt, or text outside the page. It cannot detect text hidden in images, clipping paths or behind other objects. Flagged text is kept (flag only).

## Not included
No dashboard, database, web API, LLM, job matching or ranking.