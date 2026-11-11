# VeriHire: resume ingestion library (Parts 1-3)

Local, offline-at-runtime pipeline: PDF bytes -> text extraction -> PII scrubbing -> section-aware chunks -> skills/profile -> BGE embeddings.
**There is no dashboard, frontend, database, web API, LLM or job matching in this project.**

## Tested versions
Python 3.13.5 (Windows, PyCharm). PyMuPDF 2.1, NumPy 2.5.4, pytest 9.2.0. FastEmbed: run `pip show fastembed` and record the version here. Optional spaCy not used.

## Setup (Windows / PyCharm terminal, project root)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
In PyCharm: Settings > Project > Python Interpreter > select `.venv\Scripts\python.exe`; mark `VeriHire` as Sources Root.

## One-time model cache (internet needed once)
```powershell
python -m ml.docint.embed.embedder
$env:VERIHIRE_EMBED_OFFLINE = "1"   # later runs never touch the network
```
Cache: `.cache/fastembed` (or `FASTEMBED_CACHE_PATH`). Never commit it.

## Usage
```python
from pathlib import Path
from ml.docint.ingest.ingest import run_ingest
from ml.docint.embed.embedder import get_embedder

doc, result = run_ingest(Path("sample_resumes/synthetic.pdf").read_bytes(), "synthetic.pdf")
print(doc.page_count, doc.is_scanned, doc.suspicious_hidden_text)
print(result.pii, result.profile)

embedder = get_embedder(offline=True)                       # 384-dim, L2-normalised
vectors = embedder.embed_documents([c.embedding_text() for c in result.chunks])  # (n, 384)
query = embedder.embed_query("python developer")            # (384,), query prefix added here only
```

## Output objects
`ExtractedDoc`, `IngestResult(pii, chunks, profile)`, `PiiEntities`, `Chunk`, `ResumeProfile`, `Embedder` (see `docs/architecture.md`). Embed only `Chunk.embedding_text()` (scrubbed text).

## Paths
`ml/docint/ingest/skills_dict.json` (skills), `ml/data_ingest/gold.jsonl` (synthetic gold set), `.cache/fastembed` (model cache, git-ignored).

## Tests and evaluation
```powershell
python -m pytest tests -q
$env:VERIHIRE_EMBED_OFFLINE="1"; $env:VERIHIRE_RUN_MODEL_TESTS="1"; python -m pytest tests -q   # real cached model
python -m ml.data_ingest.evaluate --verbose
```
`evaluate` exits 1 for an invalid dataset, 2 for a critical failure, 3 with `--strict` if a planned target is missed.

## Limitations
No OCR (scanned PDFs give little text); hidden-text detection is partial; English-centric heuristics; `Go`/`R`/`C` only matched in list-like context; `BS` degrees and city-level locations not handled; evaluation set is 30 synthetic records and not representative.

## Privacy
Never commit real resumes, model weights or secrets. Use synthetic data. Raw PII is never logged by the library; scrubbed text is the only text meant for embedding. Nothing is sent to external services at runtime.