"""Build a JSONL corpus from synthetic resumes or a directory of PDFs.

The record shape in this module is a proposal and should be confirmed with the
consuming team before downstream adoption. Only scrubbed text is included by
default; ``--include-original`` explicitly opts into raw text.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from ml.data_ingest.evaluate import load_gold
from ml.docint.embed.embedder import EMBEDDING_DIM, EmbedderError, embed_chunks, get_embedder
from ml.docint.ingest.extractor import ExtractionError
from ml.docint.ingest.ingest import PIPELINE_VERSION, run_ingest

DEFAULT_GOLD = Path(__file__).resolve().with_name("gold.jsonl")
DEFAULT_OUT = Path(__file__).resolve().with_name("corpus.jsonl")
_CHUNK_FIELDS = ("chunk_index", "section", "page_no", "char_start", "char_end",
                 "text_scrubbed", "token_count")


class CorpusError(Exception):
    """Invalid input or broken invariant while building the corpus."""


def render_pdf(pages: Sequence[str]) -> bytes:
    """Render synthetic page texts into PDF bytes (one text page per string)."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        for i, line in enumerate(text.split("\n")):
            if line.strip():
                page.insert_text((50, 60 + 13 * i), line, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def iter_sources(pdf_dir: Optional[str], gold: Path, limit: Optional[int]) -> Iterator[tuple[str, bytes]]:
    """Yield sources in deterministic order."""
    if pdf_dir:
        files = sorted(Path(pdf_dir).glob("*.pdf"))
        if not files:
            raise CorpusError("no PDF files found in --pdf-dir")
        for path in files[:limit]:
            yield path.name, path.read_bytes()
        return
    records, errors = load_gold(gold)
    if errors:
        raise CorpusError("gold dataset invalid: " + "; ".join(errors[:3]))
    for rec in records[:limit]:
        yield f"{rec['id']}.pdf", render_pdf(rec["pages"])


def _leaks(pii: Any, chunks: Sequence[Any]) -> list[str]:
    low = "\n".join(c.text_scrubbed for c in chunks).casefold()
    return [key for key in ("emails", "phones", "urls")
            for value in getattr(pii, key) if value.casefold() in low]


def build_records(sources: Iterable[tuple[str, bytes]], embedder: Any, mode: str = "chunks",
                  include_original: bool = False) -> list[dict[str, Any]]:
    """Run ingestion and embeddings for each source."""
    if mode not in ("chunks", "mean"):
        raise ValueError("mode must be 'chunks' or 'mean'")
    fields = _CHUNK_FIELDS + (("text_original",) if include_original else ())
    records: list[dict[str, Any]] = []
    for index, (name, data) in enumerate(sources, start=1):
        doc_id = f"candidate_{index:03d}"
        _, result = run_ingest(data, name)
        leaked = _leaks(result.pii, result.chunks)
        if leaked:
            raise CorpusError(f"{doc_id}: raw {', '.join(sorted(set(leaked)))} present in scrubbed text")
        vectors = (embed_chunks(result.chunks, embedder) if result.chunks
                   else np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
        if vectors.shape != (len(result.chunks), EMBEDDING_DIM):
            raise CorpusError(f"{doc_id}: unexpected embedding shape {vectors.shape}")
        if mode == "mean":
            if len(vectors):
                mean = vectors.mean(axis=0)
                norm = np.linalg.norm(mean)
                embedding = np.round(mean / norm, 6).tolist() if norm else []
            else:
                embedding = []
        else:
            embedding = np.round(vectors, 6).tolist()
        records.append({
            "doc_id": doc_id,
            "profile": asdict(result.profile),
            "chunks": [{f: getattr(c, f) for f in fields} for c in result.chunks],
            "embedding": embedding,
        })
    return records


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    """Write JSONL using an atomic replace."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build corpus.jsonl from synthetic resumes")
    ap.add_argument("--pdf-dir")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mode", choices=("chunks", "mean"), default="chunks")
    ap.add_argument("--include-original", action="store_true", help="also write text_original (raw PII)")
    ap.add_argument("--offline", action="store_true", help="never touch the network (model must be cached)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    try:
        sources = list(iter_sources(args.pdf_dir, Path(args.gold), args.limit))
    except CorpusError as exc:
        print(f"INVALID INPUT: {exc}")
        return 1
    try:
        embedder = get_embedder(offline=True if args.offline else None)
        records = build_records(sources, embedder, args.mode, args.include_original)
        write_jsonl(records, Path(args.out))
    except (CorpusError, ExtractionError, EmbedderError) as exc:
        print(f"FAILED: {exc}")
        return 2
    print(f"wrote {len(records)} records -> {args.out} "
          f"(pipeline {PIPELINE_VERSION}, mode {args.mode}, dim {EMBEDDING_DIM})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
