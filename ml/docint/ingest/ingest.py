# ml/docint/ingest/ingest.py
"""Part 1 + Part 2 orchestration and canonical contract exports (VeriHire).

Import order (no cycles): extractor <- pii_scrubber <- chunker <- skills <-
profile_parser <- ingest. ResumeProfile is defined in profile_parser and
re-exported here, so ``from ...ingest import ResumeProfile`` keeps working.
Order of operations: extract -> PII scrub -> chunk -> profile (from scrubbed text).
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk, chunk_document
from .extractor import ExtractedDoc, ExtractionError, extract_pdf
from .pii_scrubber import PiiEntities, scrub_pages
from .profile_parser import ResumeProfile, parse_profile

PIPELINE_VERSION = "verihire-ingest-1.0"

__all__ = [
    "Chunk", "ExtractedDoc", "ExtractionError", "IngestResult", "PIPELINE_VERSION",
    "PiiEntities", "ResumeProfile", "run_ingest",
]


@dataclass(frozen=True)
class IngestResult:
    """Frozen ingest result contract."""

    pii: PiiEntities
    chunks: list[Chunk]
    profile: ResumeProfile


def run_ingest(pdf_bytes: bytes, filename: str) -> tuple[ExtractedDoc, IngestResult]:
    """Extract, scrub, chunk and profile a resume PDF.

    ``filename`` is metadata only; it is never used as a filesystem path.

    Raises:
        ExtractionError: for invalid, corrupt or encrypted PDFs.
        TypeError: if ``filename`` is not a string.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a str")
    doc = extract_pdf(pdf_bytes)
    scrub = scrub_pages(doc.pages)
    chunks = chunk_document(doc.pages, scrub.spans)
    return doc, IngestResult(pii=scrub.entities, chunks=chunks, profile=parse_profile(chunks))
