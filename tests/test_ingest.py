# tests/test_embedder.py  (Part 2 embedder tests + Part 1 -> Part 2 integration)
import os
import zlib

import numpy as np
import pytest
from pathlib import Path

from ml.data_ingest import evaluate
from ml.docint.ingest.skills import SKILLS_DICT_PATH



try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from ml.docint.embed import embedder as emb_mod
from ml.docint.embed.embedder import (
    EMBEDDING_DIM, MODEL_NAME, OFFLINE_ENV_VAR, QUERY_PREFIX, Embedder, EmbeddingInputError,
    EmbeddingOutputError, ModelUnavailableError, embed_chunks, get_embedder,
)
from ml.docint.ingest.chunker import Chunk
from ml.docint.ingest.ingest import run_ingest


class FakeModel:
    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim
        self.calls = []

    def embed(self, documents, batch_size=256, **kwargs):
        docs = list(documents)
        self.calls.append((docs, batch_size))
        for text in docs:
            rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
            yield rng.normal(size=self.dim).astype(np.float32) * 3.0


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    monkeypatch.setattr(emb_mod, "_CACHE", {})


# ---------- Part 2: embedder ----------

def test_name_and_dim():
    e = Embedder(FakeModel())
    assert e.name == MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert e.dim == EMBEDDING_DIM == 384


def test_documents_shape_dtype_and_l2_norm():
    out = Embedder(FakeModel()).embed_documents(["alpha", "beta", "gamma"])
    assert out.shape == (3, 384) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_empty_documents():
    assert Embedder(FakeModel()).embed_documents([]).shape == (0, 384)


def test_query_prefix_only_on_query():
    fake = FakeModel()
    e = Embedder(fake)
    vec = e.embed_query("python developer")
    assert vec.shape == (384,) and np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
    assert fake.calls[-1][0] == [QUERY_PREFIX + "python developer"]
    e.embed_documents(["a resume chunk"])
    assert fake.calls[-1][0] == ["a resume chunk"]
    assert not any(d.startswith(QUERY_PREFIX) for docs, _ in fake.calls[1:] for d in docs)


def test_deterministic_and_batch_size_passed():
    fake = FakeModel()
    e = Embedder(fake, batch_size=2)
    texts = ["a", "b", "c", "d", "e"]
    first, second = e.embed_documents(texts), e.embed_documents(texts)
    assert np.array_equal(first, second)
    assert fake.calls[-1][1] == 2
    assert np.allclose(first, Embedder(FakeModel(), batch_size=5).embed_documents(texts))


@pytest.mark.parametrize("bad", ["just a string", [""], ["   "], [123], None])
def test_invalid_document_input(bad):
    with pytest.raises(EmbeddingInputError):
        Embedder(FakeModel()).embed_documents(bad)


def test_invalid_query_input():
    with pytest.raises(EmbeddingInputError):
        Embedder(FakeModel()).embed_query("  ")


def test_wrong_output_dimension_rejected():
    with pytest.raises(EmbeddingOutputError):
        Embedder(FakeModel(dim=10)).embed_documents(["x"])


def test_get_embedder_uses_local_loader_and_offline_env(monkeypatch, tmp_path):
    seen = {}

    def fake_loader(cache_dir, offline):
        seen["args"] = (cache_dir, offline)
        return FakeModel()

    monkeypatch.setattr(emb_mod, "_load_model", fake_loader)
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
    e = get_embedder(cache_dir=str(tmp_path))
    assert e.dim == 384 and seen["args"] == (str(tmp_path), True)
    assert get_embedder(cache_dir=str(tmp_path)) is e


def test_missing_model_raises_without_fallback(monkeypatch, tmp_path):
    def boom(cache_dir, offline):
        raise ModelUnavailableError("not cached")

    monkeypatch.setattr(emb_mod, "_load_model", boom)
    with pytest.raises(ModelUnavailableError):
        get_embedder(cache_dir=str(tmp_path), offline=True)


def test_embed_chunks_uses_scrubbed_text_only():
    fake = FakeModel()
    chunk = Chunk(chunk_index=0, section="header", page_no=1, char_start=0, char_end=12,
                  text_original="secret@x.com", text_scrubbed="[EMAIL]", token_count=3)
    out = embed_chunks([chunk], Embedder(fake))
    assert out.shape == (1, 384)
    assert fake.calls[-1][0] == ["[EMAIL]"]


# ---------- Part 1 -> Part 2 integration ----------

LINES = [
    "Aarav Mehta",
    "aarav.mehta@example.com | +91 98765 43210",
    "linkedin.com/in/aarav-mehta",
    "",
    "EXPERIENCE",
    "Data Analyst Intern, Acme Analytics",
    "Jan 2024 - Mar 2024",
    "- Built weekly reporting dashboards for operations teams.",
    "- Cleaned and validated large sales datasets using Python.",
    "",
    "SKILLS",
    "Python, SQL, Pandas, scikit-learn",
    "",
    "EDUCATION",
    "B.Tech in Computer Science, KIIT University",
]


def _resume_pdf():
    doc = fitz.open()
    page = doc.new_page()
    for i, line in enumerate(LINES):
        if line:
            page.insert_text((60, 60 + 16 * i), line)
    data = doc.tobytes()
    doc.close()
    return data


def test_ingest_chunks_embed_scrubbed_text_only():
    _, ingest = run_ingest(_resume_pdf(), "resume.pdf")
    fake = FakeModel()
    out = embed_chunks(ingest.chunks, Embedder(fake))
    assert out.shape == (len(ingest.chunks), 384)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    sent = [t for docs, _ in fake.calls for t in docs]
    assert sent == [c.embedding_text() for c in ingest.chunks]
    for text in sent:
        assert "aarav.mehta@example.com" not in text
        assert "98765 43210" not in text
        assert "Aarav Mehta" not in text
        assert "linkedin.com" not in text
        assert not text.startswith(QUERY_PREFIX)
    assert any("[EMAIL]" in t for t in sent)


def test_ingest_embeddings_deterministic():
    _, a = run_ingest(_resume_pdf(), "a.pdf")
    _, b = run_ingest(_resume_pdf(), "b.pdf")
    ea = embed_chunks(a.chunks, Embedder(FakeModel()))
    eb = embed_chunks(b.chunks, Embedder(FakeModel()))
    assert np.array_equal(ea, eb)


@pytest.mark.skipif(os.environ.get("VERIHIRE_RUN_MODEL_TESTS") != "1",
                    reason="set VERIHIRE_RUN_MODEL_TESTS=1 after caching the model")
def test_real_model_offline():
    e = get_embedder(offline=True)
    docs = e.embed_documents(["Python developer with SQL experience", "Chef with 10 years in kitchens"])
    query = e.embed_query("software engineer")
    assert docs.shape == (2, 384) and query.shape == (384,)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-4)
    assert float(docs[0] @ query) > float(docs[1] @ query)


@pytest.mark.skipif(os.environ.get("VERIHIRE_RUN_MODEL_TESTS") != "1",
                    reason="set VERIHIRE_RUN_MODEL_TESTS=1 after caching the model")
def test_real_model_on_ingested_resume():
    _, ingest = run_ingest(_resume_pdf(), "resume.pdf")
    out = embed_chunks(ingest.chunks, get_embedder(offline=True))
    assert out.shape == (len(ingest.chunks), 384)


#part 3

GOLD = Path(evaluate.__file__).with_name("gold.jsonl")


def test_gold_dataset_is_valid_and_synthetic():
    records, errors = evaluate.load_gold(GOLD, expected_count=30)
    assert errors == [] and len(records) == 30
    for rec in records:
        assert all(e.endswith("@example.com") for e in rec["pii"]["emails"])


def test_evaluate_runs_from_gold(capsys):
    assert evaluate.main(["--gold", str(GOLD)]) == 0
    assert "Email recall" in capsys.readouterr().out


def test_evaluate_rejects_malformed_dataset(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": 1}\nnot json\n', encoding="utf-8")
    assert evaluate.main(["--gold", str(bad), "--expected-count", "2"]) == 1


def test_skills_dict_is_plain_json_regression():
    # regression: a pasted "// path" comment line once made the file invalid JSON
    assert SKILLS_DICT_PATH.read_text(encoding="utf-8").lstrip().startswith("{")


class _CorpusFakeModel:
    def embed(self, documents, batch_size=256, **kwargs):
        for text in documents:
            yield np.random.default_rng(zlib.crc32(text.encode("utf-8"))).normal(size=384).astype(np.float32)


def _gold_record(rec_id):
    records, _ = evaluate.load_gold(GOLD)
    return next(r for r in records if r["id"] == rec_id)


def _check_synthetic_smoke(embedder):
    from ml.data_ingest import build_corpus
    rec = _gold_record("G01")
    doc, result = run_ingest(build_corpus.render_pdf(rec["pages"]), "synthetic.pdf")
    from ml.docint.ingest.extractor import ExtractedDoc
    from ml.docint.ingest.ingest import IngestResult
    assert isinstance(doc, ExtractedDoc) and isinstance(result, IngestResult)
    for chunk in result.chunks:
        assert chunk.text_original == doc.pages[chunk.page_no - 1]["text"][chunk.char_start:chunk.char_end]
        for key in ("emails", "phones", "urls"):
            for raw in rec["pii"][key]:
                assert raw not in chunk.text_scrubbed
    vectors = embed_chunks(result.chunks, embedder)
    assert vectors.shape == (len(result.chunks), 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_synthetic_pdf_smoke_fake_model():
    from ml.docint.embed.embedder import Embedder, embed_chunks
    _check_synthetic_smoke(Embedder(_CorpusFakeModel()))


@pytest.mark.skipif(os.environ.get("VERIHIRE_RUN_MODEL_TESTS") != "1", reason="needs cached model")
def test_synthetic_pdf_smoke_real_model():
    from ml.docint.embed.embedder import get_embedder
    _check_synthetic_smoke(get_embedder(offline=True))


def test_pipeline_version_exposed():
    from ml.docint.ingest.ingest import PIPELINE_VERSION
    assert isinstance(PIPELINE_VERSION, str) and PIPELINE_VERSION


def test_build_corpus_schema_and_privacy(tmp_path):
    from ml.data_ingest import build_corpus
    from ml.docint.embed.embedder import Embedder
    records, _ = evaluate.load_gold(GOLD)
    sources = [(r["id"] + ".pdf", build_corpus.render_pdf(r["pages"])) for r in records[:3]]
    out = build_corpus.build_records(sources, Embedder(_CorpusFakeModel()))
    assert all(set(r) == {"doc_id", "profile", "chunks", "embedding"} for r in out)
    for r in out:
        assert len(r["embedding"]) == len(r["chunks"])
        assert all(len(v) == 384 for v in r["embedding"])
        assert all("text_original" not in c for c in r["chunks"])
    path = tmp_path / "corpus.jsonl"
    build_corpus.write_jsonl(out, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
