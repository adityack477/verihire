# tests/test_embedder.py
import os
import zlib

import numpy as np
import pytest

from ml.docint.embed import embedder as emb_mod
from ml.docint.embed.embedder import (
    EMBEDDING_DIM, MODEL_NAME, OFFLINE_ENV_VAR, QUERY_PREFIX, Embedder, EmbeddingInputError,
    EmbeddingOutputError, ModelUnavailableError, embed_chunks, get_embedder,
)
from ml.docint.ingest.chunker import Chunk


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


@pytest.mark.skipif(os.environ.get("VERIHIRE_RUN_MODEL_TESTS") != "1",
                    reason="set VERIHIRE_RUN_MODEL_TESTS=1 after caching the model")
def test_real_model_offline():
    e = get_embedder(offline=True)
    docs = e.embed_documents(["Python developer with SQL experience", "Chef with 10 years in kitchens"])
    query = e.embed_query("software engineer")
    assert docs.shape == (2, 384) and query.shape == (384,)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-4)
    assert float(docs[0] @ query) > float(docs[1] @ query)