# ml/docint/embed/embedder.py
"""Local FastEmbed embedder for BAAI/bge-small-en-v1.5 (VeriHire Part 2).

One-time download: the ONNX weights must be cached once (needs internet), then
all runs can be offline::

    python -m ml.docint.embed.embedder          # downloads into the cache dir
    set VERIHIRE_EMBED_OFFLINE=1                # later runs: never touch the network

Cache dir: ``cache_dir`` argument, else env ``FASTEMBED_CACHE_PATH``, else
``<repo>/.cache/fastembed`` (git-ignored; weights are never committed).
There is no remote fallback: a missing model raises ``ModelUnavailableError``.

Vectors are float32, L2-normalised. ``embed_documents`` -> (n, 384) array,
``embed_query`` -> (384,) array. Only queries get the BGE query prefix.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_BATCH_SIZE = 32
OFFLINE_ENV_VAR = "VERIHIRE_EMBED_OFFLINE"
CACHE_ENV_VAR = "FASTEMBED_CACHE_PATH"


class EmbedderError(Exception):
    """Base class for embedder errors."""


class ModelUnavailableError(EmbedderError):
    """FastEmbed or the cached model is missing (no remote fallback is used)."""


class EmbeddingInputError(EmbedderError, ValueError):
    """Invalid input text(s)."""


class EmbeddingOutputError(EmbedderError):
    """Model output has the wrong shape/values."""


class _HasEmbeddingText(Protocol):
    def embedding_text(self) -> str: ...


def _default_cache_dir() -> str:
    env = os.environ.get(CACHE_ENV_VAR)
    if env:
        return env
    return str(Path(__file__).resolve().parents[3] / ".cache" / "fastembed")


def _env_offline() -> bool:
    return os.environ.get(OFFLINE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def _load_model(cache_dir: str, offline: bool) -> Any:
    """Load the FastEmbed model locally; never falls back to an online API."""
    try:
        from fastembed import TextEmbedding  # type: ignore
    except ImportError as exc:
        raise ModelUnavailableError("fastembed is not installed: pip install fastembed") from exc
    kwargs: dict[str, Any] = {"local_files_only": True} if offline else {}
    try:
        return TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir, **kwargs)
    except Exception as exc:
        mode = "offline mode: model not cached" if offline else "model download/load failed"
        raise ModelUnavailableError(
            f"{MODEL_NAME} unavailable ({mode}) in '{cache_dir}'. Run "
            "'python -m ml.docint.embed.embedder' once with internet access."
        ) from exc


class Embedder:
    """384-dim L2-normalised local embedder."""

    def __init__(self, model: Any = None, *, cache_dir: Optional[str] = None,
                 offline: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.name: str = MODEL_NAME
        self.dim: int = EMBEDDING_DIM
        self._model = model
        self._cache_dir = cache_dir or _default_cache_dir()
        self._offline = offline
        self._batch_size = batch_size

    def warm_up(self) -> None:
        """Load the model now so failures surface early."""
        self._get_model()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed document texts as-is (no prefix). Returns (n, 384) float32."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, (list, tuple)):
            raise EmbeddingInputError("texts must be a list/tuple of str")
        for text in texts:
            self._check_text(text)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(list(texts))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query with the BGE query prefix. Returns (384,) float32."""
        self._check_text(query)
        return self._encode([QUERY_PREFIX + query])[0]

    @staticmethod
    def _check_text(text: object) -> None:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError("each text must be a non-blank str")

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _load_model(self._cache_dir, self._offline)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        try:
            raw = list(model.embed(texts, batch_size=self._batch_size))
        except Exception as exc:
            raise EmbeddingOutputError("embedding model call failed") from exc
        arr = np.asarray(raw, dtype=np.float32)
        if arr.shape != (len(texts), self.dim):
            raise EmbeddingOutputError(f"expected shape {(len(texts), self.dim)}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise EmbeddingOutputError("non-finite values in embeddings")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise EmbeddingOutputError("zero-norm embedding")
        return (arr / norms).astype(np.float32)


_CACHE: dict[tuple[str, bool, int], Embedder] = {}


def get_embedder(*, cache_dir: Optional[str] = None, offline: Optional[bool] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE) -> Embedder:
    """Return a loaded local Embedder (name, dim=384, embed_documents, embed_query)."""
    resolved_offline = _env_offline() if offline is None else offline
    cache = str(cache_dir) if cache_dir else _default_cache_dir()
    key = (cache, resolved_offline, batch_size)
    embedder = _CACHE.get(key)
    if embedder is None:
        model = _load_model(cache, resolved_offline)
        embedder = Embedder(model, cache_dir=cache, offline=resolved_offline, batch_size=batch_size)
        _CACHE[key] = embedder
    return embedder


def embed_chunks(chunks: Sequence[_HasEmbeddingText],
                 embedder: Optional[Embedder] = None) -> np.ndarray:
    """Embed ``Chunk.embedding_text()`` (scrubbed text) for every chunk."""
    emb = embedder or get_embedder()
    return emb.embed_documents([c.embedding_text() for c in chunks])


def main() -> None:
    """Download/cache the model once (internet required)."""
    emb = get_embedder(offline=False)
    vec = emb.embed_query("warm-up")
    print(f"{emb.name} cached; dim={emb.dim}; sample shape={vec.shape}")


if __name__ == "__main__":
    main()