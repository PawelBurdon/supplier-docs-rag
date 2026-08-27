"""Embeddings from Google gemini-embedding-001, with an on-disk cache.

The cache is the reason this project can be evaluated by someone who has no API
key. It lives in `embedding_cache/` and is committed to the repository, so a
clone can rebuild the index, run every retrieval metric, and reproduce the
numbers in the README offline and deterministically. Committing a generated
artefact is a real cost -- it can drift from the documents -- so a miss is a
loud, actionable error rather than a silent recomputation or a silent zero
vector.

One file per chunk, named by hash. A single bundle would rewrite entirely on any
document edit, which in git means a new opaque blob per commit and no way to see
what actually changed. Per-file also makes writes atomic: an interrupted
indexing run leaves the chunks it finished, not a corrupt bundle.

Queries are cached too, under the same scheme. Without that, evaluating the
golden set in CI would still need the network to embed its 18 questions, and the
whole point of committing the cache would be lost.

Documents and queries are embedded with different task types. The model is
trained asymmetrically: RETRIEVAL_DOCUMENT for the indexed passage,
RETRIEVAL_QUERY for the question. Using one type for both is a quiet way to lose
retrieval quality, so the task type is part of the cache key -- the same text
embedded as a document and as a query is two different vectors and must never
share a cache entry.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

# gemini-embedding-001 rather than text-embedding-004: the API no longer serves
# 004. Of what it does serve, 001 is the only model that accepts a batch --
# gemini-embedding-2 returned one vector for three inputs in testing, which
# would mean one request per chunk.
DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_CACHE_DIR = Path("embedding_cache")
# 768 of the model's native 3072 dimensions, truncated through its Matryoshka
# representation. A quarter of the cache to commit and a tenth of the indexing
# time, for a quality difference this corpus is too small to show. The catch is
# in _normalise().
EMBEDDING_DIM = 768
BATCH_SIZE = 100
MAX_ATTEMPTS = 3

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


class EmbeddingCacheMiss(RuntimeError):
    """Raised when a text is not cached and no network call is possible.

    Carries the fix in the message: this is the error a contributor sees after
    editing a document without re-indexing, and after a stale-cache failure in
    CI.
    """


class Embedder:
    """Embeds text, reading from and writing to the on-disk cache.

    Args:
        model: Embedding model id. Part of the cache key, so switching models
            cannot silently mix vector spaces.
        cache_dir: Root of the cache. One subdirectory per model.
        api_key: Google AI Studio key. Falls back to GOOGLE_API_KEY.
        offline: When True, never call the network. Any cache miss raises
            EmbeddingCacheMiss. This is how CI runs.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        api_key: str | None = None,
        offline: bool = False,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir) / model.replace("/", "_")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or ""
        self.offline = offline or not self.api_key
        self._client = None
        self.cache_hits = 0
        self.api_calls = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed indexed passages. Returns an (n, dim) float32 array."""
        return self._embed(texts, DOCUMENT_TASK)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question. Returns a (dim,) float32 array."""
        return self._embed([text], QUERY_TASK)[0]

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        vectors: list[np.ndarray | None] = []
        missing: list[tuple[int, str]] = []
        for position, text in enumerate(texts):
            cached = self._read_cache(text, task_type)
            if cached is None:
                vectors.append(None)
                missing.append((position, text))
            else:
                vectors.append(cached)
                self.cache_hits += 1

        if missing:
            if self.offline:
                raise EmbeddingCacheMiss(
                    f"{len(missing)} of {len(texts)} texts are not in the embedding cache "
                    f"at {self.cache_dir}, and no API key is available "
                    f"(task type {task_type}).\n"
                    "The committed cache is stale with respect to the documents or the "
                    "golden set. Rebuild it with a key:\n"
                    "    python -m src.main index --refresh\n"
                    "then commit the new cache files."
                )
            for (position, text), vector in zip(missing, self._call_api([t for _, t in missing], task_type)):
                self._write_cache(text, task_type, vector)
                vectors[position] = vector

        if any(vector is None for vector in vectors):
            raise RuntimeError("Internal error: an input was left without an embedding")
        return np.stack(vectors)

    def _cache_path(self, text: str, task_type: str) -> Path:
        digest = hashlib.sha256(f"{self.model}|{task_type}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _read_cache(self, text: str, task_type: str) -> np.ndarray | None:
        path = self._cache_path(text, task_type)
        if not path.is_file():
            return None
        vector = np.load(path)
        if vector.shape != (EMBEDDING_DIM,):
            # A truncated or foreign file is treated as a miss rather than
            # poisoning the index with a wrong-shaped vector.
            return None
        return vector.astype(np.float32, copy=False)

    def _write_cache(self, text: str, task_type: str, vector: np.ndarray) -> None:
        path = self._cache_path(text, task_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name and rename, so a killed process cannot leave
        # a half-written vector that later loads as a plausible-looking miss.
        temporary = path.with_suffix(".npy.tmp")
        # Written through a file handle: np.save appends ".npy" to a path whose
        # name does not already end in it, which would leave the temporary file
        # under a name the rename never finds.
        with open(temporary, "wb") as handle:
            np.save(handle, vector)
        temporary.replace(path)

    def _call_api(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        """Embed uncached texts in batches, with backoff on transient failures."""
        client = self._ensure_client()
        from google.genai import types  # Imported late; the offline path never needs it.

        results: list[np.ndarray] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self._with_retries(
                lambda: client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
            )
            self.api_calls += 1
            if len(response.embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding API returned {len(response.embeddings)} vectors "
                    f"for {len(batch)} inputs; refusing to guess the alignment."
                )
            results.extend(_normalise(embedding.values) for embedding in response.embeddings)
        return results

    def _with_retries(self, call):
        delay = 1.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return call()
            except Exception:
                if attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def _ensure_client(self):
        if self._client is None:
            if not self.api_key:
                raise EmbeddingCacheMiss(
                    "No GOOGLE_API_KEY set and the text is not cached. "
                    "Copy .env.example to .env and add a key, or run with --offline."
                )
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client


def _normalise(values) -> np.ndarray:
    """Return a unit-length float32 vector.

    This is not cosmetic. A Matryoshka-truncated embedding is not unit length --
    the 768-dimensional slice of gemini-embedding-001 came back with a norm of
    about 0.59 in testing, and the norm varies with the text. Skipping this step
    leaves a dot product that is part similarity and part vector magnitude, so
    longer or denser chunks quietly outrank better matches. Nothing raises; the
    ranking is simply wrong.

    Doing it once, here, also means cosine similarity is a plain dot product
    everywhere downstream and both store implementations agree on what a score
    means. The cache therefore holds unit vectors, not the raw API response.
    """
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Embedding API returned a zero vector")
    return vector / norm
