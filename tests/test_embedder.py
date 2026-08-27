"""Tests for the embedding cache. The API is never called: `_call_api` is stubbed.

What matters here is not that Google returns good vectors, it is that the cache
is correct. A cache that returns the wrong vector for a text, or that mixes a
document vector with a query vector, would corrupt every retrieval metric while
every other test stayed green.
"""

from pathlib import Path

import numpy as np
import pytest

from src.core.embedder import DEFAULT_MODEL, EMBEDDING_DIM, Embedder, EmbeddingCacheMiss


def _fake_vector(text: str, task_type: str) -> np.ndarray:
    """Deterministic stand-in for the API: a different vector per (text, task)."""
    seed = abs(hash(f"{task_type}:{text}")) % (2**32)
    vector = np.random.default_rng(seed).normal(size=EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def embedder(tmp_path: Path, monkeypatch) -> Embedder:
    instance = Embedder(cache_dir=tmp_path, api_key="test-key")
    calls: list[list[str]] = []

    def fake_call(texts: list[str], task_type: str) -> list[np.ndarray]:
        calls.append(list(texts))
        instance.api_calls += 1
        return [_fake_vector(text, task_type) for text in texts]

    monkeypatch.setattr(instance, "_call_api", fake_call)
    instance.calls = calls  # type: ignore[attr-defined]
    return instance


def test_embeds_and_caches(embedder: Embedder, tmp_path: Path):
    vectors = embedder.embed_documents(["first text", "second text"])
    assert vectors.shape == (2, EMBEDDING_DIM)
    assert vectors.dtype == np.float32
    assert len(list((tmp_path / DEFAULT_MODEL).glob("*.npy"))) == 2


def test_second_call_hits_the_cache(embedder: Embedder):
    embedder.embed_documents(["first text", "second text"])
    embedder.calls.clear()  # type: ignore[attr-defined]
    again = embedder.embed_documents(["first text", "second text"])
    assert embedder.calls == []  # type: ignore[attr-defined]
    assert again.shape == (2, EMBEDDING_DIM)


def test_partial_cache_hit_preserves_input_order(embedder: Embedder):
    first = embedder.embed_documents(["alpha", "beta"])
    combined = embedder.embed_documents(["gamma", "alpha", "delta", "beta"])
    # Only the two new texts reach the API...
    assert embedder.calls[-1] == ["gamma", "delta"]  # type: ignore[attr-defined]
    # ...and the cached ones come back in the positions they were asked for.
    assert np.allclose(combined[1], first[0])
    assert np.allclose(combined[3], first[1])


def test_document_and_query_vectors_do_not_share_a_cache_entry(embedder: Embedder):
    as_document = embedder.embed_documents(["net 45 days"])[0]
    as_query = embedder.embed_query("net 45 days")
    assert not np.allclose(as_document, as_query)


def test_offline_miss_raises_with_an_actionable_message(tmp_path: Path):
    offline = Embedder(cache_dir=tmp_path, api_key="", offline=True)
    with pytest.raises(EmbeddingCacheMiss) as error:
        offline.embed_documents(["never cached"])
    assert "index --refresh" in str(error.value)


def test_offline_reads_a_cache_written_earlier(tmp_path: Path, embedder: Embedder):
    written = embedder.embed_documents(["cached text"])[0]
    offline = Embedder(cache_dir=tmp_path, api_key="", offline=True)
    assert np.allclose(offline.embed_documents(["cached text"])[0], written)


def test_a_corrupt_cache_file_is_treated_as_a_miss(tmp_path: Path):
    offline = Embedder(cache_dir=tmp_path, api_key="", offline=True)
    path = offline._cache_path("some text", "RETRIEVAL_DOCUMENT")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros(17, dtype=np.float32))
    with pytest.raises(EmbeddingCacheMiss):
        offline.embed_documents(["some text"])


def test_empty_input_needs_no_api_call(embedder: Embedder):
    assert embedder.embed_documents([]).shape == (0, EMBEDDING_DIM)
    assert embedder.calls == []  # type: ignore[attr-defined]


def test_changing_the_model_changes_the_cache_key(tmp_path: Path):
    one = Embedder(model="gemini-embedding-001", cache_dir=tmp_path, api_key="k")
    other = Embedder(model="gemini-embedding-2", cache_dir=tmp_path, api_key="k")
    assert one._cache_path("same text", "RETRIEVAL_DOCUMENT") != other._cache_path(
        "same text", "RETRIEVAL_DOCUMENT"
    )
