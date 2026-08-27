"""Tests for both vector stores, including that they agree with each other.

The equivalence test is the interesting one. NumpyStore is exact by
construction, so it is the oracle: if ChromaDB's approximate index disagrees
about the top results on a corpus this small, something is misconfigured -- the
wrong distance metric, an unnormalised vector, or Chroma quietly embedding the
text itself with its own default model.
"""

from pathlib import Path

import numpy as np
import pytest

from src.core.chunker import Chunk
from src.core.store import ChromaStore, NumpyStore, SearchHit

chromadb = pytest.importorskip("chromadb", reason="chromadb is required for the Chroma store")


def _chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc::{index:03d}",
        doc_id="doc",
        doc_title="Test Document",
        source_path="doc.md",
        section=f"{index}. Section",
        text=f"Body text number {index}.",
        index_text=f"Test Document > {index}. Section\n\nBody text number {index}.",
        ordinal=index,
    )


@pytest.fixture(scope="module")
def corpus() -> tuple[list[Chunk], np.ndarray]:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(40, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [_chunk(index) for index in range(40)], vectors


def test_numpy_store_returns_exact_nearest_neighbours(corpus):
    chunks, vectors = corpus
    store = NumpyStore()
    store.add(chunks, vectors)
    assert len(store) == 40

    query = vectors[13]
    hits = store.search(query, k=3)
    assert hits[0].chunk.chunk_id == chunks[13].chunk_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_numpy_store_round_trips_through_disk(corpus, tmp_path: Path):
    chunks, vectors = corpus
    store = NumpyStore()
    store.add(chunks, vectors)
    store.persist(tmp_path / "index")

    reloaded = NumpyStore.load(tmp_path / "index")
    assert len(reloaded) == len(store)
    before = store.search(vectors[5], k=5)
    after = reloaded.search(vectors[5], k=5)
    assert [hit.chunk for hit in before] == [hit.chunk for hit in after]
    assert [hit.score for hit in before] == pytest.approx([hit.score for hit in after])


def test_loading_a_missing_index_says_how_to_build_one(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as error:
        NumpyStore.load(tmp_path / "nothing-here")
    assert "src.main index" in str(error.value)


def test_misaligned_input_is_rejected(corpus):
    chunks, vectors = corpus
    store = NumpyStore()
    with pytest.raises(ValueError, match="chunks but"):
        store.add(chunks[:5], vectors[:4])


def test_unnormalised_vectors_are_rejected(corpus):
    chunks, vectors = corpus
    store = NumpyStore()
    with pytest.raises(ValueError, match="unit length"):
        store.add(chunks[:3], vectors[:3] * 2.0)


def test_empty_store_returns_no_hits():
    assert NumpyStore().search(np.ones(8, dtype=np.float32) / np.sqrt(8), k=5) == []


def test_search_is_deterministic_when_scores_tie():
    # Two identical vectors under different ids: the tie must break the same way
    # every run, or an evaluation cannot be compared with the one before it.
    vector = np.zeros(4, dtype=np.float32)
    vector[0] = 1.0
    chunks = [_chunk(2), _chunk(1)]
    store = NumpyStore()
    store.add(chunks, np.stack([vector, vector]))
    first = [hit.chunk.chunk_id for hit in store.search(vector, k=2)]
    second = [hit.chunk.chunk_id for hit in store.search(vector, k=2)]
    assert first == second == ["doc::001", "doc::002"]


def test_chroma_store_round_trips_and_reopens(corpus, tmp_path: Path):
    chunks, vectors = corpus
    store = ChromaStore(path=tmp_path / "chroma")
    store.add(chunks, vectors)
    store.persist(tmp_path / "chroma")
    assert len(store) == 40

    reopened = ChromaStore.load(tmp_path / "chroma")
    assert len(reopened) == 40
    hit = reopened.search(vectors[9], k=1)[0]
    assert hit.chunk.chunk_id == chunks[9].chunk_id
    # Metadata survived the round trip, so a citation still names its section.
    assert hit.chunk.section == chunks[9].section
    assert hit.chunk.doc_title == chunks[9].doc_title


def test_chroma_add_is_idempotent(corpus, tmp_path: Path):
    chunks, vectors = corpus
    store = ChromaStore(path=tmp_path / "chroma")
    store.add(chunks, vectors)
    store.add(chunks, vectors)
    assert len(store) == 40


def test_both_stores_agree_on_the_ranking(corpus, tmp_path: Path):
    chunks, vectors = corpus
    numpy_store = NumpyStore()
    numpy_store.add(chunks, vectors)
    chroma_store = ChromaStore(path=tmp_path / "chroma")
    chroma_store.add(chunks, vectors)

    rng = np.random.default_rng(99)
    for _ in range(5):
        query = rng.normal(size=32).astype(np.float32)
        query /= np.linalg.norm(query)
        exact = numpy_store.search(query, k=5)
        approximate = chroma_store.search(query, k=5)
        assert [hit.chunk.chunk_id for hit in exact] == [
            hit.chunk.chunk_id for hit in approximate
        ]
        assert [hit.score for hit in exact] == pytest.approx(
            [hit.score for hit in approximate], abs=1e-5
        )


def test_search_hit_scores_are_similarities_not_distances(corpus, tmp_path: Path):
    chunks, vectors = corpus
    for store in (NumpyStore(), ChromaStore(path=tmp_path / "chroma-scale")):
        store.add(chunks, vectors)
        hit: SearchHit = store.search(vectors[0], k=1)[0]
        assert hit.score == pytest.approx(1.0, abs=1e-5), "identical vectors must score 1, not 0"
