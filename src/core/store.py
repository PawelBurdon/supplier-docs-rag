"""Vector storage behind a narrow interface, with two implementations.

The interface is four methods -- add, search, persist, load -- because that is
everything the retriever needs. Anything wider would be an invitation to leak
implementation detail into the layer above.

Two implementations exist on purpose.

NumpyStore is exact brute force: one matrix multiply against every stored
vector, O(n*d). At 63 chunks that is roughly 50,000 multiply-adds, microseconds,
and it returns the true nearest neighbours by definition. It exists because the
mechanics of similarity search are three lines of numpy, and a person who has
written those three lines can say what a vector database actually does.

ChromaStore is the same interface over a real vector database: HNSW indexing,
durable storage, metadata filtering, concurrent readers. At this corpus size it
is slower and heavier than the numpy store on every axis -- that is not a
criticism of Chroma, it is what a small corpus means. The reason to have built
against it anyway is that the crossover is real: brute force scans every vector,
so cost grows linearly with the corpus, and somewhere around a million chunks a
query stops being microseconds and starts being a problem. HNSW trades exactness
for a graph walk that touches a tiny fraction of the index.

That trade is the point worth understanding: HNSW is approximate. The two stores
can disagree about the k-th result, and when they do, the numpy store is right
and the Chroma store is fast. A test asserts they agree on this corpus, which at
63 chunks they do; at scale that test would be the wrong test to write.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.chunker import Chunk

COLLECTION_NAME = "supplier_docs"


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk and its cosine similarity, in [-1, 1], higher is better.

    Both stores return the same scale. Chroma reports cosine *distance*, which is
    converted here, so a caller can compare scores across implementations without
    knowing which one it is holding.
    """

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Add vectors, search them, write them to disk, read them back."""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Store chunks with their unit-length embeddings, aligned by position."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        """Return the k most similar chunks, best first."""

    @abstractmethod
    def persist(self, path: str | Path) -> None:
        """Write the store to `path`."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Read a store previously written to `path`."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored chunks."""


def _validate(chunks: list[Chunk], vectors: np.ndarray) -> np.ndarray:
    """Reject misaligned or non-unit input at the door.

    A silent misalignment between chunks and vectors returns confident, wrong
    citations for every query afterwards, and nothing downstream can detect it.
    The norm check catches an embedder that skipped normalisation, which would
    turn cosine similarity into a length contest.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if len(chunks) == 0:
        raise ValueError("Refusing to add an empty batch")
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2-D array of vectors, got shape {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Vectors are not unit length; cosine similarity would be wrong. "
            f"Norms range from {norms.min():.4f} to {norms.max():.4f}."
        )
    return vectors.astype(np.float32, copy=False)


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "source_path": chunk.source_path,
        "section": chunk.section,
        "text": chunk.text,
        "index_text": chunk.index_text,
        "ordinal": chunk.ordinal,
    }


def _chunk_from_dict(payload: dict) -> Chunk:
    return Chunk(**payload)


class NumpyStore(VectorStore):
    """Exact cosine search over a dense matrix."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = _validate(chunks, vectors)
        if self._chunks:
            if vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError(
                    f"Vector dimension {vectors.shape[1]} does not match the "
                    f"stored dimension {self._vectors.shape[1]}"
                )
            self._vectors = np.vstack([self._vectors, vectors])
        else:
            self._vectors = vectors.copy()
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if not self._chunks or k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"Query has dimension {query.shape[0]}, store has {self._vectors.shape[1]}"
            )
        # Both sides are unit length, so the dot product is the cosine.
        scores = self._vectors @ query
        # Sorted by score descending, then by chunk_id, so that equal scores do
        # not reorder between runs and make an evaluation irreproducible.
        order = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )
        return [SearchHit(self._chunks[index], float(scores[index])) for index in order[:k]]

    def persist(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors)
        (directory / "chunks.json").write_text(
            json.dumps([_chunk_to_dict(chunk) for chunk in self._chunks], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "NumpyStore":
        directory = Path(path)
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.json"
        if not vectors_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                f"No numpy index at {directory}. Build one with: python -m src.main index"
            )
        store = cls()
        store._vectors = np.load(vectors_path).astype(np.float32, copy=False)
        store._chunks = [
            _chunk_from_dict(payload)
            for payload in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        if len(store._chunks) != len(store._vectors):
            raise ValueError(
                f"Corrupt index at {directory}: {len(store._chunks)} chunks, "
                f"{len(store._vectors)} vectors"
            )
        return store

    def __len__(self) -> int:
        return len(self._chunks)


class ChromaStore(VectorStore):
    """The same interface over ChromaDB, using its persistent client."""

    def __init__(self, path: str | Path = "chroma", collection_name: str = COLLECTION_NAME) -> None:
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # Embeddings are supplied by this project's embedder. Without
            # embedding_function=None, Chroma installs and runs its own default
            # model, which would silently embed with something else entirely.
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = _validate(chunks, vectors)
        # upsert, not add: re-indexing the same documents must be idempotent
        # rather than raising on duplicate ids or silently doubling the corpus.
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[vector.tolist() for vector in vectors],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "source_path": chunk.source_path,
                    "section": chunk.section,
                    "index_text": chunk.index_text,
                    "ordinal": chunk.ordinal,
                }
                for chunk in chunks
            ],
        )

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        count = len(self)
        if count == 0 or k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        response = self._collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for chunk_id, document, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=str(metadata["doc_id"]),
                doc_title=str(metadata["doc_title"]),
                source_path=str(metadata["source_path"]),
                section=str(metadata["section"]),
                text=document,
                index_text=str(metadata["index_text"]),
                ordinal=int(metadata["ordinal"]),
            )
            # Chroma reports cosine distance; the interface promises similarity.
            hits.append(SearchHit(chunk, 1.0 - float(distance)))
        return hits

    def persist(self, path: str | Path) -> None:
        """Nothing to do: the persistent client wrote to disk during add().

        Kept because the interface promises it, and because a caller should not
        have to know which implementation happens to need an explicit flush.
        """
        if Path(path) != self.path:
            raise ValueError(
                f"This ChromaStore is bound to {self.path}; it cannot persist to {path}"
            )

    @classmethod
    def load(cls, path: str | Path, collection_name: str = COLLECTION_NAME) -> "ChromaStore":
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"No Chroma index at {directory}. Build one with: "
                "python -m src.main index --store chroma"
            )
        return cls(path=directory, collection_name=collection_name)

    def __len__(self) -> int:
        return self._collection.count()
