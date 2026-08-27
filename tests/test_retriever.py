"""Tests for tokenisation, BM25, and reciprocal rank fusion.

No network: the embedder is a stub returning fixed vectors, so these tests
assert fusion arithmetic and ranking behaviour rather than embedding quality.
The one test that uses real text is the tokeniser, which is where the argument
for hybrid retrieval lives.
"""

import numpy as np
import pytest

from src.core.chunker import Chunk
from src.core.retriever import RRF_K, Retriever, tokenize
from src.core.store import NumpyStore

DIM = 8


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc::{index:03d}",
        doc_id="doc",
        doc_title="Test Document",
        source_path="doc.md",
        section=f"{index}. Section",
        text=text,
        index_text=f"Test Document > {index}. Section\n\n{text}",
        ordinal=index,
    )


class StubEmbedder:
    """Returns a preset query vector; never touches the network."""

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls = 0

    def embed_query(self, text: str) -> np.ndarray:
        self.calls += 1
        return self.vector


def _unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_tokenize_keeps_codes_whole_and_splits_them():
    assert tokenize("FRM-CRB-52") == ["frm-crb-52", "frm", "crb", "52"]
    assert tokenize("HCS-SLA-2024") == ["hcs-sla-2024", "hcs", "sla", "2024"]
    assert tokenize("clause 5.2 applies") == ["clause", "5.2", "5", "2", "applies"]


def test_tokenize_lets_a_partial_code_reach_a_specific_one():
    # The reason for emitting both forms: this query and this text share tokens.
    query = set(tokenize("what is the lead time for FRM-CRB"))
    text = set(tokenize("Carbon frame set FRM-CRB-52 ships in 35 days"))
    assert query & text >= {"frm", "crb"}


def test_tokenize_ignores_punctuation_and_case():
    assert tokenize("Penalty: 0.5 percent, capped!") == [
        "penalty",
        "0.5",
        "0",
        "5",
        "percent",
        "capped",
    ]


@pytest.fixture
def retriever() -> Retriever:
    chunks = [
        _chunk(0, "The late delivery penalty is 0.5 percent per calendar day."),
        _chunk(1, "Carbon frame sets FRM-CRB ship in 35 days for tier 2 suppliers."),
        _chunk(2, "Warranty on aluminium frames runs for 24 months."),
        _chunk(3, "Escalation level 2 goes to the Supply Chain Manager."),
    ]
    vectors = np.stack([_unit(1, 0, 0, 0, 0, 0, 0, 0) for _ in chunks])
    # Make chunk 2 the vector favourite and chunk 1 a distant relative.
    vectors[2] = _unit(0, 1, 0, 0, 0, 0, 0, 0)
    vectors[1] = _unit(0, 0.9, 0.4, 0, 0, 0, 0, 0)
    store = NumpyStore()
    store.add(chunks, vectors)
    return Retriever(store, StubEmbedder(_unit(0, 1, 0, 0, 0, 0, 0, 0)), chunks)


def test_keyword_path_finds_an_exact_code_the_vector_path_ranks_low(retriever: Retriever):
    # The stub query vector points at chunk 2; only BM25 knows about FRM-CRB.
    keyword_only = retriever.retrieve("FRM-CRB lead time", mode="keyword")
    assert keyword_only[0].chunk.chunk_id == "doc::001"
    assert keyword_only[0].keyword_rank == 1
    assert keyword_only[0].vector_rank is None


def test_hybrid_beats_either_path_alone(retriever: Retriever):
    hybrid = retriever.retrieve("FRM-CRB lead time", mode="hybrid")
    ids = [hit.chunk.chunk_id for hit in hybrid]
    # The keyword answer is present, and so is the vector favourite.
    assert "doc::001" in ids
    assert "doc::002" in ids
    both_paths = [hit for hit in hybrid if hit.vector_rank and hit.keyword_rank]
    assert both_paths, "at least one chunk should be found by both paths"


def test_rrf_score_matches_the_formula(retriever: Retriever):
    hits = retriever.retrieve("FRM-CRB lead time", mode="hybrid")
    for hit in hits:
        expected = 0.0
        if hit.vector_rank is not None:
            expected += 1.0 / (RRF_K + hit.vector_rank)
        if hit.keyword_rank is not None:
            expected += 1.0 / (RRF_K + hit.keyword_rank)
        assert hit.score == pytest.approx(expected)


def test_a_chunk_found_by_both_paths_outranks_one_found_by_either():
    chunks = [
        _chunk(0, "alpha beta gamma"),  # both paths
        _chunk(1, "alpha alpha alpha"),  # keyword only, rank 1 there
        _chunk(2, "unrelated content"),  # vector only
    ]
    vectors = np.stack([_unit(1, 0, 0, 0, 0, 0, 0, 0) for _ in chunks])
    vectors[2] = _unit(0.99, 0.14, 0, 0, 0, 0, 0, 0)
    store = NumpyStore()
    store.add(chunks, vectors)
    retriever = Retriever(store, StubEmbedder(_unit(1, 0, 0, 0, 0, 0, 0, 0)), chunks)

    hits = retriever.retrieve("alpha beta", mode="hybrid")
    top = hits[0]
    assert top.vector_rank is not None and top.keyword_rank is not None


def test_chunks_matching_no_query_term_are_not_fused_in(retriever: Retriever):
    hits = retriever.retrieve("escalation", mode="keyword")
    assert [hit.chunk.chunk_id for hit in hits] == ["doc::003"]


def test_query_with_no_usable_tokens_returns_nothing_from_keywords(retriever: Retriever):
    assert retriever.retrieve("!!! ???", mode="keyword") == []


def test_vector_mode_never_consults_bm25(retriever: Retriever):
    hits = retriever.retrieve("FRM-CRB lead time", mode="vector")
    assert all(hit.keyword_rank is None for hit in hits)
    assert hits[0].chunk.chunk_id == "doc::002"


def test_keyword_mode_never_embeds_the_query(retriever: Retriever):
    before = retriever.embedder.calls  # type: ignore[attr-defined]
    retriever.retrieve("penalty", mode="keyword")
    assert retriever.embedder.calls == before  # type: ignore[attr-defined]


def test_unknown_mode_is_rejected(retriever: Retriever):
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        retriever.retrieve("anything", mode="semantic")


def test_retrieval_is_deterministic(retriever: Retriever):
    first = [hit.chunk.chunk_id for hit in retriever.retrieve("penalty per day")]
    second = [hit.chunk.chunk_id for hit in retriever.retrieve("penalty per day")]
    assert first == second


def test_top_k_is_respected(retriever: Retriever):
    assert len(retriever.retrieve("penalty", k=2)) <= 2
    assert retriever.retrieve("penalty", k=0) == []
