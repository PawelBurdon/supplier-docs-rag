"""Tests for the reranker's cache and its handling of a model that misbehaves.

No network: `_call_api` is stubbed. What matters here is not that the model
ranks well, it is that a malformed ranking cannot shorten the context or show
the same passage twice, and that a cache entry can never be served for a
different model or a different candidate list.
"""

import json
from pathlib import Path

import pytest

from src.core.chunker import Chunk
from src.core.reranker import RerankCacheMiss, Reranker, _parse_ranking
from src.core.retriever import RetrievedChunk


def _candidate(index: int) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"doc::{index:03d}",
        doc_id="doc",
        doc_title="Test Document",
        source_path="doc.md",
        section=f"{index}. Section",
        text=f"Passage {index}.",
        index_text=f"Passage {index}.",
        ordinal=index,
    )
    return RetrievedChunk(chunk=chunk, score=1.0 / (index + 1), vector_rank=index + 1)


@pytest.fixture
def candidates() -> list[RetrievedChunk]:
    return [_candidate(index) for index in range(6)]


def _reranker(tmp_path: Path, order: list[int], monkeypatch) -> Reranker:
    instance = Reranker(cache_dir=tmp_path, api_key="test-key")
    calls: list[str] = []

    def fake_call(question: str, candidates: list[RetrievedChunk]) -> list[int]:
        calls.append(question)
        instance.api_calls += 1
        return order

    monkeypatch.setattr(instance, "_call_api", fake_call)
    instance.calls = calls  # type: ignore[attr-defined]
    return instance


def test_rerank_returns_the_model_order_truncated_to_top_k(tmp_path, candidates, monkeypatch):
    reranker = _reranker(tmp_path, [4, 2, 0, 1, 3, 5], monkeypatch)
    result = reranker.rerank("a question", candidates, top_k=3)
    assert [hit.chunk.chunk_id for hit in result] == ["doc::004", "doc::002", "doc::000"]


def test_reranked_hits_keep_their_retrieval_provenance(tmp_path, candidates, monkeypatch):
    reranker = _reranker(tmp_path, [5, 4, 3, 2, 1, 0], monkeypatch)
    result = reranker.rerank("a question", candidates, top_k=2)
    # The objects are the ones retrieval produced, so --verbose can still show
    # which path found each chunk after reranking moved it.
    assert result[0].vector_rank == 6
    assert result[0].score == pytest.approx(1 / 6)


def test_second_call_is_served_from_cache(tmp_path, candidates, monkeypatch):
    reranker = _reranker(tmp_path, [1, 0, 2, 3, 4, 5], monkeypatch)
    first = reranker.rerank("a question", candidates, top_k=2)
    reranker.calls.clear()  # type: ignore[attr-defined]
    second = reranker.rerank("a question", candidates, top_k=2)
    assert reranker.calls == []  # type: ignore[attr-defined]
    assert [hit.chunk.chunk_id for hit in first] == [hit.chunk.chunk_id for hit in second]


def test_a_different_candidate_list_is_a_different_cache_entry(tmp_path, candidates, monkeypatch):
    reranker = _reranker(tmp_path, [1, 0, 2, 3, 4, 5], monkeypatch)
    reranker.rerank("a question", candidates, top_k=2)
    reranker.calls.clear()  # type: ignore[attr-defined]
    reranker.rerank("a question", candidates[:5], top_k=2)
    assert reranker.calls, "a shorter candidate list must not reuse the cached ordering"


def test_reordering_the_candidates_is_a_different_cache_entry(tmp_path, candidates, monkeypatch):
    reranker = _reranker(tmp_path, [0, 1, 2, 3, 4, 5], monkeypatch)
    reranker.rerank("a question", candidates, top_k=2)
    reranker.calls.clear()  # type: ignore[attr-defined]
    reranker.rerank("a question", list(reversed(candidates)), top_k=2)
    assert reranker.calls, "the candidate order is part of the judgement"


def test_the_model_name_is_part_of_the_cache_key(tmp_path, candidates):
    one = Reranker(model="gemini-3.1-flash-lite", cache_dir=tmp_path, api_key="k")
    other = Reranker(model="gemini-3.6-flash", cache_dir=tmp_path, api_key="k")
    assert one._cache_path("q", candidates) != other._cache_path("q", candidates)


def test_offline_miss_raises_with_the_command_that_fixes_it(tmp_path, candidates):
    offline = Reranker(cache_dir=tmp_path, api_key="", offline=True)
    with pytest.raises(RerankCacheMiss) as error:
        offline.rerank("a question", candidates, top_k=3)
    assert "index --refresh" in str(error.value)


def test_a_cache_file_describing_another_list_is_ignored(tmp_path, candidates, monkeypatch):
    reranker = _reranker(tmp_path, [0, 1, 2, 3, 4, 5], monkeypatch)
    path = reranker._cache_path("a question", candidates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"order": [0, 1]}), encoding="utf-8")
    reranker.rerank("a question", candidates, top_k=2)
    assert reranker.calls, "a truncated ordering must be treated as a miss"


def test_a_ranking_that_drops_candidates_is_repaired():
    # Dropping a candidate would silently shorten the context.
    assert _parse_ranking(json.dumps({"ranking": [3, 1]}), 4) == [2, 0, 1, 3]


def test_a_ranking_that_repeats_or_invents_candidates_is_repaired():
    # Repeats would show the model the same passage twice; out-of-range numbers
    # are the reranker equivalent of a fabricated citation.
    assert _parse_ranking(json.dumps({"ranking": [2, 2, 9, 1]}), 3) == [1, 0, 2]


def test_an_empty_ranking_falls_back_to_the_fused_order():
    assert _parse_ranking(json.dumps({"ranking": []}), 3) == [0, 1, 2]


def test_non_json_from_the_reranker_is_an_error():
    with pytest.raises(RuntimeError, match="not JSON"):
        _parse_ranking("I think passage 2 is best", 3)


def test_empty_input_needs_no_call(tmp_path, monkeypatch):
    reranker = _reranker(tmp_path, [], monkeypatch)
    assert reranker.rerank("a question", [], top_k=5) == []
    assert reranker.calls == []  # type: ignore[attr-defined]


def test_the_prompt_is_part_of_the_cache_key(tmp_path, candidates):
    from src.core.reranker import PROMPTS

    base = Reranker(cache_dir=tmp_path, api_key="k", prompt="base")
    aware = Reranker(cache_dir=tmp_path, api_key="k", prompt="conflict-aware")
    assert base.prompt != aware.prompt
    assert base._cache_path("q", candidates) != aware._cache_path("q", candidates), (
        "an ordering produced under different instructions must not be served for the other"
    )
    assert set(PROMPTS) == {"base", "conflict-aware"}


def test_an_unknown_prompt_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown reranker prompt"):
        Reranker(cache_dir=tmp_path, api_key="k", prompt="whatever")
