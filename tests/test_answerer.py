"""Tests for prompt assembly and answer validation. The model is stubbed.

These tests are about what the code does with what the model says, which is the
part that has to be trustworthy. The interesting cases are the dishonest ones: a
citation pointing at a passage that was never supplied, and a citation declared
in the structured field but never written in the text.
"""

import numpy as np
import pytest

from src.core.chunker import Chunk
from src.core.retriever import RetrievedChunk, Retriever
from src.core.store import NumpyStore
from src.rag.answerer import Answerer, AnswerError
from src.rag.prompts import build_context_block, build_user_prompt


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"delivery_sla::{index:03d}",
        doc_id="delivery_sla",
        doc_title="Delivery Service Level Agreement",
        source_path="delivery_sla.md",
        section=f"{index}. Section",
        text=text,
        index_text=text,
        ordinal=index,
    )


def _retrieved(count: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=_chunk(index, f"Passage {index}."), score=0.1)
        for index in range(1, count + 1)
    ]


class StubEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(4, dtype=np.float32)
        vector[0] = 1.0
        return vector


def _answerer(payload: dict, chunk_count: int = 3) -> Answerer:
    """An Answerer whose model always returns `payload`."""
    chunks = [_chunk(index, f"Passage {index}.") for index in range(1, chunk_count + 1)]
    vectors = np.zeros((chunk_count, 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    store = NumpyStore()
    store.add(chunks, vectors)
    retriever = Retriever(store, StubEmbedder(), chunks, top_k=chunk_count)
    answerer = Answerer(retriever)
    answerer._generate = lambda user_prompt: payload  # type: ignore[method-assign]
    return answerer


def test_context_block_numbers_and_labels_every_passage():
    block = build_context_block(_retrieved(2))
    assert block.startswith("[1] Delivery Service Level Agreement, 1. Section (source: delivery_sla.md)")
    assert "[2] Delivery Service Level Agreement, 2. Section" in block
    assert "Passage 1." in block and "Passage 2." in block


def test_user_prompt_carries_the_question_and_the_context():
    prompt = build_user_prompt("What is the lead time?", _retrieved(1))
    assert "Question: What is the lead time?" in prompt
    assert "[1] Delivery Service Level Agreement" in prompt


def test_user_prompt_survives_an_empty_retrieval():
    prompt = build_user_prompt("Anything?", [])
    assert "(no passages were retrieved)" in prompt


def test_valid_citations_map_to_their_chunks():
    answerer = _answerer(
        {
            "answer": "The penalty is 0.5 percent per day [1], capped at 10 percent [2].",
            "citations": [1, 2],
            "refused": False,
            "conflict": False,
        }
    )
    answer = answerer.answer("What is the penalty?")
    assert [citation.number for citation in answer.citations] == [1, 2]
    assert answer.citations[0].chunk.doc_id == "delivery_sla"
    assert answer.is_grounded
    assert answer.cited_documents == ["delivery_sla"]


def test_a_citation_outside_the_context_is_caught_and_removed():
    answerer = _answerer(
        {
            "answer": "The cap is 10 percent [2] and the grace period is five days [7].",
            "citations": [2, 7],
            "refused": False,
            "conflict": False,
        }
    )
    answer = answerer.answer("What is the cap?")
    assert answer.invalid_citations == [7]
    assert not answer.is_grounded
    assert "[7]" not in answer.text
    assert "[2]" in answer.text
    assert [citation.number for citation in answer.citations] == [2]


def test_a_fabricated_citation_declared_but_not_written_is_still_caught():
    answerer = _answerer(
        {
            "answer": "The cap is 10 percent [1].",
            "citations": [1, 9],
            "refused": False,
            "conflict": False,
        }
    )
    answer = answerer.answer("What is the cap?")
    assert answer.invalid_citations == [9]
    assert not answer.is_grounded


def test_citation_order_follows_the_answer_text():
    answerer = _answerer(
        {
            "answer": "First point [3]. Second point [1].",
            "citations": [1, 3],
            "refused": False,
            "conflict": False,
        }
    )
    answer = answerer.answer("Anything?")
    assert [citation.number for citation in answer.citations] == [3, 1]


def test_refusal_and_conflict_flags_are_carried_through():
    refusal = _answerer(
        {
            "answer": "The provided documents do not state the freight insurance limit.",
            "citations": [],
            "refused": True,
            "conflict": False,
        }
    ).answer("What is the insurance limit?")
    assert refusal.refused and not refusal.conflict and refusal.citations == []

    conflict = _answerer(
        {
            "answer": "The SLA says 0.5 percent [1]; the FAQ says 1 percent [2].",
            "citations": [1, 2],
            "refused": False,
            "conflict": True,
        }
    ).answer("What is the penalty?")
    assert conflict.conflict and not conflict.refused


def test_malformed_flags_do_not_crash_the_parser():
    answer = _answerer(
        {"answer": "Something [1].", "citations": ["1", None, True], "refused": "yes", "conflict": 0}
    ).answer("Anything?")
    # Non-integer citation entries are ignored rather than raising; booleans are
    # not integers here, because True would otherwise read as citation 1.
    assert answer.invalid_citations == []
    assert [citation.number for citation in answer.citations] == [1]
    assert answer.refused is True


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        _answerer({"answer": "", "citations": [], "refused": True, "conflict": False}).answer("  ")


def test_missing_api_key_explains_the_retrieval_only_path(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    chunks = [_chunk(1, "Passage 1.")]
    vectors = np.zeros((1, 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    store = NumpyStore()
    store.add(chunks, vectors)
    answerer = Answerer(Retriever(store, StubEmbedder(), chunks))
    with pytest.raises(AnswerError, match="retrieval-only"):
        answerer.answer("What is the lead time?")


def test_retry_delay_is_taken_from_the_error_message():
    from src.rag.answerer import MAX_RETRY_WAIT, _requested_retry_delay

    # The two shapes Google actually returns for a 429.
    assert _requested_retry_delay("429 RESOURCE_EXHAUSTED {'retryDelay': '14s'}") == pytest.approx(14.5)
    assert _requested_retry_delay("Please retry in 14.347677307s.") == pytest.approx(14.847677307)
    # No delay quoted: fall back to the caller's exponential backoff.
    assert _requested_retry_delay("connection reset by peer") == 0.0
    # An absurd delay is capped, so a run fails visibly instead of hanging.
    assert _requested_retry_delay("'retryDelay': '600s'") == MAX_RETRY_WAIT


def test_client_errors_are_not_retried():
    from src.rag.answerer import _is_retryable

    # Worth retrying: transient, or the server told us to wait.
    assert _is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert _is_retryable(RuntimeError("503 UNAVAILABLE"))
    assert _is_retryable(RuntimeError("connection reset by peer"))
    # Not worth retrying: the request itself is wrong and will stay wrong.
    assert not _is_retryable(RuntimeError("404 NOT_FOUND. This model is no longer available"))
    assert not _is_retryable(RuntimeError("400 INVALID_ARGUMENT"))
