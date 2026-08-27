"""Tests for the golden set and the metric arithmetic.

The metric tests exist because a metric that is wrong in the flattering
direction is worse than no metric: it produces a number that looks like
evidence. The refusal pair is the clearest case -- a system that refuses every
question scores a perfect refusal accuracy, and only the false refusal rate
exposes it. There is a test for exactly that.
"""

from pathlib import Path

import pytest
import yaml

from src.core.chunker import Chunk
from src.core.loader import load_documents
from src.core.retriever import RetrievedChunk
from src.evals.run_evals import (
    AnswerResult,
    GoldenQuestion,
    evaluate_retrieval,
    load_golden_set,
    summarise_answers,
    summarise_retrieval,
)

DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"


def _chunk(doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::000",
        doc_id=doc_id,
        doc_title=doc_id,
        source_path=f"{doc_id}.md",
        section="1. Section",
        text="text",
        index_text="text",
        ordinal=0,
    )


class StubRetriever:
    """Returns a fixed document order, so the metric maths is what is under test."""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    def retrieve(self, question: str, k: int, mode: str = "hybrid") -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk=_chunk(doc), score=0.1) for doc in self.documents[:k]]


def test_golden_set_has_the_planned_shape():
    questions = load_golden_set()
    assert len(questions) == 18
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.type] = counts.get(question.type, 0) + 1
    assert counts == {"factual": 7, "multi_hop": 4, "contradiction": 3, "unanswerable": 4}


def test_every_expected_document_exists_on_disk():
    # A typo in the golden set would otherwise show up as a permanent recall
    # failure that looks like a retrieval bug.
    real = {document.doc_id for document in load_documents(DOCUMENTS_DIR)}
    for question in load_golden_set():
        for doc_id in question.expected_documents:
            assert doc_id in real, f"{question.id} expects unknown document {doc_id}"


def test_multi_source_questions_really_need_two_documents():
    for question in load_golden_set():
        if question.type in ("multi_hop", "contradiction"):
            assert len(question.expected_documents) >= 2, question.id


def test_unanswerable_questions_expect_nothing():
    for question in load_golden_set():
        if question.type == "unanswerable":
            assert question.expected_documents == []
            assert question.expected_facts == []


def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def test_duplicate_ids_are_rejected(tmp_path: Path):
    entry = {
        "id": "q01",
        "type": "factual",
        "question": "?",
        "expected_documents": ["delivery_sla"],
    }
    with pytest.raises(ValueError, match="Duplicate"):
        load_golden_set(_write(tmp_path, [entry, dict(entry)]))


def test_unknown_type_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown type"):
        load_golden_set(
            _write(tmp_path, [{"id": "x", "type": "trick", "question": "?", "expected_documents": ["a"]}])
        )


def test_answerable_question_without_expected_documents_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="expected_documents"):
        load_golden_set(_write(tmp_path, [{"id": "x", "type": "factual", "question": "?"}]))


def test_unanswerable_question_with_expected_documents_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="must not expect"):
        load_golden_set(
            _write(
                tmp_path,
                [{"id": "x", "type": "unanswerable", "question": "?", "expected_documents": ["a"]}],
            )
        )


def _question(expected: list[str], question_type: str = "multi_hop") -> GoldenQuestion:
    return GoldenQuestion(id="t1", type=question_type, question="?", expected_documents=expected)


def test_recall_gives_partial_credit_and_full_coverage_does_not():
    question = _question(["delivery_sla", "onboarding_velocore_components"])
    retriever = StubRetriever(["internal_faq_procurement", "delivery_sla"])
    result = evaluate_retrieval(retriever, [question], k=5)[0]
    assert result.recall == pytest.approx(0.5)
    assert result.full_coverage == 0.0
    assert result.reciprocal_rank == pytest.approx(0.5)


def test_full_coverage_needs_every_expected_document():
    question = _question(["delivery_sla", "onboarding_velocore_components"])
    retriever = StubRetriever(["delivery_sla", "onboarding_velocore_components"])
    result = evaluate_retrieval(retriever, [question], k=5)[0]
    assert result.recall == pytest.approx(1.0)
    assert result.full_coverage == 1.0
    assert result.reciprocal_rank == pytest.approx(1.0)


def test_reciprocal_rank_uses_the_first_expected_document():
    question = _question(["quality_inspection_procedure"], question_type="factual")
    retriever = StubRetriever(["a", "b", "quality_inspection_procedure"])
    assert evaluate_retrieval(retriever, [question], k=5)[0].reciprocal_rank == pytest.approx(1 / 3)


def test_a_missed_document_scores_zero_not_undefined():
    question = _question(["delivery_sla"], question_type="factual")
    result = evaluate_retrieval(StubRetriever(["a", "b"]), [question], k=5)[0]
    assert (result.recall, result.full_coverage, result.reciprocal_rank) == (0.0, 0.0, 0.0)


def test_unanswerable_questions_are_excluded_from_retrieval_metrics():
    questions = [
        _question(["delivery_sla"], question_type="factual"),
        GoldenQuestion(id="u1", type="unanswerable", question="?"),
    ]
    results = evaluate_retrieval(StubRetriever(["delivery_sla"]), questions, k=5)
    assert [result.question.id for result in results] == ["t1"]


def test_retrieval_summary_averages_over_questions():
    questions = [
        _question(["delivery_sla"], question_type="factual"),
        _question(["internal_faq_procurement"], question_type="factual"),
    ]
    # Give the second question a different id so both are kept.
    questions[1] = GoldenQuestion(
        id="t2", type="factual", question="?", expected_documents=["internal_faq_procurement"]
    )
    results = evaluate_retrieval(StubRetriever(["delivery_sla"]), questions, k=5)
    summary = summarise_retrieval(results)
    assert summary["questions"] == 2
    assert summary["recall"] == pytest.approx(0.5)


def _answer(question_type: str, refused: bool, conflict: bool = False, grounded: bool = True):
    return AnswerResult(
        question=GoldenQuestion(
            id=question_type,
            type=question_type,
            question="?",
            expected_documents=[] if question_type == "unanswerable" else ["delivery_sla"],
        ),
        refused=refused,
        conflict=conflict,
        grounded=grounded,
        invalid_citations=[] if grounded else [9],
        citation_count=1,
        facts_found=1,
        facts_expected=1,
        text="text",
    )


def test_a_system_that_refuses_everything_is_caught_by_the_false_refusal_rate():
    results = [
        _answer("factual", refused=True),
        _answer("multi_hop", refused=True),
        _answer("unanswerable", refused=True),
    ]
    summary = summarise_answers(results)
    assert summary["refusal_accuracy"] == 1.0, "refusing everything looks perfect here..."
    assert summary["false_refusal_rate"] == 1.0, "...and only this number shows the truth"


def test_a_system_that_never_refuses_scores_zero_refusal_accuracy():
    summary = summarise_answers(
        [_answer("factual", refused=False), _answer("unanswerable", refused=False)]
    )
    assert summary["refusal_accuracy"] == 0.0
    assert summary["false_refusal_rate"] == 0.0


def test_citation_validity_counts_answers_not_citations():
    summary = summarise_answers(
        [
            _answer("factual", refused=False, grounded=True),
            _answer("multi_hop", refused=False, grounded=False),
        ]
    )
    assert summary["citation_validity"] == pytest.approx(0.5)


def test_conflict_metrics_separate_the_two_directions():
    summary = summarise_answers(
        [
            _answer("contradiction", refused=False, conflict=True),
            _answer("factual", refused=False, conflict=True),
        ]
    )
    assert summary["conflict_detection"] == 1.0
    assert summary["false_conflict_rate"] == 1.0
