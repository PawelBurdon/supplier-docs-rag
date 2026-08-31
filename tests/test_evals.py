"""Tests for the golden set and the metric arithmetic.

The metric tests exist because a metric that is wrong in the flattering
direction is worse than no metric: it produces a number that looks like
evidence. The refusal pair is the clearest case -- a system that refuses every
question scores a perfect refusal accuracy, and only the false refusal rate
exposes it. There is a test for exactly that.
"""

import json
import re
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
    assert len(questions) == 70
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.type] = counts.get(question.type, 0) + 1
    assert counts == {
        "factual": 28,
        "multi_hop": 18,
        "contradiction": 6,
        "underspecified": 3,
        "unanswerable": 15,
    }


def test_both_phrasing_styles_are_represented():
    questions = load_golden_set()
    styles = {question.phrasing for question in questions}
    assert styles == {"prose", "code"}
    code = [question for question in questions if question.phrasing == "code"]
    assert len(code) >= 10, "too few code-phrased questions to compare the two styles"


def test_code_phrased_questions_really_contain_a_code():
    # The phrasing axis is only meaningful if the label matches the text. A
    # question tagged "code" that reads like prose would quietly flatter BM25.
    pattern = re.compile(r"(HCS-[A-Z]{2,4}-\d{4}|[A-Z]{3}-[A-Z]{3}(-\d+)?|clause \d)")
    for question in load_golden_set():
        if question.phrasing == "code":
            assert pattern.search(question.question), f"{question.id} is tagged code but reads as prose"


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
        "anchors": ["some phrase"],
    }
    with pytest.raises(ValueError, match="Duplicate"):
        load_golden_set(_write(tmp_path, [entry, dict(entry)]))


def test_unknown_phrasing_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown phrasing"):
        load_golden_set(
            _write(
                tmp_path,
                [{"id": "x", "type": "factual", "question": "?", "phrasing": "shouty",
                  "expected_documents": ["delivery_sla"]}],
            )
        )


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
    # "text" is what StubRetriever puts in every chunk it returns, so an anchor
    # of "text" is found whenever anything at all was retrieved. These tests are
    # about the document-level arithmetic; anchor scoring has its own tests.
    return GoldenQuestion(
        id="t1",
        type=question_type,
        question="?",
        expected_documents=expected,
        anchors=["text"],
    )


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


def test_write_results_records_the_context_needed_to_read_the_numbers(tmp_path: Path):
    from src.evals.run_evals import write_results

    questions = load_golden_set()
    retriever = StubRetriever(["delivery_sla"])
    results_by_mode = {"hybrid": evaluate_retrieval(retriever, questions, k=5)}
    answers = [_answer("factual", refused=False), _answer("unanswerable", refused=True)]

    path = write_results(
        results_by_mode,
        answers,
        questions,
        k=5,
        generation_model="gemini-3.1-flash-lite",
        embedding_model="gemini-embedding-001",
        path=tmp_path / "results.json",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    # A metric without its date, model and sample size is what the README warns
    # against, so the file has to carry all three.
    assert payload["generated_on"]
    assert payload["generation_model"] == "gemini-3.1-flash-lite"
    assert payload["embedding_model"] == "gemini-embedding-001"
    assert payload["golden_set"]["questions"] == 70
    assert payload["golden_set"]["by_type"]["unanswerable"] == 15
    assert payload["golden_set"]["answerable"] == 55
    assert payload["retrieval"]["hybrid"]["recall"] >= 0
    assert payload["answers"]["refusal_accuracy"] == 1.0


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_every_anchor_is_a_literal_fragment_of_an_expected_document():
    # The whole point of an anchor is that it is text from the corpus, not a
    # paraphrase of it. A paraphrase would silently score zero for ever and look
    # like a retrieval failure.
    from src.core.chunker import chunk_documents

    chunks = chunk_documents(load_documents(DOCUMENTS_DIR))
    by_document: dict[str, list[str]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.doc_id, []).append(_flat(chunk.text))

    for question in load_golden_set():
        for anchor in question.anchors:
            found = any(
                _flat(anchor) in text
                for doc_id in question.expected_documents
                for text in by_document.get(doc_id, [])
            )
            assert found, f"{question.id}: anchor is not literal text of an expected document: {anchor!r}"


def test_multi_source_questions_have_an_anchor_per_document():
    for question in load_golden_set():
        if question.type in ("multi_hop", "contradiction"):
            assert len(question.anchors) == len(question.expected_documents), question.id


def test_anchor_recall_scores_the_passage_not_the_file():
    # The q22 shape: the right document is retrieved, the wrong section of it.
    question = GoldenQuestion(
        id="t1",
        type="factual",
        question="?",
        expected_documents=["delivery_sla"],
        anchors=["the sentence that answers it"],
    )
    result = evaluate_retrieval(StubRetriever(["delivery_sla"]), [question], k=5)[0]
    assert result.recall == 1.0, "document recall cannot see the miss"
    assert result.anchor_recall == 0.0, "anchor recall must see it"
    assert result.anchor_full == 0.0


def test_anchor_recall_gives_partial_credit_across_documents():
    question = GoldenQuestion(
        id="t1",
        type="multi_hop",
        question="?",
        expected_documents=["delivery_sla", "internal_faq_procurement"],
        anchors=["text", "a phrase that is not retrieved"],
    )
    result = evaluate_retrieval(
        StubRetriever(["delivery_sla", "internal_faq_procurement"]), [question], k=5
    )[0]
    assert result.anchor_recall == pytest.approx(0.5)
    assert result.anchor_full == 0.0
    assert result.full_coverage == 1.0, "document coverage still reports both files"


def test_anchors_are_matched_across_line_wrapping():
    # The corpus is hard-wrapped, so the answering sentence spans two lines.
    question = GoldenQuestion(
        id="t1",
        type="factual",
        question="?",
        expected_documents=["delivery_sla"],
        anchors=["text text"],
    )

    class WrappedRetriever(StubRetriever):
        def retrieve(self, question: str, k: int, mode: str = "hybrid"):
            hits = super().retrieve(question, k=k, mode=mode)
            chunk = hits[0].chunk
            wrapped = Chunk(**{**chunk.__dict__, "text": "text\ntext"})
            return [RetrievedChunk(chunk=wrapped, score=0.1)]

    result = evaluate_retrieval(WrappedRetriever(["delivery_sla"]), [question], k=5)[0]
    assert result.anchor_recall == 1.0


def test_answerable_question_without_an_anchor_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one anchor"):
        load_golden_set(
            _write(
                tmp_path,
                [{"id": "x", "type": "factual", "question": "?",
                  "expected_documents": ["delivery_sla"]}],
            )
        )


def test_unanswerable_question_with_an_anchor_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="must not have anchors"):
        load_golden_set(
            _write(
                tmp_path,
                [{"id": "x", "type": "unanswerable", "question": "?", "anchors": ["nope"]}],
            )
        )


def test_the_two_cohorts_are_both_present_and_correctly_sized():
    questions = load_golden_set()
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.cohort] = counts.get(question.cohort, 0) + 1
    assert counts == {"original": 50, "extended": 20}


def test_an_unknown_cohort_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown cohort"):
        load_golden_set(
            _write(
                tmp_path,
                [{"id": "x", "type": "factual", "question": "?", "cohort": "later",
                  "expected_documents": ["delivery_sla"], "anchors": ["net 45 days"]}],
            )
        )
