"""The evaluation harness. This is the part of the project that makes a claim checkable.

Two halves, deliberately separated.

Retrieval metrics need no model and no API key. Recall@k and MRR are computed
from the golden set and the retrieved document ids, so they run in CI on every
commit, deterministically, off the committed embedding cache. A regression in
chunking, tokenisation or fusion shows up as a number that moved.

Answer metrics need a key, because they need answers. Citation validity, refusal
accuracy and the false refusal rate are computed from the structured fields the
answerer already validated, without a judge model. That keeps them cheap and
reproducible, and it keeps this project from measuring one language model with
another.

Definitions worth stating, because each is a choice:

Recall is computed over documents, not chunk ids. A chunk id changes whenever
the chunker changes, which would make the golden set unmaintainable and, worse,
would make two chunking strategies incomparable. The cost is that retrieving the
wrong section of the right document counts as a hit.

Questions needing two documents get two numbers: recall with partial credit
(one of two expected documents is 0.5) and full coverage (both, or nothing).
For a contradiction question only full coverage means anything -- a system
holding one side of a disagreement cannot report a disagreement, however good
its recall looks.

Unanswerable questions are excluded from recall and MRR. There is no document
that should be retrieved, so scoring retrieval on them would average a
meaningless zero into a real number.

A refusal is the `refused` field, and nothing else. No phrase matching. If the
model writes "the documents do not state this" while leaving the field false,
that is a defect of the system and this harness is built to show it, not to
paper over it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.retriever import Retriever
from src.rag.answerer import Answerer

GOLDEN_SET_PATH = Path(__file__).with_name("golden_set.yaml")
QUESTION_TYPES = ("factual", "multi_hop", "contradiction", "unanswerable")
RETRIEVAL_MODES = ("hybrid", "vector", "keyword")


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    type: str
    question: str
    expected_documents: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    expected_answer: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.type != "unanswerable"


@dataclass(frozen=True)
class RetrievalResult:
    question: GoldenQuestion
    recall: float
    full_coverage: float
    reciprocal_rank: float
    retrieved_documents: list[str]


@dataclass(frozen=True)
class AnswerResult:
    question: GoldenQuestion
    refused: bool
    conflict: bool
    grounded: bool
    invalid_citations: list[int]
    citation_count: int
    facts_found: int
    facts_expected: int
    text: str

    @property
    def refusal_correct(self) -> bool:
        return self.refused != self.question.is_answerable


def load_golden_set(path: str | Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    """Load and validate the golden set. A malformed entry fails loudly here."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} does not contain a list of questions")

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for entry in raw:
        question = GoldenQuestion(
            id=str(entry["id"]),
            type=str(entry["type"]),
            question=str(entry["question"]),
            expected_documents=list(entry.get("expected_documents") or []),
            expected_facts=[str(fact) for fact in entry.get("expected_facts") or []],
            expected_answer=str(entry.get("expected_answer", "")),
        )
        if question.type not in QUESTION_TYPES:
            raise ValueError(f"{question.id}: unknown type {question.type!r}")
        if question.id in seen:
            raise ValueError(f"Duplicate question id {question.id}")
        if question.is_answerable and not question.expected_documents:
            raise ValueError(f"{question.id}: an answerable question needs expected_documents")
        if not question.is_answerable and question.expected_documents:
            raise ValueError(
                f"{question.id}: an unanswerable question must not expect any document"
            )
        seen.add(question.id)
        questions.append(question)
    return questions


def evaluate_retrieval(
    retriever: Retriever,
    questions: list[GoldenQuestion],
    k: int,
    mode: str = "hybrid",
) -> list[RetrievalResult]:
    """Score retrieval for every answerable question. No model is called."""
    results: list[RetrievalResult] = []
    for question in questions:
        if not question.is_answerable:
            continue
        retrieved = retriever.retrieve(question.question, k=k, mode=mode)
        documents: list[str] = []
        for hit in retrieved:
            if hit.chunk.doc_id not in documents:
                documents.append(hit.chunk.doc_id)

        expected = set(question.expected_documents)
        found = expected & set(documents)
        reciprocal_rank = 0.0
        for position, doc_id in enumerate(documents, start=1):
            if doc_id in expected:
                reciprocal_rank = 1.0 / position
                break

        results.append(
            RetrievalResult(
                question=question,
                recall=len(found) / len(expected),
                full_coverage=1.0 if found == expected else 0.0,
                reciprocal_rank=reciprocal_rank,
                retrieved_documents=documents,
            )
        )
    return results


def summarise_retrieval(results: list[RetrievalResult]) -> dict[str, float]:
    if not results:
        return {"questions": 0, "recall": 0.0, "full_coverage": 0.0, "mrr": 0.0}
    return {
        "questions": len(results),
        "recall": statistics.fmean(result.recall for result in results),
        "full_coverage": statistics.fmean(result.full_coverage for result in results),
        "mrr": statistics.fmean(result.reciprocal_rank for result in results),
    }


def evaluate_answers(
    answerer: Answerer, questions: list[GoldenQuestion], k: int
) -> list[AnswerResult]:
    """Answer every question and score the structured fields. Needs an API key."""
    results: list[AnswerResult] = []
    for question in questions:
        answer = answerer.answer(question.question, k=k)
        lowered = answer.text.lower()
        results.append(
            AnswerResult(
                question=question,
                refused=answer.refused,
                conflict=answer.conflict,
                grounded=answer.is_grounded,
                invalid_citations=list(answer.invalid_citations),
                citation_count=len(answer.citations),
                facts_found=sum(1 for fact in question.expected_facts if fact.lower() in lowered),
                facts_expected=len(question.expected_facts),
                text=answer.text,
            )
        )
    return results


def summarise_answers(results: list[AnswerResult]) -> dict[str, float]:
    """Aggregate the answer metrics.

    False refusal rate is reported separately from refusal accuracy on purpose.
    A system that refuses everything scores a perfect refusal accuracy, and only
    the false refusal rate exposes it.
    """
    if not results:
        return {}
    answerable = [result for result in results if result.question.is_answerable]
    unanswerable = [result for result in results if not result.question.is_answerable]
    contradictions = [result for result in results if result.question.type == "contradiction"]
    others = [result for result in answerable if result.question.type != "contradiction"]
    with_facts = [result for result in answerable if result.facts_expected]

    return {
        "questions": len(results),
        "citation_validity": statistics.fmean(1.0 if r.grounded else 0.0 for r in results),
        "refusal_accuracy": (
            statistics.fmean(1.0 if r.refused else 0.0 for r in unanswerable)
            if unanswerable
            else 0.0
        ),
        "false_refusal_rate": (
            statistics.fmean(1.0 if r.refused else 0.0 for r in answerable) if answerable else 0.0
        ),
        "conflict_detection": (
            statistics.fmean(1.0 if r.conflict else 0.0 for r in contradictions)
            if contradictions
            else 0.0
        ),
        "false_conflict_rate": (
            statistics.fmean(1.0 if r.conflict else 0.0 for r in others) if others else 0.0
        ),
        "fact_coverage": (
            statistics.fmean(
                1.0 if r.facts_found == r.facts_expected else 0.0 for r in with_facts
            )
            if with_facts
            else 0.0
        ),
        "unanswerable_questions": len(unanswerable),
        "answerable_questions": len(answerable),
    }


def print_retrieval_report(
    results_by_mode: dict[str, list[RetrievalResult]], k: int, primary: str = "hybrid"
) -> None:
    """Per-question table first, aggregates second. A mean hides a regression."""
    results = results_by_mode[primary]
    print(f"\nRETRIEVAL, k={k}, mode={primary}")
    print(f"{'id':5} {'type':14} {'R@k':>5} {'full':>5} {'RR':>5}  retrieved documents")
    print("-" * 100)
    for result in results:
        documents = ", ".join(_short(doc_id) for doc_id in result.retrieved_documents)
        print(
            f"{result.question.id:5} {result.question.type:14} "
            f"{result.recall:5.2f} {result.full_coverage:5.0f} {result.reciprocal_rank:5.2f}  "
            f"{documents}"
        )

    print(f"\nBY QUESTION TYPE (mode={primary})")
    print(f"{'type':14} {'n':>3} {'recall@k':>9} {'full':>6} {'MRR':>6}")
    print("-" * 45)
    for question_type in QUESTION_TYPES:
        subset = [r for r in results if r.question.type == question_type]
        if not subset:
            continue
        summary = summarise_retrieval(subset)
        print(
            f"{question_type:14} {summary['questions']:3d} {summary['recall']:9.3f} "
            f"{summary['full_coverage']:6.2f} {summary['mrr']:6.3f}"
        )

    print("\nWHAT HYBRID BOUGHT")
    print(f"{'mode':10} {'recall@k':>9} {'full':>6} {'MRR':>6}")
    print("-" * 35)
    for mode, mode_results in results_by_mode.items():
        summary = summarise_retrieval(mode_results)
        print(
            f"{mode:10} {summary['recall']:9.3f} {summary['full_coverage']:6.2f} "
            f"{summary['mrr']:6.3f}"
        )


def print_answer_report(results: list[AnswerResult]) -> None:
    print("\nANSWERS")
    print(
        f"{'id':5} {'type':14} {'refused':>8} {'conflict':>9} {'cited':>6} "
        f"{'grounded':>9} {'facts':>6}"
    )
    print("-" * 70)
    for result in results:
        facts = (
            f"{result.facts_found}/{result.facts_expected}" if result.facts_expected else "-"
        )
        flag = "" if result.refusal_correct else "  <- wrong refusal decision"
        print(
            f"{result.question.id:5} {result.question.type:14} {str(result.refused):>8} "
            f"{str(result.conflict):>9} {result.citation_count:6d} "
            f"{str(result.grounded):>9} {facts:>6}{flag}"
        )

    summary = summarise_answers(results)
    print("\nANSWER METRICS")
    print(f"  citation validity     {summary['citation_validity']:.3f}   "
          f"(share of answers where every citation maps to a retrieved passage)")
    print(f"  refusal accuracy      {summary['refusal_accuracy']:.3f}   "
          f"(unanswerable questions correctly refused, n={summary['unanswerable_questions']})")
    print(f"  false refusal rate    {summary['false_refusal_rate']:.3f}   "
          f"(answerable questions wrongly refused, n={summary['answerable_questions']})")
    print(f"  conflict detection    {summary['conflict_detection']:.3f}   "
          f"(contradiction questions flagged as conflicting)")
    print(f"  false conflict rate   {summary['false_conflict_rate']:.3f}   "
          f"(other answerable questions wrongly flagged)")
    print(f"  fact coverage         {summary['fact_coverage']:.3f}   "
          f"(answers containing every expected figure)")

    bad = [result for result in results if not result.grounded]
    for result in bad:
        print(f"\n  {result.question.id}: fabricated citations {result.invalid_citations}")


def _short(doc_id: str) -> str:
    """Shorten doc ids so the per-question table fits a terminal."""
    return doc_id.replace("onboarding_", "onb_").replace("_procurement", "").replace(
        "supplier_framework_agreement", "framework"
    ).replace("returns_and_warranty_policy", "returns").replace(
        "quality_inspection_procedure", "quality"
    )


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point, so the harness can run without the CLI."""
    import argparse

    from src.main import build_answerer, build_retriever, enable_system_trust_store

    parser = argparse.ArgumentParser(description="Run the golden set.")
    parser.add_argument("--retrieval-only", action="store_true", help="No API key needed.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default=None)
    arguments = parser.parse_args(argv)

    enable_system_trust_store()
    questions = load_golden_set()
    retriever = build_retriever(offline=arguments.retrieval_only)
    results_by_mode = {
        mode: evaluate_retrieval(retriever, questions, k=arguments.k, mode=mode)
        for mode in RETRIEVAL_MODES
    }
    print_retrieval_report(results_by_mode, k=arguments.k)

    if not arguments.retrieval_only:
        answerer = build_answerer(retriever, model=arguments.model)
        print_answer_report(evaluate_answers(answerer, questions, k=arguments.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
