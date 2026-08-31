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

import json
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from src.core.retriever import Retriever
from src.rag.answerer import Answerer

GOLDEN_SET_PATH = Path(__file__).with_name("golden_set.yaml")
RESULTS_PATH = Path("evaluation_results.json")
QUESTION_TYPES = (
    "factual",
    "multi_hop",
    "contradiction",
    # A question two documents answer equally well, because it does not say
    # which supplier or which contract it means. Correct behaviour is to give
    # both, or to say the answer depends on that. It is its own type rather than
    # a contradiction: the sources do not disagree, the question is incomplete.
    "underspecified",
    "unanswerable",
)
RETRIEVAL_MODES = ("hybrid", "vector", "keyword")
PHRASINGS = ("prose", "code")
# Which round of the golden set a question belongs to. The original 50 were
# written against a 7-document corpus; the extended 20 were written after it
# grew to 10 and target version conflicts, near-duplicate contracts and
# questions needing three documents. Reported separately and permanently: an
# average over both hides whether a regression landed on the easy half or the
# hard one.
COHORTS = ("original", "extended")


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    type: str
    question: str
    expected_documents: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    expected_answer: str = ""
    # "code" means the question names a document reference, clause number or SKU
    # code. Reported separately because the case for keeping BM25 rests entirely
    # on those questions failing differently, and that should be measured.
    phrasing: str = "prose"
    # Literal fragments of the passages that actually answer the question, one
    # per expected document. Document-level recall cannot tell a hit from a near
    # miss inside the same file; these can. They are phrases rather than chunk
    # ids so that re-chunking does not invalidate the golden set.
    anchors: list[str] = field(default_factory=list)
    cohort: str = "original"

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
    # The same two shapes as above, scored on anchor phrases instead of document
    # ids: how many of the answering passages were actually retrieved, and
    # whether all of them were.
    anchor_recall: float = 0.0
    anchor_full: float = 0.0


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
            phrasing=str(entry.get("phrasing", "prose")),
            anchors=[str(anchor) for anchor in entry.get("anchors") or []],
            cohort=str(entry.get("cohort", "original")),
        )
        if question.type not in QUESTION_TYPES:
            raise ValueError(f"{question.id}: unknown type {question.type!r}")
        if question.phrasing not in PHRASINGS:
            raise ValueError(f"{question.id}: unknown phrasing {question.phrasing!r}")
        if question.cohort not in COHORTS:
            raise ValueError(f"{question.id}: unknown cohort {question.cohort!r}")
        if question.id in seen:
            raise ValueError(f"Duplicate question id {question.id}")
        if question.is_answerable and not question.expected_documents:
            raise ValueError(f"{question.id}: an answerable question needs expected_documents")
        if not question.is_answerable and question.expected_documents:
            raise ValueError(
                f"{question.id}: an unanswerable question must not expect any document"
            )
        if question.is_answerable and not question.anchors:
            raise ValueError(f"{question.id}: an answerable question needs at least one anchor")
        if not question.is_answerable and question.anchors:
            raise ValueError(f"{question.id}: an unanswerable question must not have anchors")
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

        # Anchors are matched on whitespace-normalised text because the corpus is
        # hard-wrapped: the sentence that answers a question is one string to a
        # reader and two lines to str.__contains__.
        retrieved_text = " || ".join(_flatten(hit.chunk.text) for hit in retrieved)
        anchors_found = sum(
            1 for anchor in question.anchors if _flatten(anchor) in retrieved_text
        )

        results.append(
            RetrievalResult(
                question=question,
                recall=len(found) / len(expected),
                full_coverage=1.0 if found == expected else 0.0,
                reciprocal_rank=reciprocal_rank,
                retrieved_documents=documents,
                anchor_recall=anchors_found / len(question.anchors) if question.anchors else 0.0,
                anchor_full=1.0 if question.anchors and anchors_found == len(question.anchors) else 0.0,
            )
        )
    return results


def _flatten(text: str) -> str:
    return " ".join(text.split())


def summarise_retrieval(results: list[RetrievalResult]) -> dict[str, float]:
    """Both granularities, side by side, never one replacing the other.

    Document recall stays because every earlier run was measured with it, and a
    metric swapped out mid-project makes its own history incomparable. Anchor
    recall is the sharper number: it asks whether the passage that answers the
    question was retrieved, not merely whether something from the right file was.
    """
    if not results:
        return {
            "questions": 0,
            "recall": 0.0,
            "full_coverage": 0.0,
            "mrr": 0.0,
            "anchor_recall": 0.0,
            "anchor_full": 0.0,
        }
    return {
        "questions": len(results),
        "recall": statistics.fmean(result.recall for result in results),
        "full_coverage": statistics.fmean(result.full_coverage for result in results),
        "mrr": statistics.fmean(result.reciprocal_rank for result in results),
        "anchor_recall": statistics.fmean(result.anchor_recall for result in results),
        "anchor_full": statistics.fmean(result.anchor_full for result in results),
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
    # Underspecified questions are left out of the false-conflict denominator.
    # An answer that says two contracts differ is right there, and scoring it as
    # a false positive would penalise the behaviour the type exists to reward.
    others = [
        result
        for result in answerable
        if result.question.type not in ("contradiction", "underspecified")
    ]
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


def write_results(
    results_by_mode: dict[str, list[RetrievalResult]],
    answer_results: list[AnswerResult],
    questions: list[GoldenQuestion],
    k: int,
    generation_model: str,
    embedding_model: str,
    path: str | Path = RESULTS_PATH,
    reranker: dict | None = None,
) -> Path:
    """Write the run's metrics and the context needed to read them.

    A metric without its date, its model and its sample size is the thing this
    project spends a README section warning about, so the file carries all
    three. It is written only by a full run: a retrieval-only run has no answer
    metrics, and half a file overwriting a whole one would quietly delete
    numbers rather than update them.
    """
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.type] = counts.get(question.type, 0) + 1

    payload = {
        "generated_on": date.today().isoformat(),
        "generation_model": generation_model,
        "embedding_model": embedding_model,
        # Whether these numbers came from the reranked pipeline, and under which
        # instructions. Without it the file records a score whose configuration
        # has to be guessed.
        "reranker": reranker or {"enabled": False},
        "top_k": k,
        "golden_set": {
            "questions": len(questions),
            "by_type": counts,
            "answerable": sum(1 for question in questions if question.is_answerable),
        },
        "retrieval": {
            mode: summarise_retrieval(results) for mode, results in results_by_mode.items()
        },
        "retrieval_by_cohort": {
            cohort: summarise_retrieval(
                [r for r in results_by_mode.get("hybrid", []) if r.question.cohort == cohort]
            )
            for cohort in COHORTS
        },
        "answers": summarise_answers(answer_results),
    }
    destination = Path(path)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def print_retrieval_report(
    results_by_mode: dict[str, list[RetrievalResult]], k: int, primary: str = "hybrid"
) -> None:
    """Per-question table first, aggregates second. A mean hides a regression."""
    results = results_by_mode[primary]
    print(f"\nRETRIEVAL, k={k}, mode={primary}")
    print(
        f"{'id':5} {'type':14} {'R@k':>5} {'full':>5} {'RR':>5} {'anchR':>6} {'anchF':>6}  "
        "retrieved documents"
    )
    print("-" * 112)
    for result in results:
        documents = ", ".join(_short(doc_id) for doc_id in result.retrieved_documents)
        flag = "" if result.anchor_full else "  <- answering passage missed"
        print(
            f"{result.question.id:5} {result.question.type:14} "
            f"{result.recall:5.2f} {result.full_coverage:5.0f} {result.reciprocal_rank:5.2f} "
            f"{result.anchor_recall:6.2f} {result.anchor_full:6.0f}  "
            f"{documents}{flag}"
        )

    print(f"\nBY QUESTION TYPE (mode={primary})")
    print(f"{'type':14} {'n':>3} {'recall@k':>9} {'full':>6} {'MRR':>6} {'anchor':>7} {'anchF':>6}")
    print("-" * 60)
    for question_type in QUESTION_TYPES:
        subset = [r for r in results if r.question.type == question_type]
        if not subset:
            continue
        summary = summarise_retrieval(subset)
        print(
            f"{question_type:14} {summary['questions']:3d} {summary['recall']:9.3f} "
            f"{summary['full_coverage']:6.2f} {summary['mrr']:6.3f} "
            f"{summary['anchor_recall']:7.3f} {summary['anchor_full']:6.2f}"
        )

    print("\nWHAT HYBRID BOUGHT (document recall first, then anchor recall)")
    print(f"{'mode':10} {'recall@k':>9} {'full':>6} {'MRR':>6} {'anchor':>7} {'anchF':>6}")
    print("-" * 50)
    for mode, mode_results in results_by_mode.items():
        summary = summarise_retrieval(mode_results)
        print(
            f"{mode:10} {summary['recall']:9.3f} {summary['full_coverage']:6.2f} "
            f"{summary['mrr']:6.3f} {summary['anchor_recall']:7.3f} "
            f"{summary['anchor_full']:6.2f}"
        )

    print(f"\nBY COHORT (mode={primary})")
    print(f"{'cohort':10} {'n':>3} {'recall@k':>9} {'full':>6} {'MRR':>6} {'anchor':>7} {'anchF':>6}")
    print("-" * 56)
    for cohort in COHORTS:
        subset = [result for result in results if result.question.cohort == cohort]
        if not subset:
            continue
        summary = summarise_retrieval(subset)
        print(
            f"{cohort:10} {summary['questions']:3d} {summary['recall']:9.3f} "
            f"{summary['full_coverage']:6.2f} {summary['mrr']:6.3f} "
            f"{summary['anchor_recall']:7.3f} {summary['anchor_full']:6.2f}"
        )

    print("\nBY QUESTION PHRASING (document recall / anchor recall)")
    print(f"{'style':7} {'n':>3}  " + "  ".join(f"{mode:>15}" for mode in results_by_mode))
    print("-" * (12 + 17 * len(results_by_mode)))
    for phrasing in PHRASINGS:
        cells = []
        count = 0
        for mode_results in results_by_mode.values():
            subset = [r for r in mode_results if r.question.phrasing == phrasing]
            count = len(subset)
            summary = summarise_retrieval(subset)
            cells.append(f"{summary['recall']:6.3f} / {summary['anchor_recall']:5.3f}")
        print(f"{phrasing:7} {count:3d}  " + "  ".join(f"{cell:>15}" for cell in cells))


def print_rerank_comparison(
    baseline: list[RetrievalResult], reranked: list[RetrievalResult], k: int
) -> None:
    """Fusion alone against fusion plus reranking, on the same questions.

    Printed as a comparison rather than a replacement because a reranker is a
    claim -- that reading candidates against the question beats counting the
    lists they appear in -- and a claim needs a before and an after.
    """
    before = summarise_retrieval(baseline)
    after = summarise_retrieval(reranked)

    print(f"\nWHAT RERANKING BOUGHT, k={k}, hybrid retrieval")
    print(f"{'':22} {'fusion only':>12} {'+ reranker':>12} {'change':>8}")
    print("-" * 58)
    for label, key in (
        ("document recall@k", "recall"),
        ("document full", "full_coverage"),
        ("MRR", "mrr"),
        ("anchor recall@k", "anchor_recall"),
        ("all anchors", "anchor_full"),
    ):
        delta = after[key] - before[key]
        print(f"{label:22} {before[key]:12.3f} {after[key]:12.3f} {delta:+8.3f}")

    moved = [
        (b.question.id, b.anchor_recall, a.anchor_recall)
        for b, a in zip(baseline, reranked)
        if b.anchor_recall != a.anchor_recall
    ]
    if moved:
        print("\nQuestions whose answering passages moved:")
        for question_id, before_value, after_value in moved:
            direction = "recovered" if after_value > before_value else "LOST"
            print(f"  {question_id}: anchor recall {before_value:.2f} -> {after_value:.2f}  {direction}")
    else:
        print("\nNo question changed its anchor recall.")


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
        from src.core.embedder import DEFAULT_MODEL as EMBEDDING_MODEL

        answerer = build_answerer(retriever, model=arguments.model)
        answer_results = evaluate_answers(answerer, questions, k=arguments.k)
        print_answer_report(answer_results)
        write_results(
            results_by_mode,
            answer_results,
            questions,
            k=arguments.k,
            generation_model=answerer.model,
            embedding_model=EMBEDDING_MODEL,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
