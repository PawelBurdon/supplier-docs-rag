"""Command line interface.

    python -m src.main index
    python -m src.main ask "what is the lead time for frames?"
    python -m src.main ask "..." --verbose
    python -m src.main eval
    python -m src.main eval --retrieval-only

Every command builds the pipeline the same way: load documents, chunk them, read
their embeddings from the on-disk cache, and put them in a store. That takes
milliseconds for the numpy store and about a second for Chroma, so there is no
separate "load the index" path to keep in sync with the documents. `index` exists
to persist a store and report what was built; at a scale where rebuilding is not
free, that persisted store is what the other commands would load instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DOCUMENTS_DIR = Path("documents")
RERANK_PROMPT_DEFAULT = "conflict-aware"
NUMPY_INDEX_DIR = Path("index/numpy")
CHROMA_DIR = Path("chroma")


def enable_system_trust_store() -> None:
    """Use the operating system trust store for TLS, when truststore is installed.

    On machines running TLS-inspecting security software -- corporate proxies,
    several consumer antivirus products -- the certificate presented to Python is
    signed by a locally installed root that certifi does not know about, and
    every HTTPS call fails with CERTIFICATE_VERIFY_FAILED. The operating system
    does know that root. `truststore` is not a dependency of this project, so
    this is a no-op where it is absent; where it is present it is the difference
    between the project running and not.
    """
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def _load_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))


def build_retriever(
    store_kind: str = "numpy",
    top_k: int = 5,
    offline: bool = False,
    documents_dir: Path = DOCUMENTS_DIR,
    rerank: bool = False,
    rerank_prompt: str | None = None,
):
    """Load, chunk, embed and index the corpus, and return a retriever over it."""
    from src.core.chunker import chunk_documents
    from src.core.embedder import Embedder
    from src.core.loader import load_documents
    from src.core.reranker import Reranker
    from src.core.retriever import Retriever
    from src.core.store import ChromaStore, NumpyStore

    chunks = chunk_documents(load_documents(documents_dir))
    embedder = Embedder(offline=offline)
    vectors = embedder.embed_documents([chunk.index_text for chunk in chunks])

    if store_kind == "numpy":
        store = NumpyStore()
    elif store_kind == "chroma":
        store = ChromaStore(path=CHROMA_DIR)
    else:
        raise ValueError(f"Unknown store {store_kind!r}; expected 'numpy' or 'chroma'")
    store.add(chunks, vectors)
    reranker = (
        Reranker(offline=offline, **({"prompt": rerank_prompt} if rerank_prompt else {}))
        if rerank
        else None
    )
    return Retriever(store, embedder, chunks, top_k=top_k, reranker=reranker)


def build_answerer(retriever, model: str | None = None):
    from src.rag.answerer import DEFAULT_GENERATION_MODEL, Answerer

    return Answerer(retriever, model=model or DEFAULT_GENERATION_MODEL)


def command_index(arguments: argparse.Namespace) -> int:
    from src.core.store import ChromaStore, NumpyStore

    if arguments.refresh and arguments.store == "numpy" and NUMPY_INDEX_DIR.exists():
        import shutil

        shutil.rmtree(NUMPY_INDEX_DIR)

    retriever = build_retriever(store_kind=arguments.store, offline=arguments.offline)
    store = retriever.store
    destination = NUMPY_INDEX_DIR if isinstance(store, NumpyStore) else CHROMA_DIR
    store.persist(destination)

    # Warm the query side of the cache too. Without this, the committed cache
    # covers the documents but not the golden set, and CI fails the first time
    # someone adds a question -- which is a correct failure, but an avoidable
    # one, since indexing is exactly when it should be fixed.
    from src.evals.run_evals import load_golden_set

    questions = [question.question for question in load_golden_set()]
    retriever.embedder.embed_queries(questions)

    reranked = 0
    if arguments.rerank:
        # Same bargain as the embedding cache: pay once with a key so that the
        # metrics run without one. The orderings are keyed on the candidate list,
        # so a change to chunking or fusion invalidates them by design.
        from src.core.reranker import Reranker

        reranker = Reranker(offline=arguments.offline, prompt=arguments.rerank_prompt)
        for question in load_golden_set():
            candidates = retriever.retrieve(question.question, k=retriever.pool)
            reranker.rerank(question.question, candidates, top_k=retriever.top_k)
            reranked += 1

    documents = sorted({chunk.doc_id for chunk in retriever.chunks})
    sizes = [len(chunk.text) for chunk in retriever.chunks]
    print(f"Indexed {len(documents)} documents into {len(retriever.chunks)} chunks")
    print(f"  chunk size: min {min(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)} characters")
    print(f"  embeddings: {retriever.embedder.cache_hits} from cache, "
          f"{retriever.embedder.api_calls} API call(s)")
    print(f"  questions:  {len(questions)} golden set queries cached")
    if arguments.rerank:
        print(f"  reranking:  {reranked} orderings cached")
    print(f"  store:      {type(store).__name__} at {destination}")
    if isinstance(store, ChromaStore):
        print("  note:       Chroma writes during add(); persist() is a no-op")
    return 0


def command_ask(arguments: argparse.Namespace) -> int:
    from src.rag.answerer import AnswerError

    retriever = build_retriever(
        store_kind=arguments.store,
        top_k=arguments.k,
        rerank=arguments.rerank,
        rerank_prompt=arguments.rerank_prompt,
    )
    answerer = build_answerer(retriever, model=arguments.model)
    try:
        answer = answerer.answer(arguments.question, k=arguments.k)
    except AnswerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if arguments.verbose:
        print(f"Retrieved {len(answer.retrieved)} chunks for: {arguments.question}\n")
        for number, hit in enumerate(answer.retrieved, start=1):
            vector = f"{hit.vector_score:.3f}" if hit.vector_score is not None else "  -  "
            keyword = f"{hit.keyword_score:.2f}" if hit.keyword_score is not None else "  -  "
            print(
                f"  [{number}] rrf={hit.score:.5f}  cosine={vector}  bm25={keyword}  "
                f"({hit.paths})"
            )
            print(f"      {hit.chunk.citation_label}  [{hit.chunk.source_path}]")
            print(f"      {_preview(hit.chunk.text)}")
        print()

    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            print(f"  {citation.label}")
    if answer.refused:
        print("\n(refused: the documents do not contain this)")
    if answer.conflict:
        print("\n(sources disagree; both are reported above)")
    if not answer.is_grounded:
        print(
            f"\nwarning: the model cited {answer.invalid_citations}, which were not in its "
            "context. Those citations were removed from the answer above.",
            file=sys.stderr,
        )
    return 0


def command_eval(arguments: argparse.Namespace) -> int:
    from src.rag.answerer import AnswerError
    from src.core.embedder import DEFAULT_MODEL as EMBEDDING_MODEL
    from src.evals.run_evals import (
        RETRIEVAL_MODES,
        evaluate_answers,
        evaluate_retrieval,
        load_golden_set,
        print_answer_report,
        print_retrieval_report,
        write_results,
    )

    questions = load_golden_set()
    retriever = build_retriever(
        store_kind=arguments.store, top_k=arguments.k, offline=arguments.retrieval_only
    )
    results_by_mode = {
        mode: evaluate_retrieval(retriever, questions, k=arguments.k, mode=mode)
        for mode in RETRIEVAL_MODES
    }
    print_retrieval_report(results_by_mode, k=arguments.k)

    if arguments.rerank:
        from src.core.reranker import Reranker
        from src.evals.run_evals import print_rerank_comparison

        # The same retriever, with the reranker attached for a second pass, so
        # the two runs differ in exactly one thing.
        retriever.reranker = Reranker(
            offline=arguments.retrieval_only, prompt=arguments.rerank_prompt
        )
        reranked = evaluate_retrieval(retriever, questions, k=arguments.k, mode="hybrid")
        print_rerank_comparison(results_by_mode["hybrid"], reranked, k=arguments.k)
        # Recorded under its own key. Without it the results file would carry
        # retrieval numbers from fusion alone next to answer numbers from the
        # reranked pipeline, which is two configurations in one artefact.
        results_by_mode["hybrid_reranked"] = reranked
        # The reranker stays attached: with --rerank the answer half measures the
        # pipeline as it would actually run, so the refusal and conflict numbers
        # below describe the reranked system rather than the one just compared
        # against it.

    if arguments.retrieval_only:
        print("\n(retrieval only: answer metrics need an API key)")
        return 0

    answerer = build_answerer(retriever, model=arguments.model)
    print(f"\nAnswering {len(questions)} questions with {answerer.model} ...", flush=True)
    try:
        results = evaluate_answers(answerer, questions, k=arguments.k)
    except AnswerError as error:
        # A rate limit or a blocked generation should end the run with one
        # readable line, not a traceback the reader has to parse.
        print(f"error: {error}", file=sys.stderr)
        print("Retrieval metrics above are unaffected.", file=sys.stderr)
        return 1

    print_answer_report(results)
    written = write_results(
        results_by_mode,
        results,
        questions,
        k=arguments.k,
        generation_model=answerer.model,
        embedding_model=EMBEDDING_MODEL,
        reranker=(
            {
                "enabled": True,
                "model": retriever.reranker.model,
                "prompt": retriever.reranker.prompt_name,
            }
            if retriever.reranker is not None
            else {"enabled": False}
        ),
    )
    print(f"\nMetrics written to {written}")
    return 0


def _preview(text: str, width: int = 96) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else f"{flat[: width - 3]}..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Retrieval-augmented question answering over supplier documents.",
    )
    parser.add_argument(
        "--store", choices=("numpy", "chroma"), default="numpy", help="Vector store to use."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build and persist the index.")
    index_parser.add_argument(
        "--refresh", action="store_true", help="Delete the persisted index before rebuilding."
    )
    index_parser.add_argument(
        "--offline", action="store_true", help="Fail rather than embed anything not cached."
    )
    index_parser.add_argument(
        "--rerank", action="store_true", help="Also cache a reranked ordering per golden question."
    )
    index_parser.add_argument(
        "--rerank-prompt",
        choices=("base", "conflict-aware"),
        default=RERANK_PROMPT_DEFAULT,
        help="Which reranker instruction set to use.",
    )
    index_parser.set_defaults(handler=command_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks and scores.")
    ask_parser.add_argument("--k", type=int, default=5, help="Chunks to retrieve.")
    ask_parser.add_argument("--model", default=None, help="Generation model id.")
    ask_parser.add_argument(
        "--rerank", action="store_true", help="Rerank the fused candidates before answering."
    )
    ask_parser.add_argument(
        "--rerank-prompt",
        choices=("base", "conflict-aware"),
        default=RERANK_PROMPT_DEFAULT,
        help="Which reranker instruction set to use.",
    )
    ask_parser.set_defaults(handler=command_ask)

    eval_parser = subparsers.add_parser("eval", help="Run the golden set.")
    eval_parser.add_argument(
        "--retrieval-only", action="store_true", help="Retrieval metrics only; no API key needed."
    )
    eval_parser.add_argument("--k", type=int, default=5)
    eval_parser.add_argument("--model", default=None, help="Generation model id.")
    eval_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Also score hybrid retrieval with reranking, and print the comparison.",
    )
    eval_parser.add_argument(
        "--rerank-prompt",
        choices=("base", "conflict-aware"),
        default=RERANK_PROMPT_DEFAULT,
        help="Which reranker instruction set to use.",
    )
    eval_parser.set_defaults(handler=command_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    enable_system_trust_store()
    _load_environment()
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
