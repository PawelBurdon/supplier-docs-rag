"""Streamlit demo: one question, one grounded answer, and the retrieval behind it.

This file is a view. It builds no index, scores no chunks and writes no prompt --
every one of those lives in `src/core` and `src/rag`, and this module calls them
through the same two constructors the CLI uses. If something here looks like it
is making a retrieval decision, it is a bug.

The demo answers one question at a time on purpose. There is no conversation
history, no document upload and no login: the thing worth showing is whether a
single answer is grounded, refused, or contested, and a chat transcript would
bury that.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.core.embedder import EmbeddingCacheMiss
from src.evals.run_evals import RESULTS_PATH, load_golden_set
from src.main import build_answerer, build_retriever, enable_system_trust_store
from src.rag.answerer import AnswerError

REPO_URL = "https://github.com/PawelBurdon/supplier-docs-rag"
TOP_K = 5

# The four examples are golden set questions, chosen one per type so that a
# visitor with a minute sees the four behaviours worth seeing. They are read from
# the golden set by id rather than copied here, so the demo cannot drift from the
# questions the evaluation actually scores.
#
# The unanswerable one matters most. Refusal is the part of this system that took
# the most care, and nobody discovers it by accident -- a visitor types questions
# the documents can answer. q17 is the sharpest of the twelve: a minimum order
# value does exist in the corpus, for the other supplier, so a system that
# pattern-matches instead of reading will confidently produce USD 15,000.
EXAMPLE_IDS = {
    "q27": "Factual, asked by document code",
    "q08": "Multi-hop, needs two documents",
    "q12": "Contradiction, two sources disagree",
    "q17": "Unanswerable, must refuse",
}

_CITATION = re.compile(r"\[(\d+)\]")

load_dotenv(".env")
enable_system_trust_store()


@st.cache_resource(show_spinner="Loading the index ...")
def load_retriever(offline: bool):
    """Build the retriever once per session rather than once per keystroke.

    Streamlit re-runs this whole script on every interaction. Without the cache
    the corpus would be re-chunked and the store rebuilt each time somebody typed
    a character.
    """
    return build_retriever(top_k=TOP_K, offline=offline)


@st.cache_data(show_spinner=False)
def load_examples() -> list[tuple[str, str]]:
    """Return (label, question) for the four demo questions, in the order above."""
    by_id = {question.id: question for question in load_golden_set()}
    return [
        (label, by_id[question_id].question)
        for question_id, label in EXAMPLE_IDS.items()
        if question_id in by_id
    ]


def highlight_citations(text: str) -> str:
    """Make [1] and [2] visible in the answer without changing a character of it."""
    return _CITATION.sub(lambda match: f":blue-background[**[{match.group(1)}]**]", text)


def retrieval_rows(retrieved) -> list[dict]:
    """Flatten the retrieved chunks into table rows.

    Every value here is read off the RetrievedChunk the retriever already
    produced -- per-path ranks and scores are part of its result, precisely so
    that a caller can show its working. Nothing is recomputed.
    """
    rows = []
    for position, hit in enumerate(retrieved, start=1):
        if hit.vector_rank and hit.keyword_rank:
            found_by = "both"
        elif hit.vector_rank:
            found_by = "vector only"
        else:
            found_by = "BM25 only"
        rows.append(
            {
                "#": position,
                "document": hit.chunk.source_path,
                "section": hit.chunk.section or "(document header)",
                "cosine": f"{hit.vector_score:.3f}" if hit.vector_score is not None else "-",
                "vector rank": hit.vector_rank or "-",
                "BM25 rank": hit.keyword_rank or "-",
                "fused score": f"{hit.score:.4f}",
                "found by": found_by,
            }
        )
    return rows


def retrieval_table(retrieved) -> str:
    """Render the rows as a markdown table.

    Not st.dataframe: that widget paints an interactive grid outside the DOM, so
    its contents cannot be selected, copied or read by a screen reader, and it
    arrives with a toolbar this page has no use for. The table here is the point
    of the section, so it is plain text.
    """
    rows = retrieval_rows(retrieved)
    if not rows:
        return "_No chunks were retrieved._"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        cells = (str(row[key]).replace("|", r"\|") for key in headers)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_results() -> dict | None:
    """Read the metrics written by the last full evaluation run, if there is one.

    Static by design: recomputing answer metrics would mean fifty model calls per
    page load. The file carries the run's date, models and sample sizes so the
    numbers cannot be read without their context.
    """
    try:
        return json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def has_api_key() -> bool:
    # The project reads GOOGLE_API_KEY everywhere -- embedder, answerer, CLI --
    # so the demo reads the same variable rather than inventing a second name.
    return bool(os.environ.get("GOOGLE_API_KEY"))


# Wide, because the retrieval table has eight columns and the centered layout
# clips the last one -- which happens to be the column showing whether a chunk
# came from dense search, from BM25, or from both.
st.set_page_config(page_title="supplier-docs-rag", layout="wide")

st.title("supplier-docs-rag")
st.write(
    "Retrieval-augmented question answering over seven supplier and delivery "
    "documents, with citations verified in code and an explicit refusal when the "
    "documents do not contain the answer."
)
st.caption(f"Source and method: {REPO_URL}")

st.divider()

state = st.session_state
state.setdefault("prefill", "")
state.setdefault("nonce", 0)
state.setdefault("run_example", False)

key_present = has_api_key()

# The input is keyed on a nonce so that clicking an example can replace its
# contents. Streamlit refuses to let a widget's own state be rewritten after the
# widget exists, and a changing key is the honest way round that rather than
# holding a second copy of the question somewhere.
question = st.text_input(
    "Question",
    value=state.prefill,
    key=f"question-{state.nonce}",
    placeholder="What penalty do we charge a supplier for a late delivery?",
    disabled=not key_present,
)
asked = st.button("Ask", type="primary", disabled=not key_present)

if not key_present:
    # Free text is disabled rather than left to fail. Without a key the embedder
    # can only read its committed cache, which holds the corpus and the golden
    # set questions -- so a typed question cannot even be retrieved, let alone
    # answered. Letting someone type one and collect an error would read as a
    # broken project rather than a missing key.
    st.info(
        "No GOOGLE_API_KEY is set, so free-text questions are turned off: their "
        "embeddings are not in the committed cache and retrieval itself would fail. "
        "The four examples below still run real retrieval against that cache; only "
        "generating the answer needs a key. Set one with "
        "`GOOGLE_API_KEY=...` in a `.env` file, from https://aistudio.google.com/apikey."
    )

st.caption("Or try one of these, taken from the evaluated golden set:")
for column, (label, example) in zip(st.columns(len(EXAMPLE_IDS)), load_examples()):
    if column.button(label, help=example, use_container_width=True):
        state.prefill = example
        state.nonce += 1
        state.run_example = True
        st.rerun()

asked = asked or state.run_example
state.run_example = False
pending = question or state.prefill


def render_retrieval(retrieved, expanded: bool) -> None:
    with st.expander("How this was retrieved", expanded=expanded):
        st.caption(
            "Dense search and BM25 each rank the corpus, and the two rankings are "
            "fused by reciprocal rank: score = sum of 1/(3 + rank) over the lists a "
            "chunk appears in. A chunk found by both paths collects from both, which "
            "is why the fused order is not either path's order."
        )
        st.markdown(retrieval_table(retrieved))
        st.caption(
            "Numbers are the passages as the model saw them, so row 1 is [1] in the "
            "answer above. A dash means that path did not return the chunk at all."
        )


if asked and not pending.strip():
    st.warning("Type a question first.")
elif asked and not key_present:
    # Retrieval still runs, for real, off the committed embedding cache. Nothing
    # here is replayed or recorded: the table below is this query, scored now.
    retriever = load_retriever(offline=True)
    try:
        retrieved = retriever.retrieve(pending, k=TOP_K)
    except EmbeddingCacheMiss:
        st.error(
            "That question is not in the committed embedding cache, so it cannot be "
            "retrieved without an API key. Use one of the examples above."
        )
    else:
        st.subheader("Answer")
        st.write(
            "Not generated: answering needs an API key. The retrieval below ran for "
            "real against the committed cache, and the Evaluation section reports what "
            "the answers scored when they were last measured."
        )
        render_retrieval(retrieved, expanded=True)
elif asked:
    retriever = load_retriever(offline=False)
    answerer = build_answerer(retriever)
    try:
        with st.spinner("Retrieving and answering ..."):
            answer = answerer.answer(pending, k=TOP_K)
    except AnswerError as error:
        st.error(str(error))
    else:
        st.subheader("Answer")
        st.markdown(highlight_citations(answer.text))

        if answer.refused:
            st.info("Refused: the documents do not contain this.")
        if answer.conflict:
            st.warning("The sources disagree, and both are reported above.")
        if not answer.is_grounded:
            st.error(
                f"The model cited {answer.invalid_citations}, which were not in its "
                "context. Those citations were removed from the answer above."
            )

        if answer.citations:
            st.subheader("Sources")
            for citation in answer.citations:
                st.write(f"[{citation.number}] {citation.chunk.citation_label}")

        render_retrieval(answer.retrieved, expanded=False)

st.divider()

with st.expander("Evaluation", expanded=False):
    results = load_results()
    if results is None:
        st.write(
            "No metrics have been generated yet. Produce them with a full evaluation "
            "run, which needs an API key:"
        )
        st.code("python -m src.main eval", language="bash")
        st.caption(
            "The retrieval half runs without a key: `python -m src.main eval "
            "--retrieval-only`, though it does not write this file because it has no "
            "answer metrics to write."
        )
    else:
        # Prefer the reranked numbers when the run had reranking on, so the
        # retrieval row and the answer rows below describe the same pipeline.
        reranked_on = results.get("reranker", {}).get("enabled")
        retrieval = results["retrieval"].get(
            "hybrid_reranked" if reranked_on else "hybrid", results["retrieval"].get("hybrid", {})
        )
        answers = results["answers"]
        golden = results["golden_set"]
        rows = [
            ("recall@" + str(results["top_k"]), retrieval.get("recall"), "hybrid retrieval, answerable questions"),
            ("MRR", retrieval.get("mrr"), "rank of the first correct document"),
            ("citation validity", answers.get("citation_validity"), "answers whose every citation was in context"),
            ("refusal accuracy", answers.get("refusal_accuracy"), "unanswerable questions correctly refused"),
            ("false refusal rate", answers.get("false_refusal_rate"), "answerable questions wrongly refused"),
        ]
        table = ["| metric | value | meaning |", "|---|---|---|"]
        for name, value, meaning in rows:
            table.append(f"| {name} | {value:.3f} | {meaning} |" if value is not None else f"| {name} | - | {meaning} |")
        st.markdown("\n".join(table))

        by_type = ", ".join(f"{count} {name}" for name, count in golden["by_type"].items())
        st.caption(
            f"Measured on {results['generated_on']} with {results['generation_model']} "
            f"and {results['embedding_model']}, over {golden['questions']} golden set "
            f"questions ({by_type})."
        )
        st.caption(
            f"Read those with their sample sizes: refusal accuracy rests on "
            f"{golden['questions'] - golden['answerable']} questions and the false "
            f"refusal rate on {golden['answerable']}. One question moves either number "
            "by several points. The failures behind them are written up in the README."
        )
