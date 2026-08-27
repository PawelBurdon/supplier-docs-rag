"""The prompt. All of it, in one place, so it can be read and argued with.

Three instructions here are not stylistic, they are the product:

Refusal. The model is told, in the imperative, that a question the context does
not answer must be refused. Without it, a language model given five plausible
passages will write a plausible paragraph, because that is what it is for. The
refusal is also a structured field, so the evaluation can count it instead of
guessing at phrasing.

Disagreement. When two sources conflict the model must report both. This corpus
contains a deliberate conflict -- the SLA says the late-delivery penalty is 0.5
percent per day capped at 10 percent, the internal FAQ says 1 percent capped at
15 percent -- and a system that silently picks one is worse than useless on
business documents, because the reader cannot tell that a choice was made at
all.

Precedence without invention. The model may state which source wins only when a
document says so. The SLA and the framework agreement contain explicit
precedence clauses; the model may repeat them. It may not infer that a contract
outranks a wiki page because that feels right.
"""

from __future__ import annotations

from src.core.retriever import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about a bicycle-parts distributor's supplier and delivery documents.

You are given numbered context passages. Follow these rules exactly.

1. Use only the numbered context. You have no other knowledge of this company, \
its suppliers, its contracts or its products. If you know something from \
elsewhere, it is not relevant here.

2. Cite inline. Every factual statement must be followed by the number of the \
passage it comes from, written as [1] or [2][3]. Cite only numbers that appear \
in the context you were given.

3. Refuse when the context does not answer the question. Set "refused" to true, \
and in "answer" state plainly what is missing, for example: "The provided \
documents do not state the freight insurance limit." Do not guess, do not \
reason from general commercial practice, and do not answer a nearby question \
instead of the one asked. A passage that is on the same topic is not an answer.

4. Surface disagreement. If two passages give different answers to the same \
question, set "conflict" to true and report both, each with its own citation, \
saying which document each figure comes from. Never silently choose one. You \
may state that one source takes precedence over another only if a passage says \
so; if no passage states a precedence rule, present the conflict without \
resolving it.

5. Quote figures, dates, percentages and code numbers exactly as they are \
written in the context. Do not convert, round, or recalculate them.

6. Be brief. Two or three sentences unless the question needs more."""


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer, with inline citations such as [1]. On refusal, "
            "a plain statement of what the documents do not contain.",
        },
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Every context number cited in the answer.",
        },
        "refused": {
            "type": "boolean",
            "description": "True when the context does not contain the answer.",
        },
        "conflict": {
            "type": "boolean",
            "description": "True when two context passages disagree about the answer.",
        },
    },
    "required": ["answer", "citations", "refused", "conflict"],
}


def build_context_block(retrieved: list[RetrievedChunk]) -> str:
    """Number the retrieved chunks and label each with the source it came from.

    The label is what makes a citation checkable by a human: "Delivery Service
    Level Agreement, 5. Late Delivery Penalties (delivery_sla.md)" can be opened
    and read. A bare filename, or a chunk id, cannot.
    """
    blocks = []
    for number, hit in enumerate(retrieved, start=1):
        blocks.append(
            f"[{number}] {hit.chunk.citation_label} (source: {hit.chunk.source_path})\n"
            f"{hit.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Assemble the context and the question into the user turn."""
    if not retrieved:
        # An empty index or a query with no candidates still goes to the model,
        # which then has nothing to answer from and must refuse. Keeping this
        # path in the model's hands rather than short-circuiting it in code means
        # the refusal metric measures one mechanism, not two.
        context = "(no passages were retrieved)"
    else:
        context = build_context_block(retrieved)
    return f"""Context passages:

{context}

Question: {question}"""
