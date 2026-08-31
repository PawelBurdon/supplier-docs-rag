"""Rerank fused candidates with a language model, with an on-disk cache.

Why a reranker at all. Fusion scores a chunk by the positions it occupies in two
ranked lists. It never reads the chunk against the question. On this corpus that
costs five questions: the passage that answers them sits at fused rank 8 to 17 of
twenty candidates, retrieved but never shown to the model. A reranker scores each
candidate on whether it answers the question, which is the judgement fusion
cannot make.

Why a language model rather than a cross-encoder. A purpose-built cross-encoder
-- Cohere Rerank, or a local model through sentence-transformers -- would be
faster, cheaper per query and more predictable than asking a generative model to
sort passages. Both were rejected for the same reason: one needs a second
provider and a second API key from everybody who clones this repository, the
other needs torch and several hundred megabytes to serve 63 chunks. This project
is meant to be reproducible by a stranger with one key, and that constraint won.
The cost is real and it is stated in the README: an LLM reranker is slower per
query and its ordering can shift in ways a trained scorer's would not.

The cache is what keeps that honest. An ordering is stored under a hash of the
model, the question, and the candidate ids in the order they were proposed --
all three, because the same question over a different candidate list is a
different judgement, and a model change that silently reused another model's
orderings would be worse than no cache at all. Cached orderings are committed
next to the embeddings, so retrieval metrics still run in CI with no key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from src.core.retriever import RetrievedChunk

DEFAULT_RERANK_MODEL = "gemini-3.1-flash-lite"
DEFAULT_CACHE_DIR = Path("rerank_cache")
MAX_ATTEMPTS = 4

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Candidate numbers, best first, most relevant to the question.",
        }
    },
    "required": ["ranking"],
}

_BASE_PROMPT = """You order retrieved passages by how well each one answers a question.

You are given a question and numbered passages from a company's supplier
documents. Return the passage numbers ordered from most to least useful for
answering that question.

Judge only whether a passage contains the information the question asks for. A
passage on the same topic that does not carry the answer ranks below one that
does. A passage stating that something is not covered, or setting a figure the
question asks about, is useful. Do not judge writing quality, length or
formality.

Return every number you were given, each exactly once, in the new order."""

# The base prompt cost q40: the question named clause 5.1 of the SLA, so the
# reranker demoted the FAQ passage that contradicts it, and a question about a
# disagreement reached the model holding one side. Relevance to the question is
# not the same thing as the set of passages needed for a grounded answer, and
# the reranker had never been told the difference.
_CONFLICT_RULE = """

One exception to relevance. When the question asks for a figure, a deadline or a
rule, a passage that states a DIFFERENT value for the same thing is useful
precisely because it disagrees, and must rank high even when the question names
one document by title or clause and that passage comes from another. Answers
here have to surface disagreements rather than resolve them silently, so both
sides have to survive this ordering."""

PROMPTS = {
    "base": _BASE_PROMPT,
    "conflict-aware": _BASE_PROMPT + _CONFLICT_RULE,
}
DEFAULT_PROMPT = "conflict-aware"


class RerankCacheMiss(RuntimeError):
    """Raised when an ordering is not cached and no network call is possible."""


class Reranker:
    """Reorders candidates with a model, reading and writing an on-disk cache."""

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        api_key: str | None = None,
        offline: bool = False,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        if prompt not in PROMPTS:
            raise ValueError(f"Unknown reranker prompt {prompt!r}; expected one of {list(PROMPTS)}")
        self.prompt_name = prompt
        self.prompt = PROMPTS[prompt]
        self.model = model
        self.cache_dir = Path(cache_dir) / model.replace("/", "_")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or ""
        self.offline = offline or not self.api_key
        self._client = None
        self.cache_hits = 0
        self.api_calls = 0

    def rerank(
        self, question: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the best `top_k` candidates, reordered.

        The returned objects are the ones passed in, so their per-path ranks and
        scores survive: the CLI and the demo still show how each chunk was found,
        now next to the order a model put them in.
        """
        if not candidates or top_k <= 0:
            return []
        if len(candidates) == 1:
            return candidates[:top_k]

        order = self._read_cache(question, candidates)
        if order is None:
            if self.offline:
                raise RerankCacheMiss(
                    f"No cached reranking for this question at {self.cache_dir}, and no "
                    "API key is available. The committed cache is stale with respect to "
                    "the documents or the golden set. Rebuild it with a key:\n"
                    "    python -m src.main index --refresh\n"
                    "then commit the new cache files."
                )
            order = self._call_api(question, candidates)
            self._write_cache(question, candidates, order)

        return [candidates[position] for position in order[:top_k]]

    def _cache_path(self, question: str, candidates: list[RetrievedChunk]) -> Path:
        # The candidate ids are part of the key in the order they arrived: the
        # same question over a different, or differently ordered, candidate list
        # is a different judgement and must not reuse this answer.
        # The prompt is in the key, hashed, for the same reason the model is: an
        # ordering produced under different instructions is a different
        # judgement, and serving it from cache after the instructions changed
        # would make the change invisible.
        prompt_digest = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:12]
        identity = "|".join(
            [
                self.model,
                prompt_digest,
                question,
                ",".join(candidate.chunk.chunk_id for candidate in candidates),
            ]
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, question: str, candidates: list[RetrievedChunk]) -> list[int] | None:
        path = self._cache_path(question, candidates)
        if not path.is_file():
            return None
        try:
            order = json.loads(path.read_text(encoding="utf-8"))["order"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        if sorted(order) != list(range(len(candidates))):
            # A cache file that does not describe this candidate list is treated
            # as absent rather than trusted.
            return None
        self.cache_hits += 1
        return order

    def _write_cache(
        self, question: str, candidates: list[RetrievedChunk], order: list[int]
    ) -> None:
        path = self._cache_path(question, candidates)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "prompt": self.prompt_name,
            "question": question,
            "candidates": [candidate.chunk.chunk_id for candidate in candidates],
            "order": order,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _call_api(self, question: str, candidates: list[RetrievedChunk]) -> list[int]:
        client = self._ensure_client()
        from google.genai import types

        blocks = []
        for number, candidate in enumerate(candidates, start=1):
            chunk = candidate.chunk
            blocks.append(f"[{number}] {chunk.citation_label}\n{chunk.text}")
        prompt = "Question: {}\n\nPassages:\n\n{}".format(question, "\n\n".join(blocks))

        response = self._with_retries(
            lambda: client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.prompt,
                    response_mime_type="application/json",
                    response_schema=RERANK_SCHEMA,
                    temperature=0.0,
                ),
            )
        )
        self.api_calls += 1
        return _parse_ranking(response.text or "", len(candidates))

    def _with_retries(self, call):
        from src.rag.answerer import _is_retryable, _requested_retry_delay

        delay = 1.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return call()
            except Exception as error:
                if not _is_retryable(error) or attempt == MAX_ATTEMPTS:
                    raise RuntimeError(f"Reranking failed: {error}") from error
                time.sleep(max(delay, _requested_retry_delay(str(error))))
                delay *= 2
        raise AssertionError("unreachable")

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            if not self.api_key:
                raise RerankCacheMiss("No GOOGLE_API_KEY set and no cached reranking available.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client


def _parse_ranking(raw: str, count: int) -> list[int]:
    """Turn the model's 1-based ranking into 0-based positions, repaired if needed.

    A reranker that drops a candidate would silently shorten the context, and one
    that repeats a candidate would show the model the same passage twice. Both are
    corrected here: unknown and duplicate numbers are discarded, and anything the
    model failed to mention is appended in its original order, so the result is
    always a permutation of the input.
    """
    try:
        ranking = json.loads(raw).get("ranking", [])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Reranker returned text that is not JSON: {raw[:200]}") from error

    order: list[int] = []
    for value in ranking:
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        position = value - 1
        if 0 <= position < count and position not in order:
            order.append(position)
    order.extend(position for position in range(count) if position not in order)
    return order
