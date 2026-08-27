"""Assemble context, call the model, and verify what comes back.

The verification is the part worth reading. A retrieval system that prints
citations is not grounded; it is decorated. Grounding means the code checks that
every citation the model produced points at a passage the model was actually
given, and reports it when one does not.

The policy on a bad citation is: drop the marker, keep the answer, and record the
failure. That is a deliberate middle position. Refusing the whole answer would
be safer but would fold two very different failures -- "the model did not know"
and "the model mistyped a number" -- into one metric. Retrying with a stricter
prompt would usually produce a clean answer and would hide exactly the number
worth knowing: how often the model invents sources. So the answer is returned
with its unsupported marker removed and `invalid_citations` populated, and the
CLI prints a warning. Note the residue this leaves: a sentence whose source has
been stripped is still in the answer, unsupported. That is the cost of this
choice, and it is why the evaluation reports citation validity per question
rather than as an average that can hide a single bad answer.

The model is never asked to decide whether retrieval was good enough. There is
no score threshold in front of it. An RRF score carries no absolute meaning --
it is a sum of reciprocal ranks, so the best chunk scores about the same whether
it is excellent or hopeless -- and a threshold on raw cosine would be a guess,
on this corpus a guess inside the narrow 0.70 to 0.80 band where everything
lives. Refusal is the model's judgement about content, and the evaluation
measures how good that judgement is, in both directions.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from src.core.chunker import Chunk
from src.core.retriever import RetrievedChunk, Retriever
from src.rag.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, build_user_prompt

DEFAULT_GENERATION_MODEL = "gemini-3.1-flash-lite"
MAX_ATTEMPTS = 4

_CITATION = re.compile(r"\[(\d+)\]")
# Google returns the wait it wants either as "retryDelay': '14s'" or as
# "Please retry in 14.34s" depending on the error surface. Both are matched.
_RETRY_DELAY = re.compile(r"retry(?:Delay'?:?\s*'?| in )\s*([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
MAX_RETRY_WAIT = 90.0


class AnswerError(RuntimeError):
    """The model call failed, or returned something that is not a usable answer."""


@dataclass(frozen=True)
class Citation:
    """A citation number in the answer, resolved to the chunk it refers to."""

    number: int
    chunk: Chunk

    @property
    def label(self) -> str:
        return f"[{self.number}] {self.chunk.citation_label} ({self.chunk.source_path})"


@dataclass(frozen=True)
class Answer:
    """One answered question, with everything needed to audit it."""

    question: str
    text: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    refused: bool
    conflict: bool
    model: str
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True when every citation the model produced maps to a retrieved passage."""
        return not self.invalid_citations

    @property
    def cited_documents(self) -> list[str]:
        """Document ids actually cited, in order of first citation."""
        seen: list[str] = []
        for citation in self.citations:
            if citation.chunk.doc_id not in seen:
                seen.append(citation.chunk.doc_id)
        return seen


class Answerer:
    """Retrieves, prompts, and validates. One question in, one audited answer out."""

    def __init__(
        self,
        retriever: Retriever,
        model: str = DEFAULT_GENERATION_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.api_key = api_key
        self._client = None

    def answer(self, question: str, k: int | None = None) -> Answer:
        if not question.strip():
            raise ValueError("Question is empty")
        retrieved = self.retriever.retrieve(question, k=k)
        payload = self._generate(build_user_prompt(question, retrieved))
        return self._validate(question, payload, retrieved)

    def _validate(self, question: str, payload: dict, retrieved: list[RetrievedChunk]) -> Answer:
        text = str(payload.get("answer", "")).strip()
        declared = {int(number) for number in payload.get("citations", []) if _is_int(number)}
        inline = {int(match) for match in _CITATION.findall(text)}
        allowed = set(range(1, len(retrieved) + 1))

        # Both sources of citation are checked. A number written inline is what a
        # reader sees; a number in the citations field is what the model claims.
        # Either one pointing outside the context is a fabricated source.
        invalid = sorted((inline | declared) - allowed)
        if invalid:
            text = _strip_citations(text, invalid)

        ordered_inline = [
            int(match) for match in _CITATION.findall(text) if int(match) in allowed
        ]
        numbers: list[int] = []
        for number in ordered_inline + sorted(declared & allowed):
            if number not in numbers:
                numbers.append(number)

        return Answer(
            question=question,
            text=text,
            citations=[Citation(number, retrieved[number - 1].chunk) for number in numbers],
            retrieved=retrieved,
            refused=bool(payload.get("refused", False)),
            conflict=bool(payload.get("conflict", False)),
            model=self.model,
            invalid_citations=invalid,
        )

    def _generate(self, user_prompt: str) -> dict:
        client = self._ensure_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ANSWER_SCHEMA,
            # Zero temperature because the evaluation compares runs. A sampled
            # answer that differs between runs turns every metric into noise
            # around an unknown mean.
            temperature=0.0,
        )
        response = self._with_retries(
            lambda: client.models.generate_content(
                model=self.model, contents=user_prompt, config=config
            )
        )
        raw = (response.text or "").strip()
        if not raw:
            raise AnswerError(
                f"{self.model} returned an empty response. This is usually a safety "
                "block or a truncated generation."
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AnswerError(
                f"{self.model} returned text that is not JSON despite a response schema: "
                f"{raw[:300]}"
            ) from error
        if not isinstance(payload, dict):
            raise AnswerError(f"Expected a JSON object, got {type(payload).__name__}")
        return payload

    def _with_retries(self, call):
        """Retry, honouring the server's own retry delay when it sends one.

        Rate limiting is not a transient network blip and must not be treated
        like one. The free tier allows five requests per minute for some models,
        and a 429 arrives carrying `retryDelay: 14s`. Backing off 1 second and
        then 2 guarantees three failures and an aborted evaluation run, which is
        how the first attempt at a cross-model comparison here died.
        """
        delay = 1.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return call()
            except Exception as error:
                if not _is_retryable(error):
                    raise AnswerError(f"Generation failed: {error}")
                if attempt == MAX_ATTEMPTS:
                    raise AnswerError(f"Generation failed after {MAX_ATTEMPTS} attempts: {error}")
                requested = _requested_retry_delay(str(error))
                time.sleep(max(delay, requested))
                delay *= 2
        raise AssertionError("unreachable")

    def _ensure_client(self):
        if self._client is None:
            import os

            from google import genai

            key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise AnswerError(
                    "No GOOGLE_API_KEY set. Answering needs a key; retrieval metrics do "
                    "not. Run: python -m src.main eval --retrieval-only"
                )
            self._client = genai.Client(api_key=key)
        return self._client


def _is_retryable(error: Exception) -> bool:
    """Rate limits and server faults are worth retrying; a bad request is not.

    Retrying a 404 four times is not resilience, it is four wasted requests
    against a quota and a slower path to the same failure. Retired model ids
    return 404 with the name of their replacement in the message, and that
    message should reach the user immediately.
    """
    text = str(error)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return True
    return not any(code in text for code in ("400 ", "401 ", "403 ", "404 ", "INVALID_ARGUMENT"))


def _requested_retry_delay(message: str) -> float:
    """Seconds the API asked us to wait, or 0 when it did not say.

    Capped: a server asking for ten minutes means the run should fail with a
    clear error, not hang looking like progress.
    """
    match = _RETRY_DELAY.search(message)
    if not match:
        return 0.0
    return min(float(match.group(1)) + 0.5, MAX_RETRY_WAIT)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strip_citations(text: str, numbers: list[int]) -> str:
    """Remove markers for citations that were never in the context."""
    for number in numbers:
        text = text.replace(f"[{number}]", "")
    return re.sub(r"[ \t]{2,}", " ", text).replace(" .", ".").replace(" ,", ",").strip()
