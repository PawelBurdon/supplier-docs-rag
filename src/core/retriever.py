"""Hybrid retrieval: dense vectors, BM25 keywords, fused by reciprocal rank.

Why not dense-only. An embedding encodes what a passage is about. It does not
reliably encode which exact string it contains, because the whole point of the
representation is that near-synonyms land near each other. Business documents
are full of tokens where near is worthless: HCS-SLA-2024 and HCS-RWP-2025 are
two different contracts, FRM-ALU-52 and FRM-CRB-52 are two different products,
and clause 5.2 is not clause 5.3. A question containing one of those is a lookup,
not a topic search, and a lookup is what BM25 is for.

This corpus gives a measured example of the same failure from the other side.
The two nearest chunks from different documents have a cosine of 0.933, and they
are boilerplate headers -- "Document reference / Owner / Version". Dense
similarity says they are almost the same passage. Any query that drifts towards
that shape pulls both, and one of them is wrong.

Why reciprocal rank fusion rather than mixing the scores. Cosine similarity and
BM25 are not on the same scale and cannot be made comparable by normalisation:
BM25 is unbounded and corpus-dependent, cosine sits in [-1, 1] and, on this
corpus, is compressed into a narrow band near 0.7 to 0.8. Any weighted sum of
the two is a weight pulled from thin air that would need retuning on every
corpus. RRF throws the scores away and keeps only the ranks, so both paths speak
the same language: position.

The constant controls how much consensus is worth against confidence. The
original RRF paper uses 60, which flattens the difference between the top
positions -- 1/61 against 1/65 -- so a chunk both paths rank reasonably beats a
chunk one path ranks first. That paper fuses many rankers over large corpora.
With two rankers over 63 chunks it is too flat: measured on the golden set, it
cost recall against dense-only search. This project uses 3, chosen from a sweep
over the whole golden set rather than from the paper, and the README shows the
sweep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.chunker import Chunk
from src.core.embedder import Embedder
from src.core.store import VectorStore

DEFAULT_TOP_K = 5
CANDIDATE_POOL = 20
# 3, not the 60 from the original RRF paper. That constant was the default here
# until the golden set grew to 50 questions and made the difference measurable:
# across six values, recall rises monotonically as the constant falls, and only
# at 3 or below does hybrid retrieval beat dense-only search on this corpus
# (recall 1.000 against 0.987, full coverage 1.00 against 0.97). The published
# value is tuned for fusing many rankers over large corpora; with two rankers
# over 63 chunks it flattens the top positions so hard that a chunk one path
# ranks first loses to a chunk both paths rank fourth. The cost is MRR, which
# drops from 0.882 to 0.877 and stays below dense-only's 0.917 -- an acceptable
# trade here, because the generator reads all k retrieved chunks, so whether the
# right one is first or third inside that set changes nothing.
RRF_K = 3

RETRIEVAL_MODES = ("hybrid", "vector", "keyword")


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk that survived fusion, with the evidence for why it is here.

    The per-path ranks are not decoration. They are what --verbose prints and
    what makes a bad answer diagnosable: a chunk that appears at keyword rank 1
    and nowhere in the vector list failed for a different reason than one that
    both paths ranked eighth.
    """

    chunk: Chunk
    score: float
    vector_rank: int | None = None
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None

    @property
    def paths(self) -> str:
        found = []
        if self.vector_rank is not None:
            found.append(f"vector#{self.vector_rank}")
        if self.keyword_rank is not None:
            found.append(f"bm25#{self.keyword_rank}")
        return "+".join(found) or "none"


# A token is a run of letters and digits, optionally joined by the separators
# that hold codes and clause numbers together: FRM-CRB-52, HCS-SLA-2024, 5.2.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
_SEPARATORS = re.compile(r"[-./]")


def tokenize(text: str) -> list[str]:
    """Lowercase, then emit each compound token whole and in parts.

    "FRM-CRB-52" becomes ["frm-crb-52", "frm", "crb", "52"]. The whole form lets
    an exact code match score hard; the parts let a query for "FRM-CRB" reach a
    chunk that only ever writes the full size-specific code.

    The cost is paid in BM25's length normalisation: a chunk dense with codes
    reports as a longer document than it reads, which slightly suppresses its own
    scores, and junk tokens like "52" can link passages that have nothing to do
    with each other.
    """
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        if _SEPARATORS.search(match):
            tokens.extend(part for part in _SEPARATORS.split(match) if part)
    return tokens


class Retriever:
    """Searches one vector store and one BM25 index, and fuses the two rankings.

    Args:
        store: Any VectorStore. The retriever never learns which one it holds.
        embedder: Used only to embed the question.
        chunks: The same chunks the store holds, in any order. BM25 needs the
            text in memory; a vector store does not hand it back cheaply enough
            to rebuild the keyword index per query.
        top_k: How many chunks the caller gets back.
        pool: How many candidates each path contributes before fusion.
        rrf_k: The RRF constant.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunks: list[Chunk],
        top_k: int = DEFAULT_TOP_K,
        pool: int = CANDIDATE_POOL,
        rrf_k: int = RRF_K,
        reranker=None,
    ) -> None:
        from rank_bm25 import BM25Okapi

        self.store = store
        self.embedder = embedder
        self.chunks = list(chunks)
        self.top_k = top_k
        self.pool = pool
        self.rrf_k = rrf_k
        # Optional. When present, fusion produces the whole candidate pool and
        # the reranker chooses the k that reach the caller. It lives here rather
        # than in the answerer so that the CLI, the evaluation harness and the
        # demo all get the same pipeline without knowing it changed.
        self.reranker = reranker
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        # BM25 indexes index_text, the same string the embedder saw, so the two
        # paths are ranking identical content and a disagreement is a real
        # disagreement about relevance rather than about what was indexed.
        self._bm25 = BM25Okapi([tokenize(chunk.index_text) for chunk in self.chunks])

    def retrieve(
        self, question: str, k: int | None = None, mode: str = "hybrid"
    ) -> list[RetrievedChunk]:
        """Retrieve for a question.

        `mode` exists so the evaluation harness can measure each path on its own
        and report what hybrid actually bought. A claim that hybrid helps is
        worth nothing without the two single-path baselines beside it.
        """
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
        k = self.top_k if k is None else k
        if k <= 0 or not self.chunks:
            return []

        vector_hits = self._vector_candidates(question) if mode != "keyword" else []
        keyword_hits = self._keyword_candidates(question) if mode != "vector" else []
        if self.reranker is None:
            return self._fuse(vector_hits, keyword_hits, k)

        # With a reranker, fusion stops being the final word and becomes the
        # shortlist: it decides which candidates are considered, the reranker
        # decides which are shown.
        candidates = self._fuse(vector_hits, keyword_hits, self.pool)
        return self.reranker.rerank(question, candidates, top_k=k)

    def _vector_candidates(self, question: str) -> list[tuple[str, float]]:
        query_vector = self.embedder.embed_query(question)
        return [
            (hit.chunk.chunk_id, hit.score)
            for hit in self.store.search(query_vector, k=self.pool)
        ]

    def _keyword_candidates(self, question: str) -> list[tuple[str, float]]:
        tokens = tokenize(question)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id),
        )
        # A zero score means not one query term occurs in the chunk. Such a chunk
        # is not a weak match, it is not a match at all, and letting it into the
        # pool would hand RRF credit out in chunk_id order.
        return [
            (self.chunks[index].chunk_id, float(scores[index]))
            for index in ranked[: self.pool]
            if scores[index] > 0.0
        ]

    def _fuse(
        self,
        vector_hits: list[tuple[str, float]],
        keyword_hits: list[tuple[str, float]],
        k: int,
    ) -> list[RetrievedChunk]:
        fused: dict[str, float] = {}
        vector_rank: dict[str, int] = {}
        vector_score: dict[str, float] = {}
        keyword_rank: dict[str, int] = {}
        keyword_score: dict[str, float] = {}

        for rank, (chunk_id, score) in enumerate(vector_hits, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            vector_rank[chunk_id] = rank
            vector_score[chunk_id] = score
        for rank, (chunk_id, score) in enumerate(keyword_hits, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            keyword_rank[chunk_id] = rank
            keyword_score[chunk_id] = score

        # Ties are broken by chunk_id, not by dictionary order, so that two runs
        # of the same evaluation are comparable.
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [
            RetrievedChunk(
                chunk=self._by_id[chunk_id],
                score=score,
                vector_rank=vector_rank.get(chunk_id),
                vector_score=vector_score.get(chunk_id),
                keyword_rank=keyword_rank.get(chunk_id),
                keyword_score=keyword_score.get(chunk_id),
            )
            for chunk_id, score in ordered[:k]
        ]
