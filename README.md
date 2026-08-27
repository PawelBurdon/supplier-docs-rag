# supplier-docs-rag

Retrieval-augmented question answering over a set of supplier and delivery
documents, with citations that are verified in code, an explicit refusal when
the documents do not contain the answer, and a measured evaluation harness.

## Problem

Naive retrieval fails on business documents in three specific ways, and all
three are visible in the corpus in `documents/`.

**Exact strings matter and embeddings do not preserve them.** An embedding
encodes what a passage is about. Its whole purpose is that near-synonyms land
near each other, which is exactly wrong for `HCS-SLA-2024` against
`HCS-RWP-2025`, `FRM-ALU-52` against `FRM-CRB-52`, or clause 5.2 against clause
5.3. A question containing one of those is a lookup, not a topic search.

**Business documents repeat themselves.** Measured on this corpus, the two most
similar chunks from different documents have a cosine similarity of 0.933, and
they are boilerplate headers: "Document reference / Owner / Version". The next
pair, at 0.930, is section 2 of one supplier onboarding record against section 2
of another. Dense similarity says these are nearly the same passage. Any query
drifting towards that shape pulls both, and one of them is wrong.

**Documents disagree, and a system that hides the disagreement is worse than
one that says nothing.** Here the Delivery SLA sets the late-delivery penalty at
0.5 percent per day capped at 10 percent; the internal FAQ says 1 percent capped
at 15 percent. A system that confidently returns one number, with a citation,
has produced something more dangerous than an error: an error that looks
checked.

And the failure that outranks all of them: a language model handed five
plausible passages will write a plausible paragraph, whether or not the answer
is in them. Refusal has to be designed, prompted for, and measured — in both
directions, because a system that refuses everything scores perfectly on
refusal.

## How it works

```
  question
     |
     +--> embed as RETRIEVAL_QUERY ---> vector search (cosine)  --> top 20
     |                                                                |
     +--> tokenize ---------------> BM25 keyword search --> top 20    |
                                                             |        |
                                    reciprocal rank fusion <-+--------+
                                    score = sum 1/(60 + rank)
                                                |
                                             top 5
                                                |
                     context assembly: number each chunk, label it with
                     its document title, section heading and file name
                                                |
                            gemini-3.1-flash-lite, temperature 0,
                            JSON schema: answer, citations, refused, conflict
                                                |
                     validation: every cited number must exist in the context;
                     fabricated ones are stripped and recorded
                                                |
                    answer with citations, or an explicit refusal,
                    or both sides of a documented disagreement
```

No LangChain, no LlamaIndex. Retrieval, prompt assembly and citation parsing are
written out in `src/`.

## Retrieval design

**Chunking** (`src/core/chunker.py`). Target 1200 characters with 15 percent
sentence-aligned overlap, but a chunk never crosses a section boundary, so the
effective policy is one section per chunk capped at 1200. The corpus produces 63
chunks with a median of 458 characters; the cap actually binds in one place.

The boundary rule is a citation decision, not a retrieval one: a chunk labelled
"section 6" whose first sentences belong to section 5 produces a citation that
points at the wrong place, and a wrong citation is worse than a missing one.
What it costs is context density — eight chunks are under 200 characters, so
five retrieved chunks are sometimes only about 2000 characters of context.

Markdown tables are atomic and travel with the sentence that introduces them.
The lead-time table is the most queried object in the corpus, and rows without
their header row are numbers without meaning. The cost is one oversized chunk
whose embedding mixes every SKU category with every supplier tier, so a question
about one row may not pull it by vector similarity at all.

Every chunk is indexed as `Document title > Section heading` followed by its
body. A chunk reading "5.2 The penalty is capped at 10 percent of the order
value" contains neither "late delivery" nor "SLA", and no realistic question
reaches it unprefixed. The cost is that chunks from one document now share a
prefix and look more alike to the embedder.

**Embeddings** (`src/core/embedder.py`). `gemini-embedding-001`, 768 of its
native 3072 dimensions. Documents are embedded as `RETRIEVAL_DOCUMENT` and
questions as `RETRIEVAL_QUERY` — the model is trained asymmetrically, and the
task type is part of the cache key so the same text in two roles can never share
an entry.

One trap worth naming: a Matryoshka-truncated embedding is **not** unit length.
The 768-dimensional slice came back with a norm of about 0.59, varying by text.
Skipping normalisation would leave a dot product that is part similarity and
part vector magnitude, so longer chunks would quietly outrank better matches,
and nothing would raise. Vectors are normalised once, at the cache boundary, and
the stores reject anything that is not unit length.

The cache is one `.npy` file per chunk, keyed by a hash of model, task type and
text, and it is committed to the repository. That is a generated artefact in
version control, which is a real cost — it can drift from the documents. It buys
something worth more: anyone can clone this repository with no API key and
reproduce every retrieval number below, and CI does exactly that on every
commit. Drift is caught, not tolerated: a missing entry raises with the command
that fixes it.

**Storage** (`src/core/store.py`). Two implementations behind four methods —
`add`, `search`, `persist`, `load`. `NumpyStore` is exact brute force, one matrix
multiply. `ChromaStore` is the same interface over a real vector database. On
this corpus, measured:

| store       | build   | search per query |
|-------------|---------|------------------|
| NumpyStore  | 0.3 ms  | 0.023 ms         |
| ChromaStore | 1033 ms | 2.0 ms           |

Brute force is 87 times faster here and returns the true nearest neighbours by
definition. That is not a criticism of Chroma; it is what a 63-chunk corpus
means. The reason to have built against both is that the crossover is real:
brute force scans every vector, so cost grows linearly, and HNSW trades
exactness for a graph walk that touches a fraction of the index. A test asserts
the two agree on this corpus — at scale, that would be the wrong test to write,
because approximate means approximate.

**Hybrid retrieval** (`src/core/retriever.py`). Vector search and BM25, each
contributing 20 candidates, fused by reciprocal rank with the constant 60 from
the original RRF paper.

Fusion is on ranks rather than scores because the two scales cannot be
reconciled: BM25 is unbounded and corpus-dependent, cosine sits in a narrow band
around 0.70 to 0.80 here. Any weighted sum would be a weight pulled from thin
air that needs retuning per corpus.

Tokenisation emits each code whole and in parts: `FRM-CRB-52` becomes
`frm-crb-52`, `frm`, `crb`, `52`. Without the parts, a question about `FRM-CRB`
matches nothing, because the documents only ever write the size-specific code.
With them, BM25 documents get artificially longer, and junk tokens like `52` can
link unrelated passages. A chunk whose BM25 score is zero is dropped before
fusion — zero means no query term occurs in it at all, and letting it in would
hand out RRF credit in chunk-id order.

## Evaluation

`src/evals/golden_set.yaml` holds 18 questions: 7 factual, 4 multi_hop, 3
contradiction, 4 unanswerable. Each names the documents that should be
retrieved, the figures a correct answer must contain, and a type.

Definitions, each of which is a choice:

- **Recall is over documents, not chunk ids.** A chunk id changes whenever the
  chunker changes, which would make two chunking strategies incomparable. The
  cost is that retrieving the wrong section of the right document counts as a
  hit.
- **Questions needing two documents get two numbers**: recall with partial
  credit, and full coverage (both documents or nothing). For a contradiction
  question only full coverage means anything — a system holding one side of a
  disagreement cannot report a disagreement, however good its recall looks.
- **Unanswerable questions are excluded from recall and MRR.** There is no
  document that should be retrieved, so a retrieval score there would be
  meaningless rather than zero.
- **A refusal is the `refused` field and nothing else.** No phrase matching. If
  the model writes "the documents do not state this" but leaves the field false,
  that is a defect and the harness is built to show it.

### Retrieval, k=5, 14 answerable questions

| mode    | recall@5 | full coverage | MRR   |
|---------|----------|---------------|-------|
| hybrid  | 0.964    | 0.93          | 0.857 |
| vector  | 1.000    | 1.00          | 0.845 |
| keyword | 0.929    | 0.86          | 0.774 |

| type          | n | recall@5 | full coverage | MRR   |
|---------------|---|----------|---------------|-------|
| factual       | 7 | 1.000    | 1.00          | 0.786 |
| multi_hop     | 4 | 0.875    | 0.75          | 0.875 |
| contradiction | 3 | 1.000    | 1.00          | 1.000 |

These run in CI with no API key.

### Answers, gemini-3.1-flash-lite, 18 questions

| metric              | value | meaning                                              |
|---------------------|-------|------------------------------------------------------|
| citation validity   | 1.000 | every citation maps to a passage the model was given |
| refusal accuracy    | 1.000 | unanswerable questions correctly refused (n=4)       |
| false refusal rate  | 0.000 | answerable questions wrongly refused (n=14)          |
| conflict detection  | 1.000 | contradiction questions flagged (n=3)                |
| false conflict rate | 0.091 | other answerable questions flagged (1 of 11)         |
| fact coverage       | 1.000 | answers containing every expected figure             |

**Read those numbers with their sample sizes.** Refusal accuracy is four
questions: one failure would read as 0.750. None of these figures is precise to
three decimals, and the next section shows two of them are misleading at one
decimal.

## Where it fails

Three honest findings, all from the run above.

### 1. Hybrid retrieval is worse than dense-only on this corpus

Vector-only scores recall 1.000 and full coverage 1.00. Hybrid scores 0.964 and
0.93. The entire difference is one question, q08: *"What lead time applies to a
carbon frame set ordered from Velocore Components?"* — a multi-hop question
needing the Velocore record (which says Tier 2) and the SLA (whose table gives
Tier 2 carbon frames 35 days).

```
--- vector ---
 1. delivery_sla                    [3. Committed Lead Times  ]  <== needed
 2. onboarding_velocore_components  [1. SUPPLIER IDENTITY     ]  <== needed
 3. internal_faq_procurement        [Warranty                 ]
 4. supplier_framework_agreement    [2. Product Scope         ]
 5. onboarding_velocore_components  [3. APPROVED SKU FAMILIES ]  <== needed

--- hybrid ---
 1. delivery_sla                    [3. Committed Lead Times  ] vector#1+bm25#2  <== needed
 2. supplier_framework_agreement    [2. Product Scope         ] vector#4+bm25#4
 3. internal_faq_procurement        [Lead times               ] vector#10+bm25#5
 4. returns_and_warranty_policy     [2. Warranty Periods      ] vector#7+bm25#9
 5. supplier_framework_agreement    [5. Delivery Terms        ] vector#6+bm25#14
```

The Velocore chunks are gone. The arithmetic: Velocore scored vector#2 and did
not appear in the BM25 list at all, giving it 1/62 = 0.0161. The framework
agreement chunk scored vector#4 and bm25#4, giving it 1/64 + 1/64 = 0.0313.
Consensus between two mediocre rankings beat one confident hit. That is the
documented cost of the RRF constant 60, which flattens the difference between
the top positions on purpose.

Lowering the constant fixes it, measured offline over the whole golden set:

| RRF constant | recall@5 | full coverage | MRR   |
|--------------|----------|---------------|-------|
| 3            | 1.000    | 1.00          | 0.857 |
| 5            | 0.964    | 0.93          | 0.857 |
| 60           | 0.964    | 0.93          | 0.857 |

The constant stays at 60. The entire gain rests on one question out of
fourteen, and tuning a hyperparameter on a single data point is how a project
like this produces a number that does not survive contact with a second corpus.
The right fix is a reranker over the fused candidates, which is first on the
roadmap; the second-right fix is a larger golden set that could justify a
weighting.

### 2. The fact-coverage metric overstates answer quality, and q08 proves it

The answer to q08 was:

> The lead time for carbon frame sets (SKU prefix FRM-CRB) depends on the
> supplier's tier, which is 28 days for Tier 1, 35 days for Tier 2, and 60 days
> for Tier 3 [1].

It contains "35", so `fact coverage` counts it as correct. It is not correct.
The question asked about Velocore, the model never established that Velocore is
Tier 2, and what came back is the table restated. The retrieval failure in
finding 1 caused it, and the metric hid it. A substring check catches a wrong
figure; it cannot catch a right figure that answers a different question.

### 3. The false-conflict metric penalises defensible behaviour

`false conflict rate` is 0.091, which is one question: q09, about whether a
penalty applies to a late Pedalworks order. The model answered correctly and
also noted that the SLA and the FAQ disagree about the penalty rate — which, in
the passages it was given, they do. The golden set does not type q09 as a
contradiction question, so the harness scores it as a false positive. The metric
definition is too blunt here. It is reported unchanged rather than adjusted
after seeing the result, because a metric edited to flatter a run is not a
metric.

## Stack

- Python 3.12 or newer (CI runs 3.12; developed on 3.13)
- `google-genai` — `gemini-embedding-001` for embeddings, `gemini-3.1-flash-lite`
  for generation
- `numpy` — the exact vector store
- `chromadb` — the alternative vector store
- `rank-bm25` — BM25 Okapi
- `pyyaml`, `python-dotenv`, `pytest`

No retrieval framework. 81 tests, none of which touch the network.

## Running locally

```bash
pip install -r requirements.txt
```

Retrieval metrics need nothing else — the embedding cache is committed:

```bash
python -m src.main eval --retrieval-only
```

For answering, copy `.env.example` to `.env` and add a Google AI Studio key
(https://aistudio.google.com/apikey):

```bash
python -m src.main index
python -m src.main ask "What penalty do we charge a supplier for a late delivery?"
python -m src.main ask "..." --verbose
python -m src.main eval
python -m src.main eval --model gemini-3.5-flash
python -m src.main index --store chroma
```

`--verbose` prints the fused ranking with both per-path scores:

```
  [1] rrf=0.03279  cosine=0.804  bm25=21.77  (vector#1+bm25#1)
      Procurement Internal FAQ, Late deliveries  [internal_faq_procurement.md]
  [2] rrf=0.03200  cosine=0.747  bm25=11.79  (vector#2+bm25#3)
      Delivery Service Level Agreement, 5. Late Delivery Penalties  [delivery_sla.md]

There is a conflict regarding the late delivery penalty. The Procurement
Internal FAQ states the penalty is 1 percent of the order value per day late,
capped at 15 percent [1]. The Delivery Service Level Agreement states the
penalty is 0.5 percent of the order value per day late, capped at 10 percent [2].
```

If HTTPS calls fail with `CERTIFICATE_VERIFY_FAILED`, a TLS-inspecting proxy or
antivirus is signing certificates with a locally installed root that `certifi`
does not carry. `pip install truststore` and the CLI will use the operating
system trust store instead; it is an optional import, not a dependency.

## Roadmap

In priority order, with the reason for the order.

1. **Grow the golden set to 60 or more questions.** Every number in this README
   has error bars wider than the differences it is being used to discuss —
   refusal accuracy rests on four questions, and the hybrid-versus-dense result
   on one. Nothing below can be evaluated honestly until this is fixed, so it is
   first.
2. **Add a reranker over the fused candidates.** This addresses finding 1
   directly and at the right layer: a cross-encoder scoring 20 candidates
   against the question would rank the Velocore chunk on its content rather than
   on how many lists it appeared in. Larger expected effect than any constant.
3. **Expand context around retrieved chunks instead of merging at index time.**
   Retrieve the small, precise chunk, then include its neighbours in the prompt.
   Fixes the thin-context problem without coarsening the index granularity that
   makes citations and recall sharp.
4. **Revisit per-path weighting once (1) is done.** Weighted RRF or a smaller
   constant may well be right; it cannot be justified on the current sample, and
   doing it before (1) would be fitting noise.
5. **Judge answer quality with a model from a different provider.** Substring
   fact checks cannot see finding 2. An LLM judge can, but a model judging
   output from its own family has a documented self-preference bias, so this
   needs a second provider and belongs after the deterministic metrics are
   saturated.
6. **Local embeddings through Ollama for a zero-quota offline path.** Deliberately
   rejected for now: it means `torch`, hundreds of megabytes, to serve a corpus
   of 63 chunks. It becomes worth it only if API quota becomes the binding
   constraint.

## Licence

MIT. See `LICENSE`.
