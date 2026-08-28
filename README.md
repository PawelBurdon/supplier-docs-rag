# supplier-docs-rag

Retrieval-augmented question answering over a set of supplier and delivery
documents, with citations that are verified in code, an explicit refusal when
the documents do not contain the answer, and a measured evaluation harness.

## Quick look

Hybrid retrieval (dense + BM25) over seven supplier documents, answers grounded
in numbered citations that the code verifies, and a 50-question golden set that
measures both the retrieval and the refusals. No RAG framework.

Reproduce the retrieval metrics, no API key needed -- the embedding cache is
committed:

```bash
pip install -r requirements.txt && python -m src.main eval --retrieval-only
```

| metric             | value | measured over                        |
|--------------------|-------|--------------------------------------|
| recall@5 (hybrid)  | 1.000 | 38 answerable questions              |
| citation validity  | 1.000 | 50 answers, gemini-3.1-flash-lite    |
| refusal accuracy   | 1.000 | 12 unanswerable questions            |
| false refusal rate | 0.079 | 3 of 38 answerable questions refused |

A Streamlit demo (`streamlit run app.py`) runs the same pipeline in a browser and
shows, per question, which chunks each retrieval path found and how the fusion
ordered them.

![The demo answering a question whose two sources disagree, then opening the
retrieval panel to show each chunk's cosine score, BM25 rank, fused score and
which path found it](docs/demo.gif)

Small samples, one invented corpus, and the three false refusals are dissected
rather than rounded away: the methodology is in Evaluation, and what still
breaks -- including a case where recall@5 scored 1.000 on a question the system
failed -- is in Where it fails.

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
                                    score = sum 1/(3 + rank)
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
contributing 20 candidates, fused by reciprocal rank with the constant 3.

The constant is not the 60 from the original RRF paper, and the reason is the
only kind that counts here: it was measured. The constant decides how much
agreement between the two rankers is worth against confidence from one of them.
At 60 the top positions are nearly indistinguishable -- 1/61 against 1/65 -- so
a chunk both paths rank fourth beats a chunk one path ranks first. That is right
when fusing many rankers over a large corpus, which is what the paper does. With
two rankers over 63 chunks it is too flat, and it cost recall. The sweep is in
the evaluation section.

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

`src/evals/golden_set.yaml` holds 50 questions: 20 factual, 12 multi_hop, 6
contradiction, 12 unanswerable. Each names the documents that should be
retrieved, the figures a correct answer must contain, a type, and a phrasing.

Definitions, each of which is a choice:

- **Recall is over documents, not chunk ids.** A chunk id changes whenever the
  chunker changes, which would make two chunking strategies incomparable. The
  cost is that retrieving the wrong section of the right document counts as a
  hit.
- **Questions needing two documents get two numbers**: recall with partial
  credit, and full coverage (both documents or nothing). For a contradiction
  question only full coverage means anything -- a system holding one side of a
  disagreement cannot report a disagreement, however good its recall looks.
- **Unanswerable questions are excluded from recall and MRR.** There is no
  document that should be retrieved, so a retrieval score there would be
  meaningless rather than zero.
- **A refusal is the `refused` field and nothing else.** No phrase matching. If
  the model writes "the documents do not state this" but leaves the field false,
  that is a defect and the harness is built to show it.
- **Phrasing is a measured axis, not a label.** 11 of the answerable questions
  are written the way somebody holding the paperwork would ask -- naming
  HCS-SLA-2024, clause 5.1, FRM-CRB -- and 27 in plain business language. The
  case for carrying a keyword index next to an embedding model rests entirely on
  those two styles failing differently, and that belongs in a table rather than
  in a paragraph.

### Retrieval, k=5, 38 answerable questions

| mode    | recall@5 | full coverage | MRR   |
|---------|----------|---------------|-------|
| hybrid  | 1.000    | 1.00          | 0.877 |
| vector  | 0.987    | 0.97          | 0.917 |
| keyword | 0.908    | 0.84          | 0.811 |

| type          | n  | recall@5 | full coverage | MRR   |
|---------------|----|----------|---------------|-------|
| factual       | 20 | 1.000    | 1.00          | 0.842 |
| multi_hop     | 12 | 1.000    | 1.00          | 0.875 |
| contradiction | 6  | 1.000    | 1.00          | 1.000 |

By phrasing, recall@5 and MRR:

| style | n  | hybrid        | vector        | keyword       |
|-------|----|---------------|---------------|---------------|
| prose | 27 | 1.000 / 0.864 | 0.981 / 0.944 | 0.870 / 0.809 |
| code  | 11 | 1.000 / 0.909 | 1.000 / 0.848 | 1.000 / 0.818 |

That last table is the honest version of "hybrid retrieval is for exact terms".
On code-phrased questions dense search finds the right documents too -- recall
1.000 at k=5 -- so BM25 is not rescuing them. What BM25 does is rank them
higher: MRR 0.909 against 0.848, the only cell where hybrid beats dense-only on
ranking. The blunter claim, that vector search cannot find a clause number, is
not what this corpus shows, and it is stated here rather than repeated.

Dense-only still has the better MRR overall (0.917 against 0.877). Hybrid was
chosen anyway, because the generator reads all five retrieved chunks: whether
the right one sits first or third inside that set changes nothing, while whether
it is in the set at all changes everything. Recall and full coverage are the
metrics that map onto the product, and hybrid wins both.

### The RRF constant, chosen by measurement

| RRF constant | recall@5 | full coverage | MRR   |
|--------------|----------|---------------|-------|
| 1            | 1.000    | 1.00          | 0.877 |
| 3 (used)     | 1.000    | 1.00          | 0.877 |
| 5            | 0.987    | 0.97          | 0.882 |
| 10           | 0.987    | 0.97          | 0.882 |
| 20           | 0.974    | 0.95          | 0.882 |
| 60 (paper)   | 0.961    | 0.92          | 0.882 |

These metrics run in CI with no API key.

### Answers, gemini-3.1-flash-lite, 50 questions

| metric              | value | meaning                                              |
|---------------------|-------|------------------------------------------------------|
| citation validity   | 1.000 | every citation maps to a passage the model was given |
| refusal accuracy    | 1.000 | unanswerable questions correctly refused (n=12)      |
| false refusal rate  | 0.079 | answerable questions wrongly refused (3 of 38)       |
| conflict detection  | 1.000 | contradiction questions flagged (n=6)                |
| false conflict rate | 0.031 | other answerable questions flagged (1 of 32)         |
| fact coverage       | 0.921 | answers containing every expected figure (35 of 38)  |

Not one fabricated citation in 50 answers, and not one unanswerable question
answered — including the four traps built to be answerable-looking: a minimum
order value that exists for the other supplier, an OTIF figure for the quarter
after the record ends, a customs code the FAQ explicitly refuses, and a
termination threshold whose plausible wrong answer (1.5 percent) is sitting in
the corpus attached to a different consequence.

The three false refusals are the interesting part, and they are dissected below.

### Cross-model comparison: attempted, not obtained

`--model` exists so the same golden set can be run against a different
generator, which turns "why this model" from an opinion into a measurement. It
is not answered here, and the reason is worth recording rather than hiding.

Both attempts hit exhausted free-tier daily quotas, not per-minute limits that
patience could absorb. Quotas and availability as observed on 27 August 2026, on
one free-tier key; all three are the provider's to change without notice:

| model                 | free-tier daily cap | outcome                                                    |
|-----------------------|---------------------|------------------------------------------------------------|
| gemini-3.5-flash      | 20 requests         | fewer than the golden set needs, before it grew to 50       |
| gemini-3.5-flash-lite | 500 requests        | already exhausted on the key in use                          |
| gemini-2.5-flash-lite | n/a                 | 404, retired for new accounts, API names its successor       |

A later probe found `gemini-3.6-flash` and `gemini-3-flash-preview` answering
normally, so the missing column is a quota window away rather than a code
change: `python -m src.main eval --model gemini-3.6-flash` produces it. Until
then, every answer metric above comes from `gemini-3.1-flash-lite` alone, and
the choice of generator rests on cost and availability rather than on a measured
comparison.

Two things did come out of trying. The first attempt died because the retry
logic treated a 429 carrying `retryDelay: 14s` as a transient blip and backed
off one second; the second wasted four requests retrying a 404 that named its
own fix in the error message. Both are fixed, and both were visible only because
the harness runs the whole set instead of one sample question.

## Where it fails

Three honest findings, all from the run above.

### 1. The first version of this project measured hybrid retrieval as a loss

This section previously reported that hybrid retrieval was worse than dense-only
search, and it was: on the original 18-question golden set, hybrid scored recall
0.964 and full coverage 0.93 against dense-only's 1.000 and 1.00. The failing
question was q08, "what lead time applies to a carbon frame set ordered from
Velocore Components". Dense search put both needed documents in the top five.
Hybrid dropped the Velocore record entirely, because it scored vector#2 and
appeared nowhere in the BM25 list -- 1/62 = 0.0161 -- while a framework agreement
chunk ranked fourth by both paths scored 1/64 + 1/64 = 0.0313. Consensus between
two mediocre rankings beat one confident hit.

A sweep at the time showed that lowering the RRF constant from 60 to 3 fixed it
and produced a perfect score. The constant was left at 60 anyway, because the
entire difference rested on one question out of fourteen, and a hyperparameter
tuned on one data point is a number that does not survive contact with a second
corpus.

Growing the golden set to 50 questions settled it. Across six values of the
constant, recall rises monotonically as it falls, over 38 answerable questions
rather than 14; at 3 or below, hybrid beats dense-only outright. The constant is
now 3, and q08 passes.

Two things are worth taking from this rather than from the final number. First,
the reason the earlier decision was right is not that it produced the better
score -- it did not -- but that the evidence did not support the change yet.
Second, a golden set too small to distinguish signal from noise makes every
downstream decision unfalsifiable, which is why enlarging it was the first item
on the roadmap and why it stayed first until it was done.

What has not been fixed: dense-only still ranks better, MRR 0.917 against 0.877.
Hybrid is kept because recall is the metric that matters when the generator
reads all five chunks, but the ranking cost is real and is not disguised
anywhere in this README.

### 2. Document-level recall reports a perfect score on a question the system failed

q22 asks "how quickly must a supplier acknowledge a purchase order". The Delivery
SLA answers it in one sentence: acknowledgement is due within 2 business days of
order receipt. The system refused.

```
retrieved for q22:
  internal_faq_procurement      [Lead times]
  internal_faq_procurement      [Late deliveries]
  supplier_framework_agreement  [1. Parties and Scope]
  delivery_sla                  [6. Escalation]
  internal_faq_procurement      [Payments]

the answer lives in:
  delivery_sla::003             [3. Committed Lead Times]

answer: "The provided documents do not state how quickly a supplier must
         acknowledge a purchase order."

retrieval score for q22: recall@5 = 1.000, MRR = 1.000
```

The retrieval metric is perfect because `delivery_sla` is in the result set. The
wrong section of it is. Recall is computed over documents, deliberately, so that
the golden set survives a change to the chunker -- and this is that decision
being charged for, on a real question, in the run reported above.

The refusal itself is correct behaviour: the model was not given the sentence
and did not invent it. The failure is upstream, and the metric that should have
caught it is looking at the wrong granularity.

Two changes would fix it, and they are on the roadmap in this order: a reranker
that scores candidates against the question rather than against how many lists
they appear in, and an anchor phrase in the golden set so recall can be scored
at section level without pinning the set to chunk ids that move whenever the
chunker changes.

The same fault produces the third false refusal, q39: the SLA was retrieved, but
section 1, Purpose, instead of section 7, Force Majeure, which is where the
answer is.

### 3. One of the three false refusals is arguably the harness being wrong

q36 asks which supplier is approved to deliver WHL-CRB carbon wheelsets. The
answer is neither. The system said:

> The provided documents do not state that any supplier is approved to deliver
> WHL-CRB carbon wheelsets. Pedalworks Manufacturing Ltd is explicitly restricted
> from providing carbon wheelsets [1].

That is correct, useful, and cited. It is scored as a false refusal because the
golden set types the question as answerable and the model set `refused` to true.
The same disagreement shows up in the false conflict rate: q09 is scored as a
false positive for flagging that the SLA and the FAQ disagree about the penalty
rate, which, in the passages it was given, they do.

Both are metric-definition problems rather than system problems, and both are
left as they are. Adjusting a definition after seeing which questions it marks
wrong is how an evaluation stops being one. The honest reading of the headline
numbers is therefore: of three false refusals, two are a real retrieval failure
with a single root cause, and one is a scoring argument.

### 4. Everything here rests on 63 chunks of invented text

Seven documents, one corpus, one language, one domain, all written by the same
author as the system that reads them. The contradictions are planted, so the
conflict detection score measures whether the system finds conflicts that were
put there to be found. A real corpus brings scanned PDFs, inconsistent headings,
tables that span pages, near-duplicate revisions of the same contract, and
disagreements that nobody designed. None of the numbers above should be read as
predicting behaviour on that.

## Stack

- Python 3.12 or newer (CI runs 3.12; developed on 3.13)
- `google-genai` — `gemini-embedding-001` for embeddings, `gemini-3.1-flash-lite`
  for generation
- `numpy` — the exact vector store
- `chromadb` — the alternative vector store
- `rank-bm25` — BM25 Okapi
- `pyyaml`, `python-dotenv`, `pytest`

Both model names were live and verified against the API on 27 August 2026. CI
runs without an API key, so it cannot notice a model being renamed or retired --
`gemini-2.5-flash-lite` had already gone that way by this date, returning 404
with the name of its successor. If generation or indexing starts failing with a
404, check the model names against `client.models.list()` before looking
anywhere else.

No retrieval framework. 85 tests, none of which touch the network.

## Running locally

```bash
pip install -r requirements.txt
```

Retrieval metrics need nothing else — the embedding cache is committed:

```bash
python -m src.main eval --retrieval-only
```

For answering, copy `.env.example` to `.env` and set `GOOGLE_API_KEY` to a Google
AI Studio key (https://aistudio.google.com/apikey). That one variable is what the
embedder, the answerer, the CLI and the demo all read:

```bash
python -m src.main index
python -m src.main ask "What penalty do we charge a supplier for a late delivery?"
python -m src.main ask "..." --verbose
python -m src.main eval
python -m src.main eval --model gemini-3.6-flash
python -m src.main index --store chroma
```

The same pipeline in a browser, one question at a time, with the retrieval behind
each answer on show:

```bash
streamlit run app.py
```

![The demo showing an answer that reports both sides of a contradiction, with
inline citations and the documents and sections they resolve
to](docs/screenshot.png)

Without a key the demo still runs real retrieval for the four example questions,
because their embeddings are in the committed cache; only the generated answer
needs one. Its Evaluation panel reads `evaluation_results.json`, which a full
`eval` run writes.

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

1. **A reranker over the fused candidates.** Two of the three false refusals
   share one root cause: the right document was retrieved and the wrong section
   of it was. Fusion ranks a chunk by how many lists it appears in; a
   cross-encoder would rank it by whether it answers the question. This is the
   only item that addresses a failure visible in the current numbers, so it is
   first.
2. **An anchor phrase per golden set question, scored alongside document
   recall.** Document-level recall reported 1.000 on a question the system
   failed, which is the trade-off documented in the evaluation section behaving
   exactly as described. A phrase anchor gives section-level accuracy without
   pinning the golden set to chunk ids that move whenever the chunker changes.
3. **Expand context around retrieved chunks rather than merging at index time.**
   Retrieve the small, precise chunk, then include its neighbours in the prompt.
   Would have rescued q22 as a side effect, but it treats the symptom, which is
   why it sits behind the reranker.
4. **The cross-model comparison, on a key with quota.** One run, one table, and
   the choice of generator stops being an assumption. Cheap, blocked only by
   quota, and worth doing before any further prompt work.
5. **Judge answer quality with a model from a different provider.** Substring
   fact checks cannot see a right figure attached to the wrong question. An LLM
   judge can, but a model judging output from its own family has a documented
   self-preference bias, so this needs a second provider and belongs after the
   deterministic metrics are saturated.
6. **Local embeddings through Ollama for a zero-quota offline path.**
   Deliberately rejected for now: it means `torch`, hundreds of megabytes, to
   serve a corpus of 63 chunks. It becomes worth it only if API quota turns into
   the binding constraint.

Done, and left in the history rather than quietly deleted: growing the golden
set from 18 to 50 questions. It was the first item on this list for exactly the
reason the evaluation section now shows -- every conclusion drawn from 18
questions rested on one or two data points, and one of them reversed when the
sample grew.

## Licence

MIT. See `LICENSE`.
