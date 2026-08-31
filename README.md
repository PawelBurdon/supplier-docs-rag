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
| document recall@5  | 0.967 | 55 answerable questions, 10 documents |
| anchor recall@5    | 0.897 | did the answering passage make top 5 |
| citation validity  | 1.000 | 70 answers, gemini-3.1-flash-lite    |
| refusal accuracy   | 0.933 | 14 of 15 unanswerable questions      |
| false refusal rate | 0.018 | 1 of 55 answerable questions refused |

The two retrieval rows are fusion alone; with reranking they read 0.991 and
0.982. Every number comes from one run over the current 10-document corpus and
70-question golden set, and Evaluation says which configuration produced it.

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
                                             top 20
                                                |
                     reranking: a model reads the twenty against the
                     question and returns the five worth showing
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

**Reranking** (`src/core/reranker.py`). Fusion decides which twenty candidates
are considered; a model reads those twenty against the question and decides
which five are shown. Fusion scores a chunk by the positions it occupies in two
ranked lists and never reads it against the question, which is exactly the
judgement the five failures above needed.

The reranker is a generative model, not a cross-encoder, and that is a
compromise rather than a preference. Cohere Rerank is purpose-built for this,
faster per query and more predictable in its ordering; a local cross-encoder
through sentence-transformers would also be better at the job. Both were
rejected for the same reason: one asks everybody who clones this repository for
a second provider and a second API key, the other asks for torch and several
hundred megabytes to serve 63 chunks. Reproducibility by a stranger with one key
won. What it costs is speed -- a generation call per query rather than a scoring
pass -- and predictability, since a generative model's ordering can move in ways
a trained scorer's would not.

Orderings are cached on disk and committed, so the metrics still run in CI with
no key. The cache key is a hash of the model, the instructions, the question and
the candidate ids in the order they were proposed. All four matter: the same
question over a different candidate list is a different judgement, and an
ordering produced under different instructions is a different judgement again.
Leaving the prompt out of that key would have made a prompt change invisible,
which is a bug this project nearly shipped -- the key originally held only the
model.

That cache is deliberately fragile. Any change to chunking, embeddings or the
fusion constant changes the candidate list and invalidates every entry, which
costs fifty calls to rebuild. It fails loudly rather than serving a stale
ordering, and that is the right trade for a project whose numbers are the point.

A model asked to sort passages can also return a list that is not a permutation
of its input. A dropped candidate would silently shorten the context; a repeated
one would show the model the same passage twice. Both are repaired in code:
unknown and duplicate numbers are discarded and anything unmentioned is appended
in fused order, so the result is always exactly the candidates that went in.

## Evaluation

`src/evals/golden_set.yaml` holds 70 questions: 28 factual, 18 multi_hop, 6
contradiction, 3 underspecified and 15 unanswerable. Each names the documents
that should be retrieved, the figures a correct answer must contain, a type, a
phrasing and a cohort.

Fifty of them were written against the original 7-document corpus. The other
twenty were added once that corpus reached its ceiling, and they target what the
larger corpus made possible: which version of a document governs, which of two
near-identical contracts applies, questions needing three documents rather than
two, a comparison across two supplier records, an answer that is an absence
rather than a statement, and refusals that sit one step from a figure the corpus
does state. The two cohorts are reported separately and permanently, because an
average across both would hide whether a regression landed on the easy half or
the hard one.

The types gained one member. `underspecified` marks a question two documents
answer equally well because it does not say which supplier or contract it means
-- "what is the liability cap under our framework agreement" now has two correct
answers, 12 months and 6. Correct behaviour is to give both, or to say the
answer depends on that, and these questions are excluded from the false-conflict
denominator: the sources are not disagreeing, the question is incomplete.

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
- **Anchor phrases score the passage, not the file.** Each answerable question
  carries a literal fragment of the passage that answers it, one per expected
  document, and the report says how many of those fragments were actually
  retrieved. Document recall stays alongside it rather than being replaced:
  every earlier run in this README was measured that way, and swapping a metric
  mid-project makes its own history incomparable. A test asserts every anchor is
  verbatim corpus text, because a paraphrased anchor would score zero for ever
  and look like a retrieval failure.
- **Phrasing is a measured axis, not a label.** 11 of the answerable questions
  are written the way somebody holding the paperwork would ask -- naming
  HCS-SLA-2024, clause 5.1, FRM-CRB -- and 27 in plain business language. The
  case for carrying a keyword index next to an embedding model rests entirely on
  those two styles failing differently, and that belongs in a table rather than
  in a paragraph.

### Retrieval, k=5, 55 answerable questions

| mode    | document recall@5 | full coverage | MRR   | anchor recall@5 | all anchors |
|---------|-------------------|---------------|-------|-----------------|-------------|
| hybrid  | 0.967             | 0.93          | 0.861 | 0.897           | 0.82        |
| vector  | 0.976             | 0.95          | 0.894 | 0.864           | 0.78        |
| keyword | 0.861             | 0.78          | 0.782 | 0.773           | 0.67        |

The gap between the two recall columns is the point of having both. Document
recall says every answerable question retrieved the right file. Anchor recall
says that in 13 percent of them the passage carrying the answer was not among
the five chunks the model saw.

| type           | n  | document recall@5 | MRR   | anchor recall@5 | all anchors |
|----------------|----|-------------------|-------|-----------------|-------------|
| factual        | 28 | 1.000             | 0.827 | 0.964           | 0.96        |
| multi_hop      | 18 | 0.935             | 0.843 | 0.806           | 0.61        |
| contradiction  | 6  | 1.000             | 1.000 | 0.917           | 0.83        |
| underspecified | 3  | 0.778             | 1.000 | 0.778           | 0.67        |

By cohort, which is the split worth watching:

| cohort   | n  | document recall@5 | MRR   | anchor recall@5 | all anchors |
|----------|----|-------------------|-------|-----------------|-------------|
| original | 38 | 0.978             | 0.829 | 0.912           | 0.84        |
| extended | 17 | 0.941             | 0.931 | 0.863           | 0.76        |

The extended cohort is harder on both coverage measures, which is what it was
written for. Its higher MRR says something more specific: when these questions
find the right document they find it first, and what they lose is the second or
third document they also need. Multi-hop is the weakest type at 0.61 full anchor
coverage over 18 questions -- the answering passages exist, they are retrieved
individually, and they do not all fit into five chunks together.

By phrasing, document recall and anchor recall:

| style | n  | hybrid        | vector        | keyword       |
|-------|----|---------------|---------------|---------------|
| prose | 40 | 0.954 / 0.871 | 0.967 / 0.863 | 0.833 / 0.762 |
| code  | 15 | 1.000 / 0.967 | 1.000 / 0.867 | 0.933 / 0.800 |

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

### Making the corpus harder, and what that exposed

Two different things happened here and they have different status. Adding
documents is a new experiment. Correcting a golden set key is a repair to the
previous one. They are reported separately on purpose.

**The experiment: three documents added.** Every retrieval metric had reached
1.000 over 38 questions, which does not mean retrieval was solved -- it means
the corpus had nothing left to confuse it with. At 63 chunks, five retrieved
chunks were eight percent of everything there was. Three documents were added to
put real confusion back:

- a **withdrawn** Delivery SLA, version 1.9, with its own lead-time table,
  its own penalty rate and its own escalation ladder, carrying an explicit
  status line and the reference of the version that replaced it
- a **second framework agreement**, for Pedalworks, in near-identical legal
  language to the first, differing where the onboarding record already said it
  differed: net 30, USD, FCA Taichung, and a liability cap of 6 months rather
  than 12
- a packaging and labelling specification, which adds a topic that overlaps the
  inspection procedure without contradicting it

The corpus is now 10 documents and 93 chunks, so five retrieved chunks are five
percent of it rather than eight.

| metric (hybrid, fusion only) | 7 documents | 10 documents |
|------------------------------|-------------|--------------|
| document recall@5            | 1.000       | 0.978        |
| anchor recall@5              | 0.921       | 0.912        |
| all anchors                  | 0.868       | 0.84         |
| MRR                          | 0.877       | 0.829        |

**The repair: two keys were wrong.** Revalidating the existing 50 questions
against the larger corpus showed five whose result changed. Three of them are
the experiment working as intended and are left exactly as they were. Two were
questions whose key had been wrong or incomplete, and the new documents only
made that visible:

- **q37** asks how long a cassette is under warranty and what the supplier
  reimburses. Its key pointed at the Velocore framework agreement and its EUR 12
  handling fee. Cassettes are DRV-CST, which Velocore is not approved to supply;
  the correct contract is the Pedalworks agreement, at USD 15. The expected
  answer was simply untrue. The wording is unchanged, because the corpus does
  support the hop from product to supplier to agreement to fee -- the Pedalworks
  record and the Pedalworks agreement both list cassettes, and Velocore's scope
  is frames and forks. A question that requires that hop is hard, not
  undecidable.
- **q39** asks whether the SLA excuses a supplier's ocean freight delays. The
  Pedalworks framework agreement answers it directly in clause 5.4 and did not
  exist when the question was written, so a third expected document was added.

One question, q35, was ambiguous rather than wrong: it asked what the
distributor can do when "a supplier" misses the OTIF target for three quarters,
and with two framework agreements in the corpus both answer it equally well. It
now names Velocore. The unnamed version is kept as a separate question with its
own type, `underspecified`, because a question that two documents answer
correctly is a real situation in a company's paperwork and worth measuring
rather than editing away.

The rule applied to all of this: a key is corrected only when the justification
holds regardless of which way the number then moves. q37's expected answer was
false, so it was corrected, and the measured result got worse.

**Numbers from before this revalidation are not directly comparable with the
ones after it.** Two questions have different keys and one has different
wording, on top of a corpus that is 48 percent larger. The tables above give the
two states side by side rather than pretending one series runs through both.

### What reranking bought, and what one wording of it cost

Three configurations, same 50 questions, same k. "base" and "conflict-aware" are
two reranker instruction sets, differing by one paragraph.

| metric                    | fusion only | + rerank (base) | + rerank (conflict-aware) |
|---------------------------|-------------|-----------------|---------------------------|
| document recall@5         | 1.000       | 0.987           | 1.000                     |
| anchor recall@5           | 0.921       | 0.987           | 1.000                     |
| all anchors               | 0.868       | 0.974           | 1.000                     |
| MRR                       | 0.877       | 0.987           | 1.000                     |
| citation validity         | 1.000       | 1.000           | 1.000                     |
| refusal accuracy (n=12)   | 1.000       | 1.000           | 1.000                     |
| false refusal rate        | 0.079       | 0.000           | 0.026                     |
| conflict detection (n=6)  | 1.000       | 0.833           | 1.000                     |
| fact coverage             | 0.921       | 0.947           | 0.974                     |

The middle column is the interesting one. Reranking with the plain instruction
recovered all five questions whose answering passage fusion had missed, and lost
a sixth: q40 asks what the penalty is "under clause 5.1 of HCS-SLA-2024", so the
reranker demoted the internal FAQ passage that contradicts that clause. By its
own instructions it was right -- the FAQ is less relevant to a question naming
the SLA -- and for this system it was wrong, because the answer then reported one
side of a disagreement as fact. Relevance to the question is not the same thing
as the set of passages a grounded answer needs, and the reranker had never been
told the difference.

One paragraph was added to the reranker's instructions: a passage stating a
different value for the same thing is useful precisely because it disagrees, even
when the question names another document. That is the third column. It recovers
q40 without giving back any of the other five, and it costs nothing measurable
elsewhere: the false refusal difference against the base variant is q36, a
question that was already refused before any reranking existed and whose scoring
this README disputes below.

Read the third column with its ceiling in mind. Five retrieval metrics at 1.000
across 38 questions does not mean retrieval is solved; it means this golden set
can no longer tell two retrieval systems apart. That is now the first item on the
roadmap.

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

### Answers, gemini-3.1-flash-lite with reranking, 70 questions

| metric              | value | measured over                                    |
|---------------------|-------|--------------------------------------------------|
| citation validity   | 1.000 | 70 answers                                        |
| refusal accuracy    | 0.933 | 15 unanswerable questions, 14 refused             |
| false refusal rate  | 0.018 | 55 answerable questions, 1 refused                |
| conflict detection  | 1.000 | 6 contradiction questions                         |
| false conflict rate | 0.087 | 46 other answerable questions, 4 flagged          |
| fact coverage       | 0.945 | 55 answerable questions, every expected figure    |

Retrieval under the same configuration: document recall 0.991, anchor recall
0.982, all anchors 0.964, MRR 0.973 over 55 answerable questions. Reranking
recovered eight questions whose answering passage fusion had missed, including
both of the hardest new ones -- the three-document question and the first
underspecified one -- and lost none.

Every count is on the table on purpose. Refusal accuracy over 15 questions and
over 4 are different claims with the same name, and the earlier version of this
README reported the second one.

The three failures behind those numbers are all worth more than the numbers.

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

### 2. The right document, the wrong section -- found, fixed, and then made measurable again

q22 asks how quickly a supplier must acknowledge a purchase order. The Delivery
SLA answers it in one sentence. The system refused, and document-level recall
scored the question 1.000, because `delivery_sla` was in the result set -- the
wrong section of it.

```
retrieved for q22, fusion only:
  internal_faq_procurement      [Lead times]
  internal_faq_procurement      [Late deliveries]
  supplier_framework_agreement  [1. Parties and Scope]
  delivery_sla                  [6. Escalation]
  internal_faq_procurement      [Payments]

the answer lives in:
  delivery_sla::003             [3. Committed Lead Times]   fused rank 10 of 20
```

Anchor phrases made that visible: anchor recall 0.921 against document recall
1.000, five questions retrieving the right files and missing the passage.
Reranking then fixed all five, and the numbers moved to 1.000 across the board.

That produced a second problem, which is why this section has two halves. Every
retrieval metric reached its maximum on 38 questions, and a ceiling on a small
sample is an exhausted test set rather than a solved problem: the next change to
chunking or fusion could not have been shown to help or hurt. Three documents
were added to the corpus to put the headroom back, and the numbers came down to
0.978 document recall and 0.912 anchor recall. What the enlarged corpus then
exposed is the next section.

### 3. The withdrawn SLA displaces the record that answers the question

Two questions show the corpus change doing exactly what it was added to do, and
they are left unfixed because they are the point.

q08 asks the lead time for a carbon frame set from Velocore. The current SLA's
lead-time table comes back first, and the **withdrawn** version of that same
table comes back second -- two documents whose section headings, wording and
shape are nearly identical, one of which has been void since January 2024. The
Velocore record, which supplies the tier the question actually needs, is pushed
out of the five. q32 fails the same way for a Pedalworks cassette order.

Neither is a scoring artefact. The system retrieves a passage that looks exactly
like the right answer and is superseded, and it does so because near-duplicate
revisions of the same document are what a document library actually contains.
Before these documents existed, both questions scored 1.000.

### 4. Multi-hop questions retrieve every passage they need, one at a time

Multi-hop is the weakest type in the current run: 0.935 document recall but only
0.61 full anchor coverage across 18 questions. The two numbers together say
something precise. The right documents come back. The passages that answer the
question are individually reachable -- measured one at a time, each of them
ranks inside the candidate pool. What fails is fitting all of them into five
chunks at once.

q58 is the clearest case. It asks how bad Velocore's worst quarter was and what
would happen if that repeated three quarters running, which needs the supplier
record for the figure, the SLA for the escalation and the framework agreement
for the contractual consequence. Retrieval returns the framework agreement, the
withdrawn SLA and the Velocore record: two of the three needed documents, plus
a superseded version of the third.

This is the failure that context expansion, not reranking, is aimed at. A
reranker chooses better among five slots; it cannot make six passages fit into
five. Retrieving a small precise chunk and pulling in its neighbours would give
each hop its own space, which is why that item sits next on the roadmap now that
the metric can finally see the problem.

### 5. The model refuses a question the corpus answers, because the document says not to quote it

q55 asks what OTIF target applied to suppliers in 2023. The withdrawn SLA states
it -- 94 percent for Tier A, 88 for Tier B -- and states its own period of force,
February 2022 to January 2024. Both facts are in the retrieved passages. The
system refused:

> The provided documents do not state the on-time in-full target that applied to
> suppliers in 2023. The withdrawn agreement [1][2] and the current agreement [3]
> provide different targets, but neither document specifies which was in effect
> during 2023.

The second sentence is false, and it is false in an interesting way. The
withdrawn document opens with a warning written into the corpus on purpose --
"do not quote any figure from this document as current" -- and the model appears
to have generalised it from "not current" to "not usable", then failed to read
the date range that answers the question. This is the only false refusal in 55
answerable questions, and it is a reasoning failure rather than a retrieval one:
everything needed was in front of it.

It is also the first failure this project has found that neither retrieval
metrics nor citation checking can see. The answer is grounded, its citations are
valid, and it is wrong.

### 6. The trap worked, and the metric caught the wrong thing

q63 asks what repacking costs per pallet. The packaging specification prices
relabelling at EUR 45 per pallet and says repacking is charged at cost, with no
rate, so the question is unanswerable and the trap is the adjacent number. The
system answered:

> Repacking is charged at cost [1].

It did not take the bait: no EUR 45, no invented rate, one valid citation, and a
true statement. The harness scores it as a failed refusal, which is why refusal
accuracy reads 0.933 rather than 1.000, and that score is defensible only if you
believe the only correct response was "not stated".

That is the third question in this golden set where the metric and a reasonable
reader disagree, after q36 and q09. All three are left scored as they are.
Adjusting a definition after seeing which questions it marks wrong is how an
evaluation stops being one, and three disputed cases out of seventy is itself a
number worth reporting.

### 7. Underspecified questions retrieve both contracts and answer as though there were one

q51 asks what the distributor can do when a supplier misses the OTIF target for
three quarters, without saying which supplier. Reranking retrieves everything
needed: both framework agreements and the SLA, all three anchors present. The
answer:

> Failure to meet the on-time delivery threshold for three consecutive quarters
> constitutes a material breach of the contract [1][2]. Under the terms of the
> agreements, either party may terminate the contract immediately if a material
> breach is not remedied within 30 days [1][2].

"The agreements", plural, cited from both -- and never a word that they are two
different contracts with two different suppliers and two different notice
periods. The ambiguity is retrieved and then flattened. Both underspecified
questions with figures that differ score 0 on fact coverage for the same reason.

This is what the type was added to measure, and it is measuring it: the
retrieval side is solved and the answering side is not. The fix is a prompt
change -- the system prompt requires surfacing disagreement between sources, and
says nothing about two sources that are both right for different subjects -- and
it is deliberately not made in the same commit that reports the finding.

### 8. Everything here rests on 93 chunks of invented text

Ten documents, one corpus, one language, one domain, all written by the same
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
python -m src.main index --rerank          # cache a reranked ordering per golden question
python -m src.main ask "..." --rerank
python -m src.main eval --rerank           # scores hybrid with and without reranking
python -m src.main eval --rerank --rerank-prompt base
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

1. **Expand context around retrieved chunks.** Multi-hop is the weakest type,
   0.61 full anchor coverage before reranking, and its failure has a shape: the
   passages are individually reachable and do not fit into five slots together.
   A reranker chooses better among five; it cannot make six fit. Retrieving the
   precise chunk and pulling in its neighbours gives each hop its own space.
2. **Teach the answerer about two sources that are both right.** The system
   prompt requires surfacing disagreement between sources and says nothing about
   two contracts that are each correct for a different supplier. Underspecified
   questions now retrieve both agreements and the answer flattens them into
   "the agreements", scoring zero on fact coverage. The measurement exists, the
   fix is one paragraph, and the two are deliberately not in the same commit.
3. **The cross-model comparison, on a key with quota.** One run, one table, and
   the choice of generator stops being an assumption.
4. **Judge answer quality with a model from a different provider.** q55 is the
   argument: a grounded answer with valid citations and a false sentence in it.
   No deterministic metric here can see that, and a model from the same family
   judging its own output has a documented self-preference bias.
5. **Replace the LLM reranker with a real cross-encoder, if the constraint ever
   changes.** Faster and steadier than a generative model asked to sort.
   Rejected today because it needs a second key from every reader, or torch.
6. **Local embeddings through Ollama for a zero-quota offline path.** Same
   trade, revisited only if API quota becomes the binding constraint.

Four items are done, and the reasoning is kept rather than deleted.

**Growing the golden set from 18 to 50** came first because every conclusion
drawn from 18 rested on one or two data points. One reversed when the sample
grew.

**Anchor phrases** were promoted above the reranker mid-project, after q22
showed document recall is blind to the class of failure a reranker fixes.

**The reranker** recovered every question whose answering passage fusion had
missed, and cost a contradiction question until its instructions were told that
a passage disagreeing with another is useful for that reason.

**Enlarging the corpus and the golden set** followed from the reranker working
too well: five retrieval metrics at 1.000 over 38 questions is an exhausted test
set. Three documents and twenty questions later, hybrid retrieval reads 0.967
document recall before reranking and 0.991 after, and the failures above are
ones no earlier version of this project could have found.

## Licence

MIT. See `LICENSE`.
