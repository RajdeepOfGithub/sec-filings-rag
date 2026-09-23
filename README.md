
# SEC Filings RAG — Hybrid Search with Citation Verification

A retrieval-augmented generation system over SEC filings (10-K, 10-Q) and
earnings call transcripts, combining dense and sparse retrieval, cross-encoder
reranking, grounded generation with citations, and automated citation
verification.

Built as a portfolio project to demonstrate production-style RAG engineering:
measured evaluation, documented failure analysis, and evidence-based
iteration rather than a single untested pipeline.

## Current results (v1.1)

**8/16 on a hand-labeled evaluation set** with source-level ground truth
(document, section, period — not chunk IDs, so the eval survives ingestion
changes). Graded by a deterministic + LLM-judge suite, not by eye.

| Category                                 | Score |
| ---------------------------------------- | ----- |
| Tabular (numeric, from financial tables) | 5/8   |
| Narrative (prose reasoning)              | 2/3   |
| Transcript (earnings call Q&A)           | 0/4   |
| No-answer (should decline)               | 1/1   |

False decline rate: 1/16 (down from 2/16 in the initial baseline).

## What it does

1. **Hybrid retrieval** — dense search (`text-embedding-3-small` + ChromaDB)
   and sparse search (BM25 with a hand-tuned stopword list) run in parallel,
   combined with Reciprocal Rank Fusion.
2. **Deduplication** — near-duplicate chunks (shingle overlap, threshold 0.85)
   are collapsed before fusion, so repeated disclosures in a filing don't
   waste retrieval slots.
3. **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` re-scores the fused
   candidates against the actual question.
4. **Grounded generation** — `gpt-4o-mini` answers strictly from retrieved
   context, with inline citations (`[1]`, `[2]`), explicit scope-matching
   instructions (period/segment discipline), and an instruction to decline
   rather than guess when context is insufficient.
5. **Citation verification** — every citation is independently checked
   (a second LLM call) against its source passage to confirm the passage
   actually supports the specific claim attached to it.
6. **Confidence scoring** — a first-pass signal combining citation-support
   rate and reranker confidence. **Documented limitation:** this does not
   yet separate correct from incorrect answers (0.73 avg confidence on
   wrong answers vs. 0.71 on correct ones) — kept in the system and reported
   honestly rather than hidden, since a badly-calibrated signal is a finding,
   not a reason to delete the feature.

## Why the score is 8/16 and not higher — the investigation

The evaluation set was deliberately built to include the corpus's hardest
cases: numbers split across multi-column tables, figures buried in earnings
call answers, and a no-answer trap. Two findings shaped the current design:

**Table flattening was destroying information before retrieval, not just
before generation.** A SEC filing's tables — read via `BeautifulSoup.get_text()`
— collapse into label-then-bare-numbers strings with no column headers
attached. One eval question asked for a specific quarter's capital ratio; the
correct 14.2% figure was never even *retrieved*, because flattened it read as
seven undifferentiated numbers, while a well-formed sentence about the
*regulatory requirement* (a different, wrong number) won retrieval on
relevance. This is documented in full in the project's decision log,
including a multi-day investigation building a table-to-structured-record
converter, measuring it (score dropped to 5/16 — the converter's records
were correct but scored poorly by a reranker trained on prose, and crowded
out other retrievable content), and making the informed call to ship the
simpler baseline while the more complete fix remains a documented next step.

**Transcript retrieval is the weakest category (0/4).** Earnings-call answers
frequently span multiple sentences of a single speaker's turn, and fixed-size
chunking cuts mid-answer — a chunk can contain the headline figures from a
CFO's answer while the *reasoning* for those figures sits in the next chunk,
never retrieved together. Speaker-turn-aware chunking is scoped but not yet
built (see Known Limitations).

## Known limitations / next steps

- **Table-structure preservation** — investigated in depth (see
  decision log), shipped as future work rather than the current baseline
  after measuring a net regression. The core problem (binding a value to its
  column header) is solved; the downstream effects on the reranker and
  candidate pool are the remaining work.
- **Speaker-turn chunking for transcripts** — scoped, not built. Transcript
  is the weakest-performing category and the clearest next improvement.
- **Confidence scoring** — built, not yet reliable. Needs either more
  training signal or a different feature set.
- **Evaluation set size** — 16 questions currently; scoped to expand to 50+
  with the same source-level ground truth methodology.
- **Corpus** — one company (JPMorgan Chase) currently. Chunking
  generalization to additional filers is scoped but not started.

## Engineering process

Every design decision — including the ones that were later reversed — is
logged with reasoning and measured evidence, including the BM25 stopword
decision reversed after measurement, the RRF fusion tuning, the dedup
threshold derivation, and the full table-conversion investigation with its
regression and root-cause analysis. The project was built to be defensible
in detail, not just to produce a working demo.

## Stack

Python, BeautifulSoup, ChromaDB, OpenAI embeddings + `gpt-4o-mini`,
`rank-bm25`, `sentence-transformers` (cross-encoder reranking).
