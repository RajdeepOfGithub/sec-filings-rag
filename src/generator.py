import argparse
import math
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from pipeline import search
from retriever import DEFAULT_COLLECTION

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You answer questions about SEC filings using only the numbered context passages provided.

Rules:
- Use only information stated in the context. Do not use outside knowledge about the company, and do not infer figures that are not written down.
- Cite the passage number for every factual claim, like this: [1]. If a claim draws on two passages, cite both: [1][2].
- Quote figures exactly as they appear, including units and the period they cover. Do not convert, round, or aggregate numbers across periods.
- When the question names a period or a scope - "Q2 2026", "the six months ended June 30", "consolidated", "the firm", a single segment - check that the figure you are about to use carries that same period and scope before you use it. The same metric appears in this corpus for several periods, and firmwide alongside per-segment. A figure that looks right is often the right metric for the wrong period or the wrong segment.
- If two or more passages give the same metric for different periods or different segments, say which period and scope the figure you cite belongs to, and do not mix figures from different scopes in one sentence.
- Tables in the context have been flattened, so column headers and values appear as plain sequences of numbers. Before using a figure from a table, state which column header it sits under and check the position matches. If a table lists headers like "2026 2025 2026 2025" followed by values, the first value belongs to the first header, the second to the second, and so on.
- If a table row appears cut off, or if you cannot confidently match a figure to its column, say so rather than guessing.
- Read all the passages before you decide the context is insufficient. The answer is often spread across several of them: combine what they say and answer. No single passage has to contain the whole answer.
- Answer everything the passages do support, and state it plainly rather than hedging. If they support part of the question, give that part in full and name only the part that is genuinely absent.
- If, after reading every passage, the answer is in none of them, say so plainly and state what is missing. Do not guess to fill the gap.
- Be brief. Answer the question asked and stop."""

# Citation numbers are 1-99 only, so bracketed years like [2025] never match.
CITATION_PATTERN = re.compile(r"\[([1-9]\d?(?:\s*,\s*[1-9]\d?)*)\]")


@dataclass
class AnswerResult:
    question: str
    answer_text: str
    citations: list          # [{"number", "chunk_id", "metadata"}]
    retrieved_chunks: list   # results from pipeline.search()
    confidence: float | None = None
    citation_verification: list | None = None  # filled in by verify.verify_answer()


def format_header(number, metadata):
    parts = [metadata["form"], metadata["period"]]
    if metadata["section"] != "none":
        parts.append(metadata["section"])
    return f"[{number}] ({', '.join(parts)})"

def build_context(results):
    """
    Returns (context, citation_map).
    citation_map: [{"number": 1, "chunk_id": "..."}, ...] in block order.
    """
    blocks = []
    citation_map = []

    for number, r in enumerate(results, start=1):
        blocks.append(f"{format_header(number, r['metadata'])}\n{r['text']}")
        citation_map.append({"number": number, "chunk_id": r["id"]})

    return "\n\n".join(blocks), citation_map

def call_llm(system_prompt, user_prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""

def parse_citations(answer_text):
    """
    Matches [1], [1][2] and [1, 2] for numbers 1-99. Returns ints in order of
    first appearance. Does not check that the numbers exist in the context.
    """
    numbers = []
    for match in CITATION_PATTERN.finditer(answer_text):
        for part in match.group(1).split(","):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers

def resolve_citations(numbers, citation_map, results):
    """
    Looks up each cited number. Numbers with no matching block are kept
    with chunk_id and metadata set to None.
    """
    id_by_number = {c["number"]: c["chunk_id"] for c in citation_map}
    metadata_by_id = {r["id"]: r["metadata"] for r in results}

    citations = []
    for number in numbers:
        chunk_id = id_by_number.get(number)
        citations.append({
            "number": number,
            "chunk_id": chunk_id,
            "metadata": metadata_by_id.get(chunk_id),
        })
    return citations


# Confidence

# A decline is an outcome, not a failure: an answer that correctly refuses gets
# this instead of 0, so "I could not find it" never scores below a confident
# wrong answer.
DECLINE_CONFIDENCE = 0.35

# Support is weighted heaviest because it is the only signal that reads the
# answer against its own sources. Retrieval strength is the tiebreaker: it says
# the cited passage was a good match for the question, not that the claim is
# right. Both are first-pass guesses, not tuned.
SUPPORT_WEIGHT = 0.6
RETRIEVAL_WEIGHT = 0.4

DECLINE_PHRASES = (
    "does not provide",
    "does not contain",
    "does not specify",
    "not enough information",
    "no information",
    "is missing",
)


def is_declined(answer_result):
    """A decline cites nothing and says so. An uncited assertion is not a decline."""
    if answer_result.citations:
        return False
    text = answer_result.answer_text.lower()
    return any(phrase in text for phrase in DECLINE_PHRASES)

def support_fraction(verifications):
    if not verifications:
        return 0.0
    return sum(v["supported"] for v in verifications) / len(verifications)

def top_cited_rerank_score(answer_result):
    """
    Cross-encoder scores are logits, roughly -6 to 9 on this corpus, so they go
    through a sigmoid to land in 0-1. Only chunks the answer actually cited count.
    """
    cited_ids = {c["chunk_id"] for c in answer_result.citations}
    scores = [r["rerank_score"] for r in answer_result.retrieved_chunks
              if r["id"] in cited_ids and "rerank_score" in r]
    if not scores:
        return 0.0
    return 1 / (1 + math.exp(-max(scores)))

def compute_confidence(answer_result: AnswerResult) -> float:
    """
    0-1 confidence from signals already on the result. No API calls.

    Declines get DECLINE_CONFIDENCE. Everything else is a weighted mix of how
    much of the answer its own citations support and how strongly the reranker
    matched the cited chunks. Verification is the larger term, so an answer
    whose citations do not hold up cannot score high on retrieval alone. If
    verification was skipped, retrieval is all that is left and carries the
    score by itself.
    """
    if is_declined(answer_result):
        return DECLINE_CONFIDENCE

    retrieval = top_cited_rerank_score(answer_result)
    if answer_result.citation_verification is None:
        return round(retrieval, 3)

    support = support_fraction(answer_result.citation_verification)
    return round(SUPPORT_WEIGHT * support + RETRIEVAL_WEIGHT * retrieval, 3)


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve, build context, generate, parse citations.")
    parser.add_argument("question")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--form", help="e.g. 10-K, 10-Q, earnings_call")
    parser.add_argument("--period", help="e.g. FY2025, Q2-2026")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Chroma collection to query; defaults to the v1 index")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip citation verification (saves one LLM call per citation)")
    return parser.parse_args()

def print_result(context, result):
    print("=" * 70)
    print("CONTEXT")
    print("=" * 70)
    print(context)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer_text)

    print("\n" + "=" * 70)
    print("CITATIONS")
    print("=" * 70)
    if not result.citations:
        print("(none parsed)")
    for c in result.citations:
        print(f"[{c['number']}] -> {c['chunk_id']}")

    if result.citation_verification is None:
        return

    print("\n" + "=" * 70)
    print("CITATION VERIFICATION")
    print("=" * 70)
    if not result.citation_verification:
        print("(no citations to verify)")
    for v in result.citation_verification:
        mark = "SUPPORTED    " if v["supported"] else "NOT SUPPORTED"
        print(f"{mark} [{v['number']}] {v['chunk_id']}")
        print(f"  claim:  {v['claim']}")
        print(f"  reason: {v['reason']}")


def print_confidence(result):
    if result.confidence is None:
        return
    print()
    print("=" * 70)
    print("CONFIDENCE")
    print("=" * 70)
    print(f"{result.confidence:.2f}" + ("  (declined)" if is_declined(result) else ""))


if __name__ == "__main__":
    from verify import verify_answer  # imported here: verify imports generator

    args = parse_args()

    filters = {}
    if args.form:
        filters["form"] = args.form
    if args.period:
        filters["period"] = args.period

    results = search(args.question, n=args.n, top_k=args.top_k, filters=filters or None,
                     collection=args.collection)
    context, citation_map = build_context(results)

    answer_text = call_llm(SYSTEM_PROMPT, f"{context}\n\nQuestion: {args.question}")
    numbers = parse_citations(answer_text)

    result = AnswerResult(
        question=args.question,
        answer_text=answer_text,
        citations=resolve_citations(numbers, citation_map, results),
        retrieved_chunks=results,
    )

    if not args.no_verify:
        verify_answer(result)

    result.confidence = compute_confidence(result)

    print_result(context, result)
    print_confidence(result)
