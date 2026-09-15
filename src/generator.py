import argparse
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from pipeline import search

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You answer questions about SEC filings using only the numbered context passages provided.

Rules:
- Use only information stated in the context. Do not use outside knowledge about the company, and do not infer figures that are not written down.
- Cite the passage number for every factual claim, like this: [1]. If a claim draws on two passages, cite both: [1][2].
- Quote figures exactly as they appear, including units and the period they cover. Do not convert, round, or aggregate numbers across periods.
- Tables in the context have been flattened, so column headers and values appear as plain sequences of numbers. Before using a figure from a table, state which column header it sits under and check the position matches. If a table lists headers like "2026 2025 2026 2025" followed by values, the first value belongs to the first header, the second to the second, and so on.
- If a table row appears cut off, or if you cannot confidently match a figure to its column, say so rather than guessing.
- If the context does not contain enough information to answer, say so plainly and state what is missing. Do not answer partially and hope it passes.
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


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve, build context, generate, parse citations.")
    parser.add_argument("question")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--form", help="e.g. 10-K, 10-Q, earnings_call")
    parser.add_argument("--period", help="e.g. FY2025, Q2-2026")
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

if __name__ == "__main__":
    questions = [
        "How much did the firm spend on share repurchases?",
        "What was the CEO's total compensation?",
        "How many shares were repurchased in Q2 2026, and what did they cost?",
    ]

    for q in questions:
        results = search(q, n=20, top_k=5)
        context, citation_map = build_context(results)

        user_prompt = f"{context}\n\nQuestion: {q}"
        answer_text = call_llm(SYSTEM_PROMPT, user_prompt)

        numbers = parse_citations(answer_text)
        result = AnswerResult(
            question=q,
            answer_text=answer_text,
            citations=resolve_citations(numbers, citation_map, results),
            retrieved_chunks=results,
        )

        print("\n" + "=" * 70)
        print(f"Q: {q}\n")
        print("RETRIEVED:")
        for i, r in enumerate(results, start=1):
            m = r["metadata"]
            print(f"  [{i}] {r['id']}  {m['form']}/{m['period']}")
        print(f"\nANSWER:\n{answer_text}\n")
        print(f"CITATIONS: {result.citations}")
        if "Q2 2026" in q:
            print("\nFULL CONTEXT AS SENT:")
            print(context)