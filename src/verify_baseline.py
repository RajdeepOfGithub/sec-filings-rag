"""
Run citation verification over a frozen baseline's answers.

Reads the run files a baseline already captured and verifies the citations in
them. Nothing is retrieved and no answer is regenerated, so the answers stay
exactly as that baseline froze them; only verify.py's calls are new.

Cost is one gpt-4o-mini call per citation occurrence, not per question.

Usage (from the project root):
    python src/verify_baseline.py                              # cost estimate, then stops
    python src/verify_baseline.py --confirm
    python src/verify_baseline.py --base baseline/v1 --confirm
    python src/verify_baseline.py --confirm --ids q05 q09
"""
import argparse
import glob
import json
import os

from generator import AnswerResult
from verify import split_claims, verify_answer

DEFAULT_BASE = "baseline/v2c"

# USD per 1M tokens, matching freeze_baseline.py.
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60

# One verification call is a passage plus a claim in, one line out.
EST_INPUT_TOKENS = 700
EST_OUTPUT_TOKENS = 25


def read_runs(base, ids=None):
    records = []
    for path in sorted(glob.glob(os.path.join(base, "runs", "q*.json"))):
        if path.endswith(".error.json"):
            continue
        record = json.load(open(path, encoding="utf-8"))
        if ids and record["question_id"] not in ids:
            continue
        records.append(record)
    return records

def count_citations(records):
    """Citation occurrences, which is what the cost scales with: [1][2] is two."""
    return sum(len(split_claims(r["answer_text"])) for r in records)

def print_cost_estimate(citation_count):
    prompt = citation_count * EST_INPUT_TOKENS * PRICE_INPUT / 1_000_000
    completion = citation_count * EST_OUTPUT_TOKENS * PRICE_OUTPUT / 1_000_000
    print(f"{citation_count} citations x 1 gpt-4o-mini verification call")
    print(f"  estimated cost: ${prompt + completion:.4f}")

def verify_run(record):
    """Rebuilds the AnswerResult this baseline captured and verifies its citations."""
    result = AnswerResult(
        question=record["question"],
        answer_text=record["answer_text"],
        citations=record["resolved_citations"],
        retrieved_chunks=record["reranked"],
    )
    return verify_answer(result)

def write_report(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the citations in a frozen baseline's answers.")
    parser.add_argument("--base", default=DEFAULT_BASE, help="baseline directory, e.g. baseline/v1")
    parser.add_argument("--out", help="report path (default: <base>/verification.json)")
    parser.add_argument("--ids", nargs="+", help="verify only these question ids")
    parser.add_argument("--confirm", action="store_true",
                        help="required; without it only the cost estimate prints")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    records = read_runs(args.base, args.ids)
    if not records:
        raise SystemExit(f"no run files under {args.base}/runs")

    citation_count = count_citations(records)
    print(f"{args.base}: {len(records)} questions")
    print_cost_estimate(citation_count)
    if not args.confirm:
        print()
        print("No API calls made. Re-run with --confirm to execute.")
        raise SystemExit(1)

    rows, total, supported = [], 0, 0
    for record in records:
        result = verify_run(record)
        hits = sum(v["supported"] for v in result.citation_verification)
        total += len(result.citation_verification)
        supported += hits
        rows.extend({"question_id": record["question_id"], **v} for v in result.citation_verification)
        print(f"{record['question_id']}: {len(result.citation_verification)} citation(s), {hits} supported",
              flush=True)

    print()
    print(f"TOTAL {total} citations, {supported} supported, {total - supported} not supported")
    for row in rows:
        if not row["supported"]:
            print(f"  FAIL {row['question_id']} [{row['number']}] {row['chunk_id']}")
            print(f"       claim:  {row['claim']}")
            print(f"       reason: {row['reason']}")

    write_report(args.out or os.path.join(args.base, "verification.json"), rows)
