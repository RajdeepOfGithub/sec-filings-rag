"""
Grade a run directory against the baseline questions.

Usage (from the project root):
    python src/run_grades.py --runs baseline/v1/runs
    python src/run_grades.py --runs baseline/v2c/runs --no-judge   # no API calls
    python src/run_grades.py --compare baseline/v1 baseline/v2 baseline/v2b baseline/v2c

Only claims_score calls the LLM; everything else is deterministic. Nothing is
written to the run directories.
"""
import argparse
import json
import os
from collections import Counter

import grading
from grading import grade

QUESTIONS_PATH = "data/baseline_questions.jsonl"

# gpt-4o-mini, USD per 1M tokens, for the judge-cost estimate.
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60
EST_JUDGE_INPUT_TOKENS = 500
EST_JUDGE_OUTPUT_TOKENS = 30


def read_questions(path=QUESTIONS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return {row["id"]: row for row in (json.loads(line) for line in f if line.strip())}

def read_runs(runs_dir):
    runs = {}
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json") or name.endswith(".error.json"):
            continue
        with open(os.path.join(runs_dir, name), "r", encoding="utf-8") as f:
            record = json.load(f)
        runs[record["question_id"]] = record
    return runs

def resolve_runs_dir(path):
    """Accepts either baseline/v1 or baseline/v1/runs."""
    candidate = os.path.join(path, "runs")
    return candidate if os.path.isdir(candidate) else path

def grade_directory(path, questions, judge=True):
    runs = read_runs(resolve_runs_dir(path))
    return {qid: grade(run, questions[qid], judge=judge)
            for qid, run in runs.items() if qid in questions}


def flag(value):
    return {True: "yes", False: "NO", None: "-"}[value]

def score_cell(value):
    return "-" if value is None else f"{value:.2f}"

def print_table(grades, label, judged=True):
    print(f"\n{'=' * 104}")
    print(f"{label}")
    print("=" * 104)
    print(f"{'id':<5}{'category':<11}{'correct':<9}{'doc':<6}{'period':<8}{'unit':<6}"
          f"{'value':<7}{'claims':<8}{'false_decline':<15}{'cites':<6}")
    print("-" * 104)

    for qid in sorted(grades):
        g = grades[qid]
        print(f"{qid:<5}{g['category']:<11}{flag(g['correct']):<9}{flag(g['document_correct']):<6}"
              f"{flag(g['period_correct']):<8}{flag(g['unit_correct']):<6}"
              f"{flag(g['value_correct']):<7}{score_cell(g['claims_score']):<8}"
              f"{flag(g['false_decline']):<15}{g['citations']:<6}")

    correct = sum(1 for g in grades.values() if g["correct"])
    print("-" * 104)
    print(f"OVERALL: {correct}/{len(grades)}")

    print("\nby category:")
    for category in ("tabular", "narrative", "transcript", "no_answer"):
        members = [g for g in grades.values() if g["category"] == category]
        if members:
            hits = sum(1 for g in members if g["correct"])
            print(f"  {category:<12} {hits}/{len(members)}")

    extras = [
        ("false declines", sum(1 for g in grades.values() if g["false_decline"])),
        ("unit stated but wrong/missing",
         sum(1 for g in grades.values() if g["unit_correct"] is False)),
        ("right number, wrong scope",
         sum(1 for g in grades.values()
             if g["value_correct"] and not (g["document_correct"] and g["period_correct"]))),
        ("claims graders that failed to parse",
         sum(1 for g in grades.values()
             if judged and g["category"] in ("narrative", "transcript")
             and g["claims_score"] is None)),
    ]
    print("\nflags:")
    for name, count in extras:
        print(f"  {name:<34} {count}")

def print_comparison(all_grades):
    versions = list(all_grades)
    print(f"\n{'=' * 104}")
    print("FOUR VERSIONS, GRADED IDENTICALLY")
    print("=" * 104)

    header = f"{'id':<5}{'category':<11}" + "".join(f"{v:<14}" for v in versions)
    print(header)
    print("-" * len(header))

    for qid in sorted(next(iter(all_grades.values()))):
        cells = ""
        for version in versions:
            g = all_grades[version].get(qid)
            if g is None:
                cells += f"{'(missing)':<14}"
                continue
            mark = "PASS" if g["correct"] else "fail"
            if g["claims_score"] is not None:
                mark += f" {g['claims_score']:.2f}"
            cells += f"{mark:<14}"
        category = all_grades[versions[0]][qid]["category"]
        print(f"{qid:<5}{category:<11}{cells}")

    print("-" * len(header))
    totals = f"{'TOTAL':<5}{'':<11}"
    for version in versions:
        grades = all_grades[version]
        totals += f"{sum(1 for g in grades.values() if g['correct'])}/{len(grades):<12}"
    print(totals)

    print("\nper category:")
    for category in ("tabular", "narrative", "transcript", "no_answer"):
        line = f"  {category:<12}"
        for version in versions:
            members = [g for g in all_grades[version].values() if g["category"] == category]
            hits = sum(1 for g in members if g["correct"])
            line += f"{hits}/{len(members):<12}"
        print(line)

    print("\nflags:")
    for name, predicate in (
        ("false declines", lambda g: g["false_decline"]),
        ("right number, wrong scope",
         lambda g: g["value_correct"] and not (g["document_correct"] and g["period_correct"])),
        ("unit stated but wrong/missing", lambda g: g["unit_correct"] is False),
    ):
        line = f"  {name:<32}"
        for version in versions:
            line += f"{sum(1 for g in all_grades[version].values() if predicate(g)):<14}"
        print(line)

def print_cost():
    calls = grading.USAGE["judge_calls"]
    if not calls:
        print("\nno judge calls made (deterministic graders only)")
        return

    cost = calls * (EST_JUDGE_INPUT_TOKENS * PRICE_INPUT
                    + EST_JUDGE_OUTPUT_TOKENS * PRICE_OUTPUT) / 1_000_000
    print(f"\njudge: {calls} calls over {grading.USAGE['claims_judged']} claims, "
          f"estimated cost ${cost:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grade baseline run directories.")
    parser.add_argument("--runs", help="a single run directory (baseline/v1 or baseline/v1/runs)")
    parser.add_argument("--compare", nargs="+", help="two or more run directories to compare")
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--no-judge", action="store_true",
                        help="skip claims_score, so no API calls are made")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    questions = read_questions(args.questions)
    judge = not args.no_judge

    if args.compare:
        all_grades = {}
        for path in args.compare:
            label = os.path.basename(path.rstrip("/\\")) or path
            print(f"grading {label}...")
            all_grades[label] = grade_directory(path, questions, judge=judge)
            print_table(all_grades[label], label, judged=judge)
        print_comparison(all_grades)
        print_cost()
    elif args.runs:
        label = args.runs
        grades = grade_directory(args.runs, questions, judge=judge)
        print_table(grades, label, judged=judge)
        print_cost()
    else:
        raise SystemExit("pass --runs or --compare")
