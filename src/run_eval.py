"""
Run the baseline questions through the current pipeline and capture results in
the same per-question format as the v1 freeze. Re-run this after the refactor.

No grading. Run and capture only.

Usage (from the project root):
    python src/run_eval.py                          # cost estimate, then stops
    python src/run_eval.py --confirm                # writes baseline/v2/runs/
    python src/run_eval.py --confirm --out baseline/v2.1 --ids q01 q04
"""
import argparse

import reranker
from freeze_baseline import (QUESTIONS_PATH, print_cost_estimate, read_questions,
                            run_all_questions, write_config, write_chunks_snapshot)

DEFAULT_OUT_DIR = "baseline/v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the baseline questions and capture results.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--ids", nargs="+", help="run only these question ids")
    parser.add_argument("--confirm", action="store_true", help="required; without it only the cost estimate prints")
    parser.add_argument("--snapshot", action="store_true",
                        help="also write chunks.jsonl and config.json for this run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rows = read_questions(args.questions)
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids]

    print_cost_estimate(len(rows))
    if not args.confirm:
        print("\nNo API calls made. Re-run with --confirm to execute.")
        raise SystemExit(1)

    print(f"\nloading cross-encoder ({reranker.MODEL_NAME})...")
    reranker.get_model()

    succeeded, failed = run_all_questions(rows, args.out)

    if args.snapshot:
        chunks = write_chunks_snapshot(args.out)
        write_config(args.out, chunks)

    print(f"\ndone: {len(succeeded)} succeeded, {len(failed)} failed -> {args.out}")
    if failed:
        print(f"failed ids: {failed}")
