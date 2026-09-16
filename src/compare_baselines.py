"""
Side-by-side comparison of two run directories, for human reading.

Chunk IDs do not match across versions by design, so they are printed next to
each other rather than compared. No scoring, no metrics.

Usage (from the project root):
    python src/compare_baselines.py baseline/v1 baseline/v2
"""
import argparse
import json
import os


def load_runs(run_dir):
    runs_dir = os.path.join(run_dir, "runs")
    runs = {}

    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json") or name.endswith(".error.json"):
            continue
        with open(os.path.join(runs_dir, name), "r", encoding="utf-8") as f:
            record = json.load(f)
        runs[record["question_id"]] = record

    return runs

def reranked_ids(record):
    """(id, kind) - v1 chunks predate the kind metadata, so they show as '-'."""
    return [(r["id"], r["metadata"].get("kind", "-")) for r in record["reranked"]]

def short_kind(kind):
    return {"table_records": "TABLE", "narrative": "prose", "-": "-"}.get(kind, kind)

def cited_sources(record):
    """(form, period) pairs for resolved citations, in citation order."""
    sources = []
    for citation in record["resolved_citations"]:
        meta = citation["metadata"]
        pair = (meta["form"], meta["period"]) if meta else (None, None)
        if pair not in sources:
            sources.append(pair)
    return sources

def format_sources(sources):
    if not sources:
        return "(no citations)"
    return ", ".join(f"{form}/{period}" if form else "UNRESOLVED" for form, period in sources)

def compare_question(qid, a, b, label_a, label_b):
    print("=" * 78)
    print(f"{qid}  [{a.get('category', '?')}]  {a['question']}")
    print("=" * 78)

    print(f"expected: {a.get('expected')}")
    answer_changed = a["answer_text"] != b["answer_text"]
    print(f"answer text changed: {'YES' if answer_changed else 'no'}")

    sources_a, sources_b = cited_sources(a), cited_sources(b)
    print(f"cited sources changed: {'YES' if sources_a != sources_b else 'no'}")
    print(f"  {label_a}: {format_sources(sources_a)}")
    print(f"  {label_b}: {format_sources(sources_b)}")

    print("reranked top 5 (ids differ across versions by design):")
    ids_a, ids_b = reranked_ids(a), reranked_ids(b)
    for i in range(max(len(ids_a), len(ids_b))):
        left = f"{ids_a[i][0]} [{short_kind(ids_a[i][1])}]" if i < len(ids_a) else ""
        right = f"{ids_b[i][0]} [{short_kind(ids_b[i][1])}]" if i < len(ids_b) else ""
        print(f"  #{i + 1}  {left:<44} | {right}")

    print(f"\n{label_a} answer:\n{a['answer_text']}\n")
    print(f"{label_b} answer:\n{b['answer_text']}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two baseline run directories.")
    parser.add_argument("dir_a")
    parser.add_argument("dir_b")
    parser.add_argument("--ids", nargs="+", help="compare only these question ids")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    runs_a = load_runs(args.dir_a)
    runs_b = load_runs(args.dir_b)
    label_a, label_b = args.dir_a, args.dir_b

    shared = [qid for qid in runs_a if qid in runs_b]
    if args.ids:
        shared = [qid for qid in shared if qid in args.ids]

    for qid in shared:
        compare_question(qid, runs_a[qid], runs_b[qid], label_a, label_b)

    only_a = sorted(set(runs_a) - set(runs_b))
    only_b = sorted(set(runs_b) - set(runs_a))
    print("=" * 78)
    print(f"compared {len(shared)} questions")
    if only_a:
        print(f"only in {label_a}: {only_a}")
    if only_b:
        print(f"only in {label_b}: {only_b}")
