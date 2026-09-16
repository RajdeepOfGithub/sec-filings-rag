"""
Conversion report: what the table converter did to the corpus.

Reports routing counts and reason codes, the serialization delta per
document, before/after samples, and the three tables with measured failures
attached.

Nothing is wired into corpus.py and nothing is indexed. Inspection only.

Usage (from the project root):
    python src/table_conversion_report.py
    python src/table_conversion_report.py --samples 5
"""
import argparse
import warnings

from ir import TableBlock, serialize_document
from loader import load_document, load_document_ir
from table_convert import (AMBIGUOUS_PRESERVED, CONVERTED_DATA, CONVERTED_SUPERHEADER,
                           LAYOUT_PASSTHROUGH, convert_table)
from table_profile import DOCUMENTS, find_table

warnings.filterwarnings("ignore")

STATUSES = [CONVERTED_DATA, CONVERTED_SUPERHEADER, LAYOUT_PASSTHROUGH, AMBIGUOUS_PRESERVED]

NAMED_TABLES = [
    ("q01 FAILURE: 10-Q share repurchases (v1 answered 29.8M / $7,500M, the 2025 column)",
     "JPMC_10-Q_Q2-2026", ["21.7", "29.8", "49.3", "59.8"]),
    ("q04 FAILURE: 10-Q consolidated financial highlights (v1 answered 11.5%, the requirement)",
     "JPMC_10-Q_Q2-2026", ["2Q26", "1Q26", "4Q25"]),
    ("CONTROL: 10-K FY2025 share repurchases (v1 answered correctly)",
     "JPMC_10-K_FY2025", ["114.4", "91.7", "69.5"]),
]


def flattened(block):
    """What this table looked like before conversion."""
    from bs4 import BeautifulSoup
    return BeautifulSoup(block.raw_html, "lxml").get_text(separator=" ", strip=True)

def load_all(doc_ids):
    loaded = {}

    for doc_id in doc_ids:
        path, doc_type = DOCUMENTS[doc_id]
        stats = {}
        document = load_document_ir(path, doc_type, doc_id, convert_tables=True, stats=stats)
        loaded[doc_id] = {
            "document": document,
            "stats": stats,
            "before": load_document(path, doc_type),
            "after": serialize_document(document),
            "tables": [b for b in document.blocks if isinstance(b, TableBlock)],
        }
        print(f"  loaded {doc_id}: {len(document.blocks)} blocks, "
              f"{len(loaded[doc_id]['tables'])} tables")

    return loaded

def print_routing(loaded):
    print("\n" + "=" * 78)
    print("ROUTING")
    print("=" * 78 + "\n")

    totals = {}
    for doc_id, data in loaded.items():
        counts = {status: data["stats"].get(status, 0) for status in STATUSES}
        print(f"{doc_id}")
        for status, count in counts.items():
            totals[status] = totals.get(status, 0) + count
            print(f"  {status:<24} {count:>6,}")
        print()

    print("corpus total")
    for status in STATUSES:
        print(f"  {status:<24} {totals.get(status, 0):>6,}")

def print_reason_codes(loaded):
    print("\n" + "=" * 78)
    print("REASON CODES")
    print("=" * 78 + "\n")

    reasons = {}
    for data in loaded.values():
        for key, count in data["stats"].items():
            if key.isupper() or key.startswith("table_"):
                reasons[key] = reasons.get(key, 0) + count

    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<34} {count:>6,}")

def print_deltas(loaded):
    print("\n" + "=" * 78)
    print("SERIALIZATION DELTA (the acceptance test deliberately no longer holds)")
    print("=" * 78 + "\n")

    print(f"{'document':<30}{'before':>14}{'after':>14}{'delta':>14}{'change':>10}")
    print("-" * 82)

    for doc_id, data in loaded.items():
        before, after = len(data["before"]), len(data["after"])
        delta = after - before
        marker = "  IDENTICAL" if delta == 0 and data["before"] == data["after"] else ""
        print(f"{doc_id:<30}{before:>14,}{after:>14,}{delta:>+14,}{delta / before:>9.0%}{marker}")

    print("\n  documents with no tables must stay byte-identical:")
    for doc_id, data in loaded.items():
        if not data["tables"]:
            identical = data["before"] == data["after"]
            print(f"    {doc_id:<30} {'IDENTICAL' if identical else 'CHANGED - REGRESSION'}")

def print_samples(loaded, count):
    print("\n" + "=" * 78)
    print(f"BEFORE / AFTER SAMPLES ({count} per status)")
    print("=" * 78)

    by_status = {}
    for doc_id, data in loaded.items():
        for block in data["tables"]:
            by_status.setdefault(block.conversion_status, []).append((doc_id, block))

    for status in STATUSES:
        blocks = by_status.get(status, [])
        if not blocks:
            continue

        print(f"\n--- {status} ({len(blocks):,} tables) ---")
        step = max(len(blocks) // count, 1)
        for doc_id, block in blocks[::step][:count]:
            print(f"\n  {block.block_id}")
            print(f"  reason: {block.reason_code}")
            print(f"  BEFORE: {flattened(block)[:280]}")
            after = block.serialized_text()
            for line in after.split("\n")[:4]:
                print(f"  AFTER : {line[:280]}")
            extra = len(after.split("\n")) - 4
            if extra > 0:
                print(f"          ... {extra} more record(s)")

def print_named_tables():
    print("\n" + "=" * 78)
    print("THE THREE TABLES THAT MATTER, IN FULL")
    print("=" * 78)

    for label, doc_id, needles in NAMED_TABLES:
        table, profile = find_table(doc_id, contains=needles)
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

        if table is None:
            print("  NOT FOUND")
            continue

        conversion = convert_table(table, profile)
        print(f"\nstatus: {conversion.status}   reason: {conversion.reason}")
        print(f"title:  {conversion.title!r}   scale: {conversion.unit!r}   "
              f"superheader: {profile.superheader_suspected}")

        print("\nBEFORE (today's get_text output, one flat string):\n")
        print("  " + " ".join(table.get_text(separator=" ", strip=True).split()))

        print(f"\nAFTER ({len(conversion.records)} self-describing records):\n")
        for record in conversion.records:
            print("  " + record)


def parse_args():
    parser = argparse.ArgumentParser(description="Report on table conversion.")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--docs", nargs="+", default=sorted(DOCUMENTS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("loading documents (no API calls, nothing indexed)...")
    loaded = load_all(args.docs)

    print_routing(loaded)
    print_reason_codes(loaded)
    print_deltas(loaded)
    print_samples(loaded, args.samples)
    print_named_tables()
