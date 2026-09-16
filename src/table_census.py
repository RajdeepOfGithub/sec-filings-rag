"""
Table census report: what table shapes exist in the corpus, and how the
conservative classifier buckets them.

Report only. Nothing here converts a table.

Usage (from the project root):
    python src/table_census.py
    python src/table_census.py --examples 5
"""
import argparse
import warnings
from collections import Counter

from table_profile import DOCUMENTS, find_table, profile_document

warnings.filterwarnings("ignore")

BUCKETS = ["likely_data", "likely_layout", "merged_header", "nested",
           "mostly_text", "ixbrl_heavy", "ambiguous"]

# Tables with measured retrieval or answer failures attached.
NAMED_TABLES = [
    ("q01 FAILS: 10-Q share repurchases, four columns over two superheaders",
     "JPMC_10-Q_Q2-2026", ["21.7", "29.8", "49.3", "59.8"]),
    ("q04 FAILS: 10-Q consolidated financial highlights, seven period columns",
     "JPMC_10-Q_Q2-2026", ["2Q26", "1Q26", "4Q25"]),
    ("CONTROL, answered correctly: 10-K FY2025 share repurchases, three columns",
     "JPMC_10-K_FY2025", ["114.4", "91.7", "69.5"]),
]


def percentiles(values, points=(0, 25, 50, 75, 90, 100)):
    if not values:
        return {p: 0 for p in points}
    ordered = sorted(values)
    return {p: ordered[min(int(p / 100 * len(ordered)), len(ordered) - 1)] for p in points}

def print_distribution(label, values, fmt="{:.0f}"):
    stats = percentiles(values)
    cells = "  ".join(f"p{p}={fmt.format(v)}" for p, v in stats.items())
    print(f"  {label:<22} {cells}")

def print_bucket_counts(all_profiles):
    print("\n" + "=" * 78)
    print("BUCKETS PER DOCUMENT")
    print("=" * 78 + "\n")

    print(f"{'bucket':<18}" + "".join(f"{doc_id:>30}" for doc_id in all_profiles) + f"{'total':>8}")
    print("-" * (18 + 30 * len(all_profiles) + 8))

    print(f"{'(tables)':<18}" + "".join(f"{len(p):>30,}" for p in all_profiles.values())
          + f"{sum(len(p) for p in all_profiles.values()):>8,}")
    print()

    print("primary bucket:")
    for bucket in BUCKETS:
        counts = [sum(1 for p in profiles if p.primary_bucket == bucket) for profiles in all_profiles.values()]
        if sum(counts):
            print(f"  {bucket:<16}" + "".join(f"{c:>30,}" for c in counts) + f"{sum(counts):>8,}")

    print("\nall buckets that apply (tables can carry several):")
    for bucket in BUCKETS:
        counts = [sum(1 for p in profiles if bucket in p.buckets) for profiles in all_profiles.values()]
        if sum(counts):
            print(f"  {bucket:<16}" + "".join(f"{c:>30,}" for c in counts) + f"{sum(counts):>8,}")

def print_shape_distributions(profiles, doc_id):
    print(f"\n{doc_id}  ({len(profiles):,} tables)")
    print_distribution("rows", [p.rows for p in profiles])
    print_distribution("max_cols", [p.max_cols for p in profiles])
    print_distribution("numeric_ratio", [p.numeric_ratio for p in profiles], "{:.2f}")
    print_distribution("empty_cell_ratio", [p.empty_cell_ratio for p in profiles], "{:.2f}")
    print_distribution("col_consistency", [p.col_count_consistency for p in profiles], "{:.2f}")
    print_distribution("avg_cell_chars", [p.avg_cell_chars for p in profiles], "{:.1f}")
    print_distribution("total_chars", [p.total_chars for p in profiles], "{:,.0f}")

def print_merged_headers(all_profiles):
    print("\n" + "=" * 78)
    print("MERGED HEADERS")
    print("=" * 78 + "\n")

    for doc_id, profiles in all_profiles.items():
        if not profiles:
            continue
        merged = [p for p in profiles if "merged_header" in p.buckets]
        superheader = [p for p in profiles if p.superheader_suspected]
        rowspan = [p for p in profiles if p.rowspan_count]
        colspan = [p for p in profiles if p.colspan_count]

        print(f"{doc_id}:")
        print(f"  merged_header bucket        {len(merged):>5,} / {len(profiles):,} ({len(merged) / len(profiles):.0%})")
        print(f"  superheader suspected       {len(superheader):>5,}  <- the q01/q04 failure shape")
        print(f"  any rowspan > 1             {len(rowspan):>5,}")
        print(f"  any colspan > 1             {len(colspan):>5,}")

def print_ixbrl(all_profiles):
    print("\n" + "=" * 78)
    print("iXBRL COVERAGE")
    print("=" * 78 + "\n")

    for doc_id, profiles in all_profiles.items():
        if not profiles:
            continue
        facts = [p.ix_nonfraction_count + p.ix_nonnumeric_count for p in profiles]
        tagged = [f for f in facts if f]
        print(f"{doc_id}:")
        print(f"  tables carrying facts       {len(tagged):>5,} / {len(profiles):,} ({len(tagged) / len(profiles):.0%})")
        print(f"  total facts                 {sum(facts):>5,}  "
              f"(nonFraction {sum(p.ix_nonfraction_count for p in profiles):,}, "
              f"nonNumeric {sum(p.ix_nonnumeric_count for p in profiles):,})")
        print(f"  ixbrl_heavy bucket          {sum(1 for p in profiles if 'ixbrl_heavy' in p.buckets):>5,}")

def print_examples(all_profiles, per_bucket):
    print("\n" + "=" * 78)
    print(f"REPRESENTATIVE EXAMPLES ({per_bucket} per bucket)")
    print("=" * 78)

    everything = [p for profiles in all_profiles.values() for p in profiles]

    for bucket in BUCKETS:
        members = [p for p in everything if p.primary_bucket == bucket]
        if not members:
            continue

        print(f"\n--- {bucket}  ({len(members):,} tables) ---")
        step = max(len(members) // per_bucket, 1)
        for profile in members[::step][:per_bucket]:
            print(f"\n  {profile.table_id}")
            print(f"  section: {profile.section_id}")
            print(f"  rows={profile.rows} max_cols={profile.max_cols} numeric={profile.numeric_ratio} "
                  f"empty={profile.empty_cell_ratio} avg_cell={profile.avg_cell_chars} "
                  f"ix={profile.ix_nonfraction_count + profile.ix_nonnumeric_count}")
            print(f"  buckets: {profile.buckets}")
            print(f"  get_text(): {profile.text_preview}")

def print_named_tables():
    print("\n" + "=" * 78)
    print("TABLES WITH MEASURED FAILURES (shown regardless of bucket)")
    print("=" * 78)

    for label, doc_id, needles in NAMED_TABLES:
        _, profile = find_table(doc_id, contains=needles)
        print(f"\n--- {label} ---")
        if profile is None:
            print("  NOT FOUND")
            continue

        print(f"  {profile.table_id}")
        print(f"  section: {profile.section_id}")
        print(f"  primary: {profile.primary_bucket}   buckets: {profile.buckets}")
        print(f"  rows={profile.rows}  max_cols={profile.max_cols}  min_cols={profile.min_cols}  "
              f"consistency={profile.col_count_consistency}")
        print(f"  numeric_ratio={profile.numeric_ratio}  empty_cell_ratio={profile.empty_cell_ratio}  "
              f"year_count={profile.year_count}  currency={profile.currency_count}  percent={profile.percent_count}")
        print(f"  rowspan={profile.rowspan_count}  colspan={profile.colspan_count}  "
              f"header_row={profile.has_header_row_candidate}  SUPERHEADER={profile.superheader_suspected}")
        print(f"  ix facts={profile.ix_nonfraction_count + profile.ix_nonnumeric_count}  "
              f"avg_cell_chars={profile.avg_cell_chars}  total_chars={profile.total_chars}")
        print(f"  get_text(): {profile.text_preview}")


def parse_args():
    parser = argparse.ArgumentParser(description="Table census over the corpus.")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--docs", nargs="+", default=sorted(DOCUMENTS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_profiles = {}
    for doc_id in args.docs:
        print(f"profiling {doc_id}...")
        all_profiles[doc_id] = profile_document(doc_id)

    print_bucket_counts(all_profiles)

    print("\n" + "=" * 78)
    print("SHAPE DISTRIBUTIONS")
    print("=" * 78)
    for doc_id, profiles in all_profiles.items():
        if profiles:
            print_shape_distributions(profiles, doc_id)

    print_merged_headers(all_profiles)
    print_ixbrl(all_profiles)
    print_examples(all_profiles, args.examples)
    print_named_tables()

    print("\n" + "=" * 78)
    total = sum(len(p) for p in all_profiles.values())
    ambiguous = sum(1 for profiles in all_profiles.values() for p in profiles if p.primary_bucket == "ambiguous")
    print(f"{total:,} tables profiled, {ambiguous:,} left ambiguous on purpose "
          f"({ambiguous / total:.0%})" if total else "no tables found")
