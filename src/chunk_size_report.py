"""
What record-aware chunking costs: chunk counts, token estimate and embedding
cost for the converted corpus versus the frozen v1 config.

Estimates only. No API calls, nothing indexed, corpus.py untouched.

Token counts are estimated from characters (no tokenizer installed in this
venv), so treat them as a bracket, not a measurement - see CHARS_PER_TOKEN.

Usage (from the project root):
    python src/chunk_size_report.py
    python src/chunk_size_report.py --skip-v1     # skip re-parsing for v1 counts
"""
import argparse
import warnings
from collections import Counter

from chunker import structure_aware_chunk_ir
from ir import TableBlock
from loader import load_document_ir
from sections import attribute_sections_report
from table_profile import DOCUMENTS

warnings.filterwarnings("ignore")

# text-embedding-3-small, USD per 1M tokens.
EMBEDDING_PRICE_PER_MILLION = 0.02
# English prose runs ~4 chars/token. Records are denser (repeated labels,
# digits, pipes), so the real count is likely higher than this estimate.
CHARS_PER_TOKEN = 4.0
V1_FROZEN_CHUNKS = 4138   # baseline/v1/config.json


def chunk_documents(chunk_size, overlap):
    results = {}

    for doc_id, (path, doc_type) in sorted(DOCUMENTS.items()):
        print(f"  chunking {doc_id}...")
        document = load_document_ir(path, doc_type, doc_id, convert_tables=True)
        attributed, _ = attribute_sections_report(document, strict=False)

        stats = {}
        chunks = structure_aware_chunk_ir(attributed, chunk_size, overlap, stats)
        tables = [b for b in attributed.blocks if isinstance(b, TableBlock) and b.records]

        results[doc_id] = {
            "chunks": chunks,
            "stats": stats,
            "record_counts": [len(b.records) for b in tables],
            "titles": {b.block_id: b.table_title for b in tables},
            "converted": {b.block_id for b in tables
                          if b.conversion_status.startswith("converted")},
        }

    return results

def v1_chunk_counts(chunk_size, overlap):
    """
    The frozen v1 configuration: unconverted text through the old chunkers.

    Deliberately not build_corpus(). Since step 1b that routes through
    load_document_ir(), so it now picks up converted tables and no longer
    reproduces the frozen baseline.
    """
    from chunker import fixed_size_chunk, structure_aware_chunk
    from loader import load_document

    counts = {}
    for doc_id, (path, doc_type) in sorted(DOCUMENTS.items()):
        text = load_document(path, doc_type)
        if doc_id.startswith("JPMC_10-K"):
            counts[doc_id] = len(structure_aware_chunk(text, chunk_size, overlap))
        else:
            counts[doc_id] = len(fixed_size_chunk(text, chunk_size, overlap))

    return counts

def build_corpus_count():
    """What build_corpus() produces today, to show the step 1b side effect."""
    from corpus import build_corpus
    return len(build_corpus())

def tokens_of(chunks):
    return sum(len(c["text"]) for c in chunks) / CHARS_PER_TOKEN

def percentiles(values, points=(0, 25, 50, 75, 90, 95, 100)):
    if not values:
        return {p: 0 for p in points}
    ordered = sorted(values)
    return {p: ordered[min(int(p / 100 * len(ordered)), len(ordered) - 1)] for p in points}


def print_counts(results, v1_counts):
    print("\n" + "=" * 78)
    print("CHUNK COUNTS: frozen v1 vs converted tables")
    print("=" * 78 + "\n")

    print(f"{'document':<30}{'v1':>10}{'v2':>10}{'delta':>12}{'change':>10}")
    print("-" * 72)

    v1_total = v2_total = 0
    for doc_id, data in results.items():
        v1 = v1_counts.get(doc_id, 0) if v1_counts else 0
        v2 = len(data["chunks"])
        v1_total += v1
        v2_total += v2
        change = f"{(v2 / v1 - 1):+.0%}" if v1 else "n/a"
        print(f"{doc_id:<30}{v1:>10,}{v2:>10,}{v2 - v1:>+12,}{change:>10}")

    print("-" * 72)
    ratio = f"{v2_total / v1_total:.2f}x" if v1_total else "n/a"
    print(f"{'TOTAL':<30}{v1_total:>10,}{v2_total:>10,}{v2_total - v1_total:>+12,}{ratio:>10}")
    if v1_total:
        print(f"\n  frozen baseline records {V1_FROZEN_CHUNKS:,} chunks; "
              f"v1 recomputed here is {v1_total:,}")
    print(f"  overall growth factor: {v2_total / (v1_total or V1_FROZEN_CHUNKS):.2f}x")

def print_composition(results):
    print("\n" + "=" * 78)
    print("COMPOSITION: table records vs narrative")
    print("=" * 78 + "\n")

    print(f"{'document':<30}{'table':>10}{'narrative':>12}{'table share':>14}")
    print("-" * 66)

    totals = Counter()
    for doc_id, data in results.items():
        kinds = Counter(c["kind"] for c in data["chunks"])
        totals.update(kinds)
        total = sum(kinds.values()) or 1
        print(f"{doc_id:<30}{kinds['table_records']:>10,}{kinds['narrative']:>12,}"
              f"{kinds['table_records'] / total:>13.0%}")

    grand = sum(totals.values()) or 1
    print("-" * 66)
    print(f"{'TOTAL':<30}{totals['table_records']:>10,}{totals['narrative']:>12,}"
          f"{totals['table_records'] / grand:>13.0%}")

    print(f"\n  every table chunk holds records only, with no surrounding prose:")
    print(f"    pure table-record chunks   {totals['table_records']:>8,}")
    print(f"    mixed record+prose chunks  {0:>8,}  (by construction: prose is flushed"
          f" before a table)")

def print_cost(results, v1_counts):
    print("\n" + "=" * 78)
    print("TOKEN AND EMBEDDING COST ESTIMATE")
    print("=" * 78 + "\n")

    print(f"  assumption: {CHARS_PER_TOKEN} chars/token, no tokenizer installed")
    print(f"  price: ${EMBEDDING_PRICE_PER_MILLION}/1M tokens (text-embedding-3-small)\n")

    print(f"{'document':<30}{'chars':>14}{'est tokens':>14}{'est cost':>12}")
    print("-" * 70)

    total_tokens = 0
    for doc_id, data in results.items():
        chars = sum(len(c["text"]) for c in data["chunks"])
        tokens = chars / CHARS_PER_TOKEN
        total_tokens += tokens
        print(f"{doc_id:<30}{chars:>14,}{tokens:>14,.0f}"
              f"{tokens / 1e6 * EMBEDDING_PRICE_PER_MILLION:>12.4f}")

    print("-" * 70)
    cost = total_tokens / 1e6 * EMBEDDING_PRICE_PER_MILLION
    print(f"{'TOTAL':<30}{'':>14}{total_tokens:>14,.0f}{cost:>12.4f}")

    print(f"\n  one full re-embed of the converted corpus: ~${cost:.3f}")
    print(f"  (v1's 4,138 chunks at ~500 chars each were ~{4138 * 500 / CHARS_PER_TOKEN / 1e6:.2f}M "
          f"tokens, ~${4138 * 500 / CHARS_PER_TOKEN / 1e6 * EMBEDDING_PRICE_PER_MILLION:.3f})")

def print_record_distribution(results):
    print("\n" + "=" * 78)
    print("RECORDS PER TABLE")
    print("=" * 78 + "\n")

    everything = []
    for doc_id, data in results.items():
        counts = data["record_counts"]
        everything.extend(counts)
        if not counts:
            continue
        stats = percentiles(counts)
        print(f"{doc_id}  ({len(counts):,} converted tables)")
        print("   " + "  ".join(f"p{p}={v:,}" for p, v in stats.items()))
        print(f"   total records {sum(counts):,}   mean {sum(counts) / len(counts):.1f}")

    if everything:
        stats = percentiles(everything)
        print(f"\ncorpus ({len(everything):,} converted tables, {sum(everything):,} records)")
        print("   " + "  ".join(f"p{p}={v:,}" for p, v in stats.items()))
        heavy = sorted(everything)[-5:]
        print(f"   five largest tables: {heavy}")

    print("\nchunking behaviour:")
    for doc_id, data in results.items():
        stats = data["stats"]
        if not stats:
            continue
        print(f"  {doc_id}")
        print(f"    table chunks            {stats.get('table_chunks', 0):>6,}")
        print(f"    tables spanning chunks  {stats.get('tables_spanning_chunks', 0):>6,}")
        print(f"    oversize records        {stats.get('oversize_records', 0):>6,}"
              f"   (emitted whole, never truncated)")

def print_continuation_example(results, chunk_size):
    print("\n" + "=" * 78)
    print("WORKED EXAMPLE: one table spanning two chunks")
    print("=" * 78)

    for doc_id, data in results.items():
        chunks = data["chunks"]
        by_block = {}
        for i, chunk in enumerate(chunks):
            if chunk["kind"] == "table_records":
                by_block.setdefault(chunk["block_id"], []).append(i)

        # The smallest table that still spans exactly two chunks, so the whole
        # thing is readable, preferring one with a title so the continuation
        # header shows a real table name.
        titles, converted = data["titles"], data["converted"]
        candidates = [(ids, sum(len(chunks[i]["text"]) for i in ids))
                      for block_id, ids in by_block.items()
                      if len(ids) == 2 and titles.get(block_id) and block_id in converted]
        if not candidates:
            candidates = [(ids, sum(len(chunks[i]["text"]) for i in ids))
                          for ids in by_block.values() if len(ids) == 2]
        if not candidates:
            continue

        ids, _ = min(candidates, key=lambda pair: pair[1])
        print(f"\ndocument: {doc_id}")
        print(f"block:    {chunks[ids[0]]['block_id']}")
        print(f"section:  {chunks[ids[0]]['section']}")

        for position, index in enumerate(ids, start=1):
            chunk = chunks[index]
            print(f"\n--- chunk {position} of {len(ids)}  ({len(chunk['text'])} chars, "
                  f"budget {chunk_size}) ---")
            for line in chunk["text"].split("\n"):
                print(f"  {line}")

        print("\n  note: chunk 2 opens with the continuation header, so a reader landing")
        print("  there still knows which table these figures belong to.")
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk size and cost report.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--skip-v1", action="store_true",
                        help="skip rebuilding v1 counts (saves ~90s of parsing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("chunking converted corpus (no API calls)...")
    results = chunk_documents(args.chunk_size, args.overlap)

    v1_counts = {}
    if not args.skip_v1:
        print("\nrebuilding frozen v1 chunk counts for comparison...")
        v1_counts = v1_chunk_counts(args.chunk_size, args.overlap)

        print("checking what build_corpus() produces today...")
        current = build_corpus_count()
        frozen = sum(v1_counts.values())
        print(f"\n  !! build_corpus() now yields {current:,} chunks, not the frozen {frozen:,}.")
        print("     Since step 1b it loads via load_document_ir(), so converted tables")
        print("     already reach it. Nothing is indexed, but it no longer reproduces v1.")

    print_counts(results, v1_counts)
    print_composition(results)
    print_cost(results, v1_counts)
    print_record_distribution(results)
    print_continuation_example(results, args.chunk_size)
