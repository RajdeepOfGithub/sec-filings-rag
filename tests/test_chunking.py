"""
Tests for record-aware chunking.

The invariant: a table record is atomic. Splitting one recreates the orphaned
value problem the converter exists to remove.

Run from the project root:
    python tests/test_chunking.py
"""
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from chunker import (WHOLE_TABLE_BUDGET_MULTIPLE, continuation_header,
                     structure_aware_chunk_ir)
from ir import Document, TableBlock, TextBlock, assign_block_ids
from loader import load_document_ir
from sections import attribute_sections_report

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)

def make_document(blocks):
    for block in blocks:
        block.section_id = "Item 7 - Management's Discussion and Analysis"
    return Document(doc_id="TEST_10-K_FY2025", source_path="synthetic",
                    doc_type="sec_filing", blocks=assign_block_ids(blocks))

def records(count, width=60):
    return [f"Capital | Metric row {i:02d} | Year ended December 31, 2025 | {i}.7 | millions".ljust(width)[:width]
            for i in range(count)]


def test_records_are_never_split():
    print("\nrecords stay whole across chunk boundaries")

    table_records = records(60)
    document = make_document([
        TableBlock(doc_id="TEST_10-K_FY2025", source_order=0, raw_html="<table></table>",
                   records=table_records, table_title="Capital", unit="millions"),
    ])
    chunks = structure_aware_chunk_ir(document, chunk_size=500, overlap=50)
    table_chunks = [c for c in chunks if c["kind"] == "table_records"]

    emitted = []
    for chunk in table_chunks:
        for line in chunk["text"].split("\n"):
            if "(continued)" not in line:
                emitted.append(line)

    check("more than one chunk produced", len(table_chunks) > 1, f"{len(table_chunks)} chunks")
    check("every emitted line is a complete record", all(line in table_records for line in emitted))
    check("every record appears exactly once",
          sorted(emitted) == sorted(table_records), f"{len(emitted)} of {len(table_records)}")
    check("no chunk exceeds the budget", all(len(c["text"]) <= 500 for c in table_chunks),
          f"max {max(len(c['text']) for c in table_chunks)}")


def test_continuation_header_on_later_chunks_only():
    print("\ncontinuation header marks every chunk after the first")

    document = make_document([
        TableBlock(doc_id="TEST_10-K_FY2025", source_order=0, raw_html="<table></table>",
                   records=records(60), table_title="Capital", unit="millions"),
    ])
    chunks = [c for c in structure_aware_chunk_ir(document, 500, 50) if c["kind"] == "table_records"]

    check("first chunk has no continuation header", "(continued)" not in chunks[0]["text"])
    check("later chunks all start with it",
          all(c["text"].startswith("Capital (continued)") for c in chunks[1:]),
          chunks[1]["text"].split("\n")[0])
    check("header names the unit", "values in millions" in chunks[1]["text"])

    untitled = TableBlock(doc_id="d", source_order=0, raw_html="<table></table>",
                          records=records(4), table_title="", unit="billions")
    check("untitled table says what it knows instead of naming it 'table'",
          continuation_header(untitled).startswith("(continued from the preceding table)"),
          continuation_header(untitled))


def test_small_table_is_kept_whole():
    """q01: a five-row table split in two separated share counts from dollars."""
    print("\na table under the whole-table budget is never split")

    small = records(12)   # ~730 chars: over one chunk, under 3 x 500
    document = make_document([
        TableBlock(doc_id="TEST_10-K_FY2025", source_order=0, raw_html="<table></table>",
                   records=small, table_title="Capital", unit="millions"),
    ])
    stats = {}
    chunks = [c for c in structure_aware_chunk_ir(document, 500, 50, stats)
              if c["kind"] == "table_records"]

    check("emitted as a single chunk", len(chunks) == 1, f"{len(chunks)} chunks")
    check("holds every record", all(r in chunks[0]["text"] for r in small))
    check("counted as kept whole", stats.get("tables_kept_whole") == 1, str(stats))
    check("exceeds the plain chunk budget, as intended",
          len(chunks[0]["text"]) > 500, f"{len(chunks[0]['text'])} chars")
    check("stays within the whole-table budget",
          len(chunks[0]["text"]) <= WHOLE_TABLE_BUDGET_MULTIPLE * 500)


def test_oversize_record_is_emitted_whole():
    print("\na record longer than the budget is emitted whole, not truncated")

    long_record = "Capital | " + ("A very long row label " * 40) + "| 2025 | 1.0 | millions"
    # The table must exceed the whole-table budget, or it is kept whole and
    # the oversize-record path never runs.
    document = make_document([
        TableBlock(doc_id="TEST_10-K_FY2025", source_order=0, raw_html="<table></table>",
                   records=[records(1)[0], long_record] + records(20),
                   table_title="Capital", unit="millions"),
    ])
    stats = {}
    chunks = structure_aware_chunk_ir(document, 500, 50, stats)
    texts = [c["text"] for c in chunks]

    check("oversize record survives intact", any(long_record in t for t in texts),
          f"{len(long_record)} chars")
    check("counted in stats", stats.get("oversize_records", 0) >= 1, str(stats))
    check("it did not swallow the other records",
          any(records(1)[0] in t for t in texts))


def test_no_overlap_inside_tables_but_prose_keeps_it():
    print("\noverlap applies to prose, not to records")

    table_records = records(60)
    document = make_document([
        TableBlock(doc_id="TEST_10-K_FY2025", source_order=0, raw_html="<table></table>",
                   records=table_records, table_title="Capital", unit="millions"),
        TextBlock(doc_id="TEST_10-K_FY2025", source_order=1, text="word " * 400),
    ])
    chunks = structure_aware_chunk_ir(document, 500, 50)
    table_chunks = [c for c in chunks if c["kind"] == "table_records"]
    prose_chunks = [c for c in chunks if c["kind"] == "narrative"]

    def record_set(chunk):
        return {l for l in chunk["text"].split("\n") if "(continued)" not in l}

    overlaps = [record_set(a) & record_set(b)
                for a, b in zip(table_chunks, table_chunks[1:])]
    check("adjacent table chunks share no record", all(not o for o in overlaps))
    check("prose still chunked with overlap", len(prose_chunks) > 1,
          f"{len(prose_chunks)} prose chunks")
    check("prose chunks overlap by the configured amount",
          prose_chunks[0]["text"][-50:] == prose_chunks[1]["text"][:50])


def test_real_document_has_no_split_records():
    print("\nreal 10-Q: no record is split anywhere")

    document = load_document_ir("data/raw/jpmc/jpmc_10q_q2_2026.htm", "sec_filing",
                                "JPMC_10-Q_Q2-2026")
    attributed, _ = attribute_sections_report(document, strict=False)
    stats = {}
    chunks = structure_aware_chunk_ir(attributed, 500, 50, stats)

    known = set()
    for block in attributed.blocks:
        known.update(getattr(block, "records", []) or [])

    partial = [line for chunk in chunks if chunk["kind"] == "table_records"
               for line in chunk["text"].split("\n")
               if "(continued" not in line and line not in known]

    check("no partial record lines", not partial, f"{len(partial)} partial lines")
    check("table and narrative chunks both present",
          any(c["kind"] == "table_records" for c in chunks)
          and any(c["kind"] == "narrative" for c in chunks),
          f"{len(chunks):,} chunks total")
    check("chunks carry the block id of their table",
          all(c["block_id"] for c in chunks if c["kind"] == "table_records"))


if __name__ == "__main__":
    test_records_are_never_split()
    test_small_table_is_kept_whole()
    test_continuation_header_on_later_chunks_only()
    test_oversize_record_is_emitted_whole()
    test_no_overlap_inside_tables_but_prose_keeps_it()
    test_real_document_has_no_split_records()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("All checks passed.")
