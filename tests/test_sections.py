"""
Tests for dynamic section detection.

The invariants that matter:
  - attributing sections does not change serialized text (splitting is exact)
  - detection depends on content, not on absolute offsets
  - a table of contents is discarded structurally
  - out-of-sequence headings are an error, not a silent relabel

Run from the project root:
    python tests/test_sections.py
"""
import sys

sys.path.insert(0, "src")

from chunker import structure_aware_chunk_ir
from ir import Document, TextBlock, assign_block_ids, serialize_document
from loader import load_document, load_document_ir
from sections import (FRONT_MATTER, SECTION_UNKNOWN, SectionDetectionError,
                      attribute_sections_report, detect_sections_report)

TEN_K = ("data/raw/jpmc/jpmc_10k_2025.htm", "sec_filing", "JPMC_10-K_FY2025")

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)

def make_document(text, doc_id="TEST_10-K_FY2025"):
    blocks = assign_block_ids([TextBlock(doc_id=doc_id, source_order=0, text=text)])
    return Document(doc_id=doc_id, source_path="synthetic", doc_type="sec_filing", blocks=blocks)

def synthetic_filing(pad=""):
    """A filing with a TOC cluster, then real sections with prose between."""
    toc = ("TABLE OF CONTENTS Item 1. Business 3 Item 1A. Risk Factors 9 "
           "Item 2. Properties 22 Item 3. Legal Proceedings 24 ")
    body = ("Item 1. Business " + "We operate a bank. " * 60 +
            "Item 1A. Risk Factors " + "Rates could move against us. " * 60 +
            "Item 2. Properties " + "We lease offices. " * 60 +
            "Item 3. Legal Proceedings " + "Refer to Note 22. " * 60)
    return pad + toc + body


def test_attribution_preserves_serialization():
    print("\nattributing sections does not change the serialized text")

    path, doc_type, doc_id = TEN_K
    document = load_document_ir(path, doc_type, doc_id)
    loaded_text = serialize_document(document)
    attributed, _ = attribute_sections_report(document, strict=False)

    # Since step 4 the loader's own output differs from load_document() for
    # filings with tables. What attribution must not do is change the text it
    # was given, whatever that text is.
    check("attribution does not change the loaded text",
          serialize_document(attributed) == loaded_text, f"{len(loaded_text):,} chars")
    check("blocks were split at boundaries", len(attributed.blocks) > len(document.blocks),
          f"{len(document.blocks)} -> {len(attributed.blocks)} blocks")
    check("every block has a section_id", all(b.section_id for b in attributed.blocks))
    check("block ids stay content-derived", all(b.block_id.startswith(doc_id + "/") for b in attributed.blocks))


def test_detection_is_offset_independent():
    """The whole point of step 2: shifting the text must not break labels."""
    print("\ndetection follows content, not absolute offsets")

    plain, _ = detect_sections_report(make_document(synthetic_filing()).blocks, strict=False)
    padded_doc = make_document(synthetic_filing(pad="COVER PAGE. " * 500))
    padded, _ = detect_sections_report(padded_doc.blocks, strict=False)

    check("same sections found after padding",
          [b.section_id for b in plain] == [b.section_id for b in padded],
          f"{[b.item_number for b in plain]}")
    shift = padded[0].start - plain[0].start
    check("every boundary shifts by the same amount",
          all(p.start - q.start == shift for p, q in zip(padded, plain)), f"shift={shift}")
    check("section lengths identical",
          [b.end - b.start for b in plain[:-1]] == [b.end - b.start for b in padded[:-1]])


def test_toc_is_discarded_structurally():
    print("\ntable of contents is discarded, body headings kept")

    document = make_document(synthetic_filing())
    boundaries, diagnostics = detect_sections_report(document.blocks, strict=False)
    codes = [d.code for d in diagnostics]

    check("one boundary per item, not two", len(boundaries) == 4, f"{len(boundaries)} boundaries")
    check("toc_discarded warning raised", "toc_discarded" in codes, str(codes))
    check("no duplicate items survive", len({b.item_number for b in boundaries}) == len(boundaries))
    check("kept boundaries are the body ones, not the TOC",
          boundaries[0].start > 100, f"first boundary at {boundaries[0].start}")


def test_non_monotonic_is_an_error():
    print("\nout-of-sequence headings are an error")

    text = ("Item 1. Business " + "Prose. " * 100 +
            "Item 7. Management's Discussion " + "Prose. " * 100 +
            "Item 1A. Risk Factors " + "Prose. " * 100)
    document = make_document(text)

    try:
        detect_sections_report(document.blocks, strict=True)
        check("strict=True raises SectionDetectionError", False, "no error raised")
    except SectionDetectionError as exc:
        check("strict=True raises SectionDetectionError", True, str(exc)[:60])

    boundaries, diagnostics = detect_sections_report(document.blocks, strict=False)
    check("strict=False reports an error diagnostic",
          any(d.severity == "error" for d in diagnostics),
          next((d.code for d in diagnostics if d.severity == "error"), ""))
    check("surviving sequence is monotonic",
          [b.item_number for b in boundaries] == sorted({b.item_number for b in boundaries},
                                                        key=lambda n: (int(n[0]), n[1:])),
          str([b.item_number for b in boundaries]))


def test_front_matter_and_unknown():
    print("\nfront matter and unbounded tail are explicit")

    document = make_document(synthetic_filing())
    attributed, _ = attribute_sections_report(document, strict=False)
    labels = [b.section_id for b in attributed.blocks]

    check("text before the first heading is front_matter", labels[0] == FRONT_MATTER, labels[0])
    check("no block is left without a label", all(labels))

    path, doc_type, doc_id = TEN_K
    ten_k, diagnostics = attribute_sections_report(load_document_ir(path, doc_type, doc_id), strict=False)
    ten_k_labels = {b.section_id for b in ten_k.blocks}
    check("10-K tail is section_unknown, not a confident label", SECTION_UNKNOWN in ten_k_labels)
    check("unbounded_tail warning raised", any(d.code == "unbounded_tail" for d in diagnostics))
    check("10-K front matter is labelled rather than dropped", FRONT_MATTER in ten_k_labels)


def test_chunker_carries_section_and_covers_document():
    print("\nstructure_aware_chunk_ir carries section_id and loses no text")

    path, doc_type, doc_id = TEN_K
    document = load_document_ir(path, doc_type, doc_id)
    expected_chars = len(serialize_document(document))
    attributed, _ = attribute_sections_report(document, strict=False)
    chunks = structure_aware_chunk_ir(attributed, chunk_size=500, overlap=50)

    check("every chunk has a section label", all(c["section"] for c in chunks))
    check("chunk_index_in_section restarts per section",
          min(c["chunk_index_in_section"] for c in chunks) == 0)

    covered = sum(len(block.serialized_text()) for block in attributed.blocks)
    check("chunks cover the whole document including front matter",
          covered == expected_chars, f"{covered:,} chars")


def test_ten_q_does_not_crash_without_ten_k_items():
    print("\na document without 10-K item headings degrades safely")

    document = make_document("Consolidated statements of income. Net revenue was 57,347. " * 50)
    boundaries, diagnostics = detect_sections_report(document.blocks, strict=False)
    attributed, _ = attribute_sections_report(document, strict=False)

    check("no boundaries invented", boundaries == [], f"{len(boundaries)} found")
    check("no_candidates warning raised", any(d.code == "no_candidates" for d in diagnostics))
    check("blocks fall back to section_unknown",
          all(b.section_id == SECTION_UNKNOWN for b in attributed.blocks))


if __name__ == "__main__":
    test_attribution_preserves_serialization()
    test_detection_is_offset_independent()
    test_toc_is_discarded_structurally()
    test_non_monotonic_is_an_error()
    test_front_matter_and_unknown()
    test_chunker_carries_section_and_covers_document()
    test_ten_q_does_not_crash_without_ten_k_items()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("All checks passed.")
