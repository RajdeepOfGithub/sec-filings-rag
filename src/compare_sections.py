"""
Diffs the hardcoded-offset chunker against dynamic section detection.

Reports boundaries side by side, how much of the document changes section
label, and where blocks land in front_matter or section_unknown.

Usage (from the project root):
    python src/compare_sections.py
    python src/compare_sections.py --doc 10-Q
"""
import argparse

from chunker import fixed_size_chunk, group_blocks_by_section, structure_aware_chunk, structure_aware_chunk_ir
from corpus import make_doc_id
from ir import serialize_document
from loader import load_document, load_document_ir
from sections import (FRONT_MATTER, SECTION_UNKNOWN, attribute_sections_report,
                      detect_sections_report, section_id_at)

DOCUMENTS = {
    "10-K": ("data/raw/jpmc/jpmc_10k_2025.htm", "sec_filing", "JPMC", "10-K", "FY2025"),
    "10-Q": ("data/raw/jpmc/jpmc_10q_q2_2026.htm", "sec_filing", "JPMC", "10-Q", "Q2-2026"),
}

# The hardcoded regions being replaced, copied from structure_aware_chunk().
OLD_BOUNDARIES = [
    ("Item 1 - Business", 6857, 45993),
    ("Item 1A - Risk Factors", 45993, 158511),
    ("Part II - pointer block", 158511, 167019),
    ("Body - MD&A and financials", 167019, None),
]


def old_label_at(offset, text_length):
    for name, start, end in OLD_BOUNDARIES:
        if start <= offset < (end if end is not None else text_length):
            return name
    return "(dropped by v1)"

def print_boundaries(boundaries, text_length):
    print("\n--- BOUNDARIES: hardcoded (v1) vs detected (v2) ---\n")
    print(f"{'v1 hardcoded region':<34} {'start':>9} {'chars':>9}   |   "
          f"{'v2 detected section':<34} {'start':>9} {'chars':>9}")
    print("-" * 78 + "+" + "-" * 60)

    old_rows = [(name, start, (end if end is not None else text_length) - start)
                for name, start, end in OLD_BOUNDARIES]
    new_rows = [(b.section_id, b.start, b.end - b.start) for b in boundaries]

    for i in range(max(len(old_rows), len(new_rows))):
        left = old_rows[i] if i < len(old_rows) else ("", "", "")
        right = new_rows[i] if i < len(new_rows) else ("", "", "")
        left_text = f"{left[0]:<34} {left[1]:>9} {left[2]:>9}" if left[0] else " " * 54
        right_text = f"{right[0]:<34} {right[1]:>9} {right[2]:>9}" if right[0] else ""
        print(f"{left_text}   |   {right_text}")

    print(f"\nv1 found {len(old_rows)} regions, v2 detected {len(new_rows)} sections")

def print_label_agreement(boundaries, text_length):
    print("\n--- CHARACTER-LEVEL LABEL AGREEMENT ---\n")
    changed = 0
    per_old = {}

    for offset in range(0, text_length, 100):  # sampled every 100 chars
        old = old_label_at(offset, text_length)
        new = section_id_at(offset, boundaries)
        if old != new:
            changed += 1
        per_old.setdefault(old, {}).setdefault(new, 0)
        per_old[old][new] += 1

    sampled = len(range(0, text_length, 100))
    print(f"sampled every 100 chars: {changed:,}/{sampled:,} positions change label "
          f"({changed / sampled:.1%} of the document)\n")

    for old_name, news in per_old.items():
        print(f"v1 '{old_name}' ->")
        for new_name, count in sorted(news.items(), key=lambda kv: -kv[1]):
            print(f"      {count * 100:>9,} chars  {new_name}")

def print_chunk_comparison(old_chunks, new_chunks):
    print("\n--- CHUNKS ---\n")
    print(f"v1 produced {len(old_chunks):,} chunks, v2 produced {len(new_chunks):,}")

    def counts(chunks):
        out = {}
        for chunk in chunks:
            out[chunk["section"]] = out.get(chunk["section"], 0) + 1
        return out

    old_counts, new_counts = counts(old_chunks), counts(new_chunks)
    print(f"\n{'v1 section':<34} {'chunks':>8}   |   {'v2 section':<34} {'chunks':>8}")
    print("-" * 45 + "+" + "-" * 47)
    old_rows = sorted(old_counts.items(), key=lambda kv: -kv[1])
    new_rows = sorted(new_counts.items(), key=lambda kv: -kv[1])
    for i in range(max(len(old_rows), len(new_rows))):
        left = f"{old_rows[i][0]:<34} {old_rows[i][1]:>8}" if i < len(old_rows) else " " * 43
        right = f"{new_rows[i][0]:<34} {new_rows[i][1]:>8}" if i < len(new_rows) else ""
        print(f"{left}   |   {right}")

def print_chunk_label_changes(document, boundaries, chunk_size, overlap):
    """For each v2 chunk, what v1 would have called the same text."""
    print("\n--- WHERE v2 CHUNKS DISAGREE WITH v1 ---\n")
    text_length = len(serialize_document(document))

    offset = 0
    changed = 0
    total = 0
    pairs = {}

    for section_id, section_text in group_blocks_by_section(document.blocks):
        start_of_section = offset
        for i, piece in enumerate(fixed_size_chunk(section_text, chunk_size, overlap)):
            chunk_start = start_of_section + i * (chunk_size - overlap)
            old = old_label_at(chunk_start, text_length)
            total += 1
            if old != section_id:
                changed += 1
            pairs[(old, section_id)] = pairs.get((old, section_id), 0) + 1
        offset += len(section_text)

    print(f"{changed:,}/{total:,} v2 chunks carry a different section label than v1 "
          f"would have given the same text ({changed / total:.1%})\n")
    for (old, new), count in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]:
        marker = "  " if old == new else "->"
        print(f"  {count:>6} chunks  {marker} v1 '{old}'  =>  v2 '{new}'")

def print_block_placement(document):
    print("\n--- BLOCK PLACEMENT ---\n")
    totals = {}
    for block in document.blocks:
        length = len(block.serialized_text())
        entry = totals.setdefault(block.section_id, [0, 0])
        entry[0] += 1
        entry[1] += length

    for section_id, (count, chars) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        flag = ""
        if section_id == SECTION_UNKNOWN:
            flag = "   <- unlabelled on purpose"
        elif section_id == FRONT_MATTER:
            flag = "   <- silently dropped by v1"
        print(f"  {count:>4} block(s)  {chars:>10,} chars  {section_id}{flag}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare hardcoded vs detected sections.")
    parser.add_argument("--doc", default="10-K", choices=sorted(DOCUMENTS))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path, doc_type, company, form, period = DOCUMENTS[args.doc]
    doc_id = make_doc_id(company, form, period)

    print(f"document: {path}\ndoc_id:   {doc_id}")

    text = load_document(path, doc_type)
    document = load_document_ir(path, doc_type, doc_id)

    print("\n--- DIAGNOSTICS ---\n")
    attributed, diagnostics = attribute_sections_report(document, strict=False)
    if not diagnostics:
        print("(none)")
    for diagnostic in diagnostics:
        print(f"  {diagnostic}")

    boundaries, _ = detect_sections_report(document.blocks, strict=False)

    print_boundaries(boundaries, len(text))
    print_label_agreement(boundaries, len(text))

    old_chunks = structure_aware_chunk(text, args.chunk_size, args.overlap)
    new_chunks = structure_aware_chunk_ir(attributed, args.chunk_size, args.overlap)
    print_chunk_comparison(old_chunks, new_chunks)
    print_chunk_label_changes(attributed, boundaries, args.chunk_size, args.overlap)
    print_block_placement(attributed)
