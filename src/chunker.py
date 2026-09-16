import re

from ir import TableBlock


def find_section_markers(text):
    pattern = r"Item\s+\d+[A-Z]?\."
    matches = list(re.finditer(pattern, text))

    print(f"Found {len(matches)} potential section markers\n")

    prev_pos = 0
    for m in matches:
        gap = m.start() - prev_pos
        print(f"  '{m.group()}' at position {m.start()}   (gap from previous: {gap})")
        prev_pos = m.start()

    return matches

def find_real_section_headers(text):
    known_titles = {
        "1": "Business",
        "1A": "Risk Factors",
        "1B": "Unresolved Staff Comments",
        "1C": "Cybersecurity",
        "2": "Properties",
        "3": "Legal Proceedings",
        "4": "Mine Safety Disclosures",
        "5": "Market for Registrant",
        "6": "Reserved",
        "7": "Management",
        "7A": "Quantitative and Qualitative",
        "8": "Financial Statements",
        "9": "Changes in and Disagreements",
        "9A": "Controls and Procedures",
        "9B": "Other Information",
        "9C": "Disclosure Regarding Foreign",
    }

    matches_found = []
    for item_num, title_start in known_titles.items():
        pattern = rf"Item\s+{item_num}\.?\s*{re.escape(title_start)}"
        all_matches = list(re.finditer(pattern, text))

        if all_matches:
            positions = [m.start() for m in all_matches]
            chosen = all_matches[-1].start()
            matches_found.append((item_num, chosen))
            print(f"Item {item_num}: {len(all_matches)} match(es) at {positions} -> using {chosen}")
        else:
            print(f"Item {item_num}: NOT FOUND")

    return matches_found

def structure_aware_chunk(text, chunk_size=500, overlap=50):
    boundaries = [
        ("Item 1 - Business", 6857, 45993),
        ("Item 1A - Risk Factors", 45993, 158511),
        ("Part II - pointer block", 158511, 167019),
        ("Body - MD&A and financials", 167019, len(text)),
    ]

    all_chunks = []
    for section_name, start, end in boundaries:
        section_text = text[start:end]
        pieces = fixed_size_chunk(section_text, chunk_size, overlap)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "section": section_name,
                "chunk_index_in_section": i,
            })

    return all_chunks

def structure_aware_chunk_ir(document, chunk_size=500, overlap=50, stats=None):
    """
    Chunks per section using blocks already attributed by
    sections.attribute_sections(). Replaces the hardcoded offsets in
    structure_aware_chunk(), which is kept alongside for comparison.

    Table records are atomic. Each one already carries its table title, row
    label, column header, value and unit, so splitting one in half would
    recreate the orphaned-value problem the converter exists to remove.
    Narrative text keeps the fixed-size behaviour, overlap included; overlap
    within a table is pointless because each record is self-contained.
    """
    stats = {} if stats is None else stats
    all_chunks = []

    for section_id, group in group_blocks_by_section(document.blocks):
        index = 0
        prose = []

        def flush_prose():
            """Consecutive prose is chunked as one run, so the text between
            two tables does not become its own tiny chunk."""
            nonlocal index
            if not prose:
                return
            for piece in fixed_size_chunk("".join(prose), chunk_size, overlap):
                all_chunks.append({
                    "text": piece, "section": section_id,
                    "chunk_index_in_section": index,
                    "kind": "narrative", "block_id": "",
                })
                index += 1
            prose.clear()

        for text, block in group:
            if isinstance(block, TableBlock) and block.records:
                flush_prose()
                for piece in chunk_table_records(block.records, chunk_size,
                                                 continuation_header(block), stats):
                    all_chunks.append({
                        "text": piece, "section": section_id,
                        "chunk_index_in_section": index,
                        "kind": "table_records", "block_id": block.block_id,
                    })
                    index += 1
            else:
                prose.append(text)

        flush_prose()

    return all_chunks

def continuation_header(block):
    """
    What a reader landing mid-table needs: which table this is, and the scale
    its figures are stated in.
    """
    # With no title, say what is actually known rather than naming it "table".
    header = (f"{block.table_title} (continued)" if block.table_title
              else "(continued from the preceding table)")
    if block.unit:
        header += f" | values in {block.unit} unless the record states otherwise"
    return header

def chunk_table_records(records, chunk_size, header, stats):
    """
    Packs whole records into chunks. Continuation chunks start with the
    header. A record longer than the budget becomes its own chunk rather than
    being truncated.
    """
    chunks = []
    index = 0

    while index < len(records):
        current = []
        length = 0

        if chunks and header:
            current.append(header)
            length = len(header)

        while index < len(records):
            record = records[index]
            addition = len(record) + (1 if current else 0)

            if current and length + addition > chunk_size:
                break
            if not current and len(record) > chunk_size:
                stats["oversize_records"] = stats.get("oversize_records", 0) + 1

            current.append(record)
            length += addition
            index += 1

        if len(current) <= (1 if (chunks and header) else 0):
            # Only the header fit: emit the next record on its own instead of
            # looping forever on a record that cannot share a chunk.
            current.append(records[index])
            stats["oversize_records"] = stats.get("oversize_records", 0) + 1
            index += 1

        chunks.append("\n".join(current))

    stats["table_chunks"] = stats.get("table_chunks", 0) + len(chunks)
    if len(chunks) > 1:
        stats["tables_spanning_chunks"] = stats.get("tables_spanning_chunks", 0) + 1

    return chunks

def group_blocks_by_section(blocks):
    """
    [(section_id, [(text, block), ...])] for each run of consecutive blocks
    sharing a section. Blocks stay separate so tables can be chunked by
    record while prose is chunked by size.
    """
    groups = []

    for block in sorted(blocks, key=lambda b: b.source_order):
        entry = (block.serialized_text(), block)
        if groups and groups[-1][0] == block.section_id:
            groups[-1][1].append(entry)
        else:
            groups.append((block.section_id, [entry]))

    return groups

def fixed_size_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


if __name__ == "__main__":
    from loader import load_document

    text = load_document("data/raw/jpmc/jpmc_10k_2025.htm", doc_type="sec_filing")
    chunks = structure_aware_chunk(text)

    print(f"Total chunks: {len(chunks)}\n")

    from collections import Counter
    for section, count in Counter(c["section"] for c in chunks).items():
        print(f"{section}: {count} chunks")