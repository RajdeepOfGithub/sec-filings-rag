import re

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

def structure_aware_chunk_ir(document, chunk_size=500, overlap=50):
    """
    Chunks per section using blocks already attributed by
    sections.attribute_sections(). Replaces the hardcoded offsets in
    structure_aware_chunk(), which is kept alongside for comparison.

    Consecutive blocks sharing a section_id are chunked together, so a chunk
    never spans a section boundary.
    """
    all_chunks = []

    for section_id, section_text in group_blocks_by_section(document.blocks):
        pieces = fixed_size_chunk(section_text, chunk_size, overlap)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "section": section_id,
                "chunk_index_in_section": i,
            })

    return all_chunks

def group_blocks_by_section(blocks):
    """[(section_id, concatenated text)] for each run of consecutive blocks."""
    groups = []

    for block in sorted(blocks, key=lambda b: b.source_order):
        text = block.serialized_text()
        if groups and groups[-1][0] == block.section_id:
            groups[-1][1].append(text)
        else:
            groups.append((block.section_id, [text]))

    return [(section_id, "".join(parts)) for section_id, parts in groups]

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