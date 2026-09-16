from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pypdf import PdfReader
import warnings

from ir import Document, TableBlock, TextBlock, assign_block_ids

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def _filing_soup(filepath):
    """One parsing path for filings, shared by load_html() and the IR loader."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "lxml")

    for hidden in soup.find_all("ix:header"):
        hidden.decompose()

    return soup


def load_html(filepath):            #Fn for SEC filing  (10K and 10Q)
    text = _filing_soup(filepath).get_text(separator=" ", strip=True)
    return text


def load_earnings_transcript(filepath):      #Fn for Earning transcript
    with open(filepath, "r", encoding="utf-8") as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "lxml")

    marker = soup.find("h2", id="full-conference-call-transcript")
    if marker is None:
        raise ValueError("Could not find transcript start marker — page structure may have changed")

    transcript_parts = []
    for sibling in marker.find_next_siblings():
        transcript_parts.append(sibling.get_text(separator=" ", strip=True))

    text = " ".join(transcript_parts)
    return text

def load_pdf(filepath):
    reader = PdfReader(filepath)

    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())

    text = " ".join(text_parts)
    return text

def load_document(filepath, doc_type):
    """
    doc_type must be one of: "sec_filing", "earnings_transcript", "pdf"
    """
    if doc_type == "sec_filing":
        return load_html(filepath)
    elif doc_type == "earnings_transcript":
        return load_earnings_transcript(filepath)
    elif doc_type == "pdf":
        return load_pdf(filepath)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type}")


def sec_filing_blocks(filepath, doc_id, convert_tables, stats):
    """
    Splits a filing into TextBlocks and TableBlocks in source order.

    Table spans are located by character offset in the same flattened text the
    old loader produced, so the text between tables is carried through
    verbatim. With convert_tables=False every TableBlock falls back to
    get_text() and serialization is unchanged - which is what the equivalence
    test checks.
    """
    from table_convert import convert_table
    from table_profile import profile_table, table_text_offsets

    soup = _filing_soup(filepath)
    full_text = soup.get_text(separator=" ", strip=True)
    offsets = table_text_offsets(soup)

    spans = []
    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            continue  # nested tables travel with their parent

        flattened = table.get_text(separator=" ", strip=True)
        start = offsets.get(id(table))
        if start is None or not flattened:
            continue
        if full_text[start:start + len(flattened)] != flattened:
            stats["table_offset_mismatch"] = stats.get("table_offset_mismatch", 0) + 1
            continue
        spans.append((start, start + len(flattened), table))

    spans.sort()
    blocks = []
    cursor = 0

    for start, end, table in spans:
        if start < cursor:
            stats["table_overlap_skipped"] = stats.get("table_overlap_skipped", 0) + 1
            continue

        if start > cursor:
            blocks.append(TextBlock(doc_id=doc_id, source_order=len(blocks),
                                    text=full_text[cursor:start]))

        block = TableBlock(doc_id=doc_id, source_order=len(blocks), raw_html=str(table))
        if convert_tables:
            profile = profile_table(table, doc_id, "", len(blocks))
            conversion = convert_table(table, profile)
            block.records = conversion.records
            block.conversion_status = conversion.status
            block.reason_code = conversion.reason
            block.table_title = conversion.title
            block.unit = conversion.unit
            stats[conversion.status] = stats.get(conversion.status, 0) + 1
            stats[conversion.reason] = stats.get(conversion.reason, 0) + 1

        blocks.append(block)
        cursor = end

    if cursor < len(full_text):
        blocks.append(TextBlock(doc_id=doc_id, source_order=len(blocks), text=full_text[cursor:]))

    return blocks

def load_document_ir(filepath, doc_type, doc_id, convert_tables=True, stats=None):
    """
    IR version of load_document(). SEC filings become TextBlocks and
    TableBlocks in source order; other document types stay a single TextBlock.

    doc_id is required and must match the corpus naming (JPMC_10-Q_Q2-2026),
    so block ids and chunk ids never drift into two schemes.

    Serialization deliberately no longer matches load_document() for filings
    with converted tables. It still matches exactly for documents without
    tables, and for filings loaded with convert_tables=False.
    """
    stats = {} if stats is None else stats

    if doc_type == "sec_filing":
        blocks = sec_filing_blocks(filepath, doc_id, convert_tables, stats)
    else:
        blocks = [TextBlock(doc_id=doc_id, source_order=0,
                            text=load_document(filepath, doc_type))]

    return Document(
        doc_id=doc_id,
        source_path=filepath,
        doc_type=doc_type,
        blocks=assign_block_ids(blocks),
    )


if __name__ == "__main__":
    documents = [
        {"filepath": "data/raw/jpmc/jpmc_10k_2025.htm", "doc_type": "sec_filing"},
        {"filepath": "data/raw/jpmc/jpmc_10q_q2_2026.htm", "doc_type": "sec_filing"},
        {"filepath": "data/raw/jpmc/jpmc_earnings_call_q2_2026.htm", "doc_type": "earnings_transcript"},
    ]

    for doc in documents:
        text = load_document(doc["filepath"], doc["doc_type"])
        print(f"\n=== {doc['filepath']} ({doc['doc_type']}) ===")
        print(f"Total characters extracted: {len(text)}")
        print(text[:300])