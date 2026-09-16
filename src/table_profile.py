"""
Table census: profile and classify every <table> in a document.

Profiling only. Nothing here converts a table or changes serialization; the
point is to learn what shapes exist before writing a converter.

Classification is deliberately conservative. Calling a layout table "data"
invents header-to-value relationships that were never in the document, which
is the same class of error as a wrong section label: confident and wrong.
When signals disagree, a table lands in "ambiguous" rather than being forced
into a bucket.

Thresholds are generic properties of tables (how many rows, how numeric, how
much prose per cell), not values tuned to any filer.
"""
import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup, CData, NavigableString

from ir import make_block_id, normalize
from sections import detect_sections_report, section_id_at

# Mirrors the document list in corpus.build_corpus(); kept here so profiling
# can run without executing the corpus builder.
DOCUMENTS = {
    "JPMC_10-K_FY2025": ("data/raw/jpmc/jpmc_10k_2025.htm", "sec_filing"),
    "JPMC_10-Q_Q2-2026": ("data/raw/jpmc/jpmc_10q_q2_2026.htm", "sec_filing"),
    "JPMC_earnings_call_Q2-2026": ("data/raw/jpmc/jpmc_earnings_call_q2_2026.htm", "earnings_transcript"),
}

NUMERIC_CELL = re.compile(r"^[\s$€£¥(){}\[\]\-–—%0-9.,:]*[0-9][\s$€£¥(){}\[\]\-–—%0-9.,:]*$")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
PAREN_NEGATIVE = re.compile(r"^\(\s*[\d.,]+\s*\)$")
CURRENCY = re.compile(r"[$€£¥]")
PERCENT = re.compile(r"%")
# "2026", "2Q26", "Q2 2026", "June 30, 2026", "FY2025", "1H26"
PERIOD_LIKE = re.compile(
    r"(\b(?:19|20)\d{2}\b)|(\b[1-4]\s*[QH]\s*\d{2}\b)|(\b[QH]\s*[1-4]\b)|(\bFY\s*\d{2,4}\b)",
    re.IGNORECASE,
)

# Generic shape thresholds.
MIN_DATA_ROWS = 3               # fewer rows than this is not a repeated shape
MIN_DATA_COLS = 2
CONSISTENCY_STRONG = 0.8        # share of rows sharing the modal width
NUMERIC_STRONG = 0.30
NUMERIC_WEAK = 0.15
MIN_NUMERIC_ROWS = 2            # a shape has to repeat to be a shape
MOSTLY_EMPTY = 0.90             # spacer cells are pervasive in SEC HTML, so
                                # only near-empty scaffolding counts as layout
PROSE_CELL_CHARS = 80           # average cell length that reads as prose
LONG_CELL_CHARS = 400           # a single cell holding a paragraph
IXBRL_FACTS_PER_CELL = 0.20
HEADER_REGION_ROWS = 3          # rows searched for header/superheader patterns


@dataclass
class TableProfile:
    table_id: str
    doc_id: str
    section_id: str
    source_order: int

    rows: int
    max_cols: int
    min_cols: int
    col_count_consistency: float

    td_count: int
    th_count: int

    numeric_ratio: float      # over non-empty cells; SEC tables are full of spacers
    empty_cell_ratio: float
    numeric_row_count: int    # rows holding 2+ numbers: the "repeated shape" signal
    year_count: int
    currency_count: int
    percent_count: int
    paren_negative_count: int

    rowspan_count: int
    colspan_count: int
    nested_table_count: int

    ix_nonfraction_count: int
    ix_nonnumeric_count: int

    avg_cell_chars: float
    max_cell_chars: int
    total_chars: int

    has_header_row_candidate: bool
    superheader_suspected: bool

    text_preview: str = ""    # first 300 chars of today's get_text() output
    primary_bucket: str = ""
    buckets: list = field(default_factory=list)


def cell_text(cell):
    return " ".join(cell.get_text(separator=" ", strip=True).split())

def span_of(cell, attribute):
    try:
        value = int(cell.get(attribute, 1))
    except (TypeError, ValueError):
        return 1
    return max(value, 1)

def is_numeric_cell(text):
    return bool(text) and bool(NUMERIC_CELL.match(text)) and any(ch.isdigit() for ch in text)

def is_period_cell(text):
    return bool(PERIOD_LIKE.search(text))

def direct_rows(table):
    """Rows belonging to this table, not to a table nested inside it."""
    return [row for row in table.find_all("tr") if row.find_parent("table") is table]

def direct_cells(row):
    return [cell for cell in row.find_all(["td", "th"]) if cell.find_parent("tr") is row]

def count_ix_facts(table, suffix):
    return len(table.find_all(lambda tag: tag.name and tag.name.lower().endswith(suffix)))


def profile_table(table, doc_id, section_id, source_order):
    rows = direct_rows(table)
    row_cells = [direct_cells(row) for row in rows]
    cells = [cell for row in row_cells for cell in row]
    texts = [cell_text(cell) for cell in cells]

    widths = [sum(span_of(cell, "colspan") for cell in row) for row in row_cells if row]
    modal_width = max(set(widths), key=widths.count) if widths else 0
    consistency = (widths.count(modal_width) / len(widths)) if widths else 0.0

    filled = [t for t in texts if t]
    numeric_cells = [t for t in filled if is_numeric_cell(t)]
    lengths = [len(t) for t in filled]
    table_text = table.get_text(separator=" ", strip=True)

    profile = TableProfile(
        table_id=make_block_id(doc_id, "table", normalize(table_text) or f"empty-{source_order}"),
        doc_id=doc_id,
        section_id=section_id,
        source_order=source_order,
        rows=len(rows),
        max_cols=max(widths) if widths else 0,
        min_cols=min(widths) if widths else 0,
        col_count_consistency=round(consistency, 3),
        td_count=sum(1 for c in cells if c.name == "td"),
        th_count=sum(1 for c in cells if c.name == "th"),
        numeric_ratio=round(len(numeric_cells) / len(filled), 3) if filled else 0.0,
        empty_cell_ratio=round(1 - len(filled) / len(texts), 3) if texts else 0.0,
        numeric_row_count=sum(1 for row in row_cells
                              if sum(1 for c in row if is_numeric_cell(cell_text(c))) >= 2),
        year_count=sum(1 for t in texts if YEAR.search(t)),
        currency_count=sum(1 for t in texts if CURRENCY.search(t)),
        percent_count=sum(1 for t in texts if PERCENT.search(t)),
        paren_negative_count=sum(1 for t in texts if PAREN_NEGATIVE.match(t)),
        rowspan_count=sum(1 for c in cells if span_of(c, "rowspan") > 1),
        colspan_count=sum(1 for c in cells if span_of(c, "colspan") > 1),
        nested_table_count=len(table.find_all("table")),
        ix_nonfraction_count=count_ix_facts(table, "nonfraction"),
        ix_nonnumeric_count=count_ix_facts(table, "nonnumeric"),
        avg_cell_chars=round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        max_cell_chars=max(lengths) if lengths else 0,
        total_chars=len(table_text),
        has_header_row_candidate=header_row_candidate(row_cells),
        superheader_suspected=superheader_suspected(row_cells),
        text_preview=" ".join(table_text.split())[:300],
    )

    profile.primary_bucket, profile.buckets = classify(profile)
    return profile

def header_row_candidate(row_cells):
    """First non-empty row is mostly non-numeric labels."""
    for row in row_cells:
        texts = [cell_text(c) for c in row]
        filled = [t for t in texts if t]
        if not filled:
            continue
        non_numeric = [t for t in filled if not is_numeric_cell(t)]
        return len(filled) >= 2 and len(non_numeric) / len(filled) >= 0.5
    return False

def superheader_suspected(row_cells):
    """A spanning row sitting above a row of period-like values."""
    for i, row in enumerate(row_cells[:HEADER_REGION_ROWS]):
        spans = [c for c in row if span_of(c, "colspan") > 1]
        if not spans:
            continue
        for below in row_cells[i + 1:i + 3]:
            period_cells = [c for c in below if is_period_cell(cell_text(c))]
            if len(period_cells) >= 2:
                return True
    return False


def classify(profile):
    """
    Weak signals, combined. likely_data needs several independent reasons and
    no strong layout signal; everything unclear stays ambiguous.
    """
    data_signals = []
    layout_signals = []

    # Two gates, both required before anything can be called data. Numbers
    # alone are not enough: a cover page carries a date and a file number.
    # What makes a data table is the same shape repeating down the rows.
    has_numbers = profile.numeric_ratio >= NUMERIC_STRONG
    has_repeated_rows = profile.numeric_row_count >= MIN_NUMERIC_ROWS

    if profile.rows >= MIN_DATA_ROWS:
        data_signals.append("rows>=3")
    if profile.max_cols >= MIN_DATA_COLS:
        data_signals.append("cols>=2")
    if profile.col_count_consistency >= CONSISTENCY_STRONG:
        data_signals.append("consistent_widths")
    if profile.numeric_ratio >= NUMERIC_STRONG:
        data_signals.append("numeric")
    if profile.numeric_row_count >= MIN_NUMERIC_ROWS:
        data_signals.append("repeated_numeric_rows")
    if profile.year_count >= 2 or profile.superheader_suspected:
        data_signals.append("period_headers")
    if profile.currency_count or profile.percent_count or profile.paren_negative_count:
        data_signals.append("accounting_marks")
    if profile.has_header_row_candidate:
        data_signals.append("header_row")

    if profile.rows < MIN_DATA_ROWS:
        layout_signals.append("few_rows")
    if profile.col_count_consistency < 0.5:
        layout_signals.append("irregular_widths")
    if profile.max_cell_chars >= LONG_CELL_CHARS:
        layout_signals.append("paragraph_cell")
    if profile.avg_cell_chars >= PROSE_CELL_CHARS:
        layout_signals.append("prose_cells")
    if profile.nested_table_count:
        layout_signals.append("nested")
    if profile.max_cols <= 1:
        layout_signals.append("single_column")
    if profile.empty_cell_ratio >= MOSTLY_EMPTY:
        layout_signals.append("mostly_empty")
    if profile.numeric_ratio < NUMERIC_WEAK:
        layout_signals.append("few_numbers")
    if not has_repeated_rows:
        layout_signals.append("no_repeated_rows")

    buckets = []
    if profile.nested_table_count:
        buckets.append("nested")
    if profile.ix_nonfraction_count + profile.ix_nonnumeric_count >= 5 and cell_count(profile) and (
            (profile.ix_nonfraction_count + profile.ix_nonnumeric_count) / cell_count(profile) >= IXBRL_FACTS_PER_CELL):
        buckets.append("ixbrl_heavy")
    if profile.superheader_suspected or (profile.colspan_count and profile.has_header_row_candidate) \
            or profile.rowspan_count:
        buckets.append("merged_header")
    if profile.avg_cell_chars >= PROSE_CELL_CHARS and profile.numeric_ratio < NUMERIC_WEAK:
        buckets.append("mostly_text")

    is_data = has_numbers and has_repeated_rows and len(data_signals) >= 4 and len(layout_signals) <= 1
    is_layout = len(layout_signals) >= 2

    if is_data and not is_layout:
        buckets.append("likely_data")
        primary = "likely_data"
    elif is_layout:
        buckets.append("likely_layout")
        primary = "likely_layout"
    else:
        buckets.append("ambiguous")
        primary = "ambiguous"

    # A nested or prose-heavy table is never handed to a converter as data.
    if "nested" in buckets and primary == "likely_data":
        primary = "nested"
    elif "mostly_text" in buckets and primary != "likely_layout":
        primary = "mostly_text"

    return primary, buckets

def cell_count(profile):
    return profile.td_count + profile.th_count


# Document walking

def table_text_offsets(soup):
    """
    Character offset of each table within get_text(separator=" ", strip=True),
    so tables can be attributed to detected sections. Mirrors how bs4 joins
    stripped strings, which is what the loader produces.
    """
    ancestors = {}

    def tables_of(tag):
        key = id(tag)
        if key not in ancestors:
            parent = tag.parent
            inherited = tables_of(parent) if parent is not None else ()
            ancestors[key] = inherited + ((tag,) if tag.name == "table" else ())
        return ancestors[key]

    offsets = {}
    position = 0

    for string in text_nodes(soup):
        text = string.strip()
        if not text:
            continue
        for table in tables_of(string.parent):
            offsets.setdefault(id(table), position)
        position += len(text) + 1

    return offsets

def text_nodes(soup):
    """
    The strings get_text() actually joins. Exact type match on purpose:
    Comment, Doctype, Script and Stylesheet all subclass NavigableString but
    are excluded from get_text(), and counting them drifts every offset.
    """
    for descendant in soup.descendants:
        if type(descendant) in (NavigableString, CData):
            yield descendant

def load_soup(path, doc_type):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    if doc_type == "sec_filing":
        for hidden in soup.find_all("ix:header"):
            hidden.decompose()

    return soup

def profile_document(doc_id, boundaries=None):
    """Profiles every table in a document, attributing each to a section."""
    path, doc_type = DOCUMENTS[doc_id]
    soup = load_soup(path, doc_type)

    if boundaries is None:
        boundaries = section_boundaries_for(doc_id)

    offsets = table_text_offsets(soup)
    profiles = []

    for order, table in enumerate(soup.find_all("table")):
        offset = offsets.get(id(table), 0)
        profiles.append(profile_table(table, doc_id, section_id_at(offset, boundaries), order))

    return profiles

def section_boundaries_for(doc_id):
    from loader import load_document_ir

    path, doc_type = DOCUMENTS[doc_id]
    document = load_document_ir(path, doc_type, doc_id)
    boundaries, _ = detect_sections_report(document.blocks, strict=False)
    return boundaries


def find_table(doc_id, table_id=None, contains=None):
    """Locate a table by id or by a string in its flattened text."""
    path, doc_type = DOCUMENTS[doc_id]
    soup = load_soup(path, doc_type)
    boundaries = section_boundaries_for(doc_id) if table_id or contains else []
    offsets = table_text_offsets(soup)

    for order, table in enumerate(soup.find_all("table")):
        profile = profile_table(table, doc_id, section_id_at(offsets.get(id(table), 0), boundaries), order)
        text = table.get_text(separator=" ", strip=True)

        if table_id and profile.table_id == table_id:
            return table, profile
        if contains and all(needle in text for needle in contains):
            return table, profile

    return None, None

def dump_table(doc_id, table_id=None, contains=None, html_limit=3000):
    """Prints raw HTML, current get_text() output and the profile record."""
    table, profile = find_table(doc_id, table_id=table_id, contains=contains)

    if table is None:
        print(f"no table found in {doc_id} for table_id={table_id} contains={contains}")
        return None

    print("=" * 78)
    print(f"{profile.table_id}")
    print(f"section: {profile.section_id}   source_order: {profile.source_order}")
    print("=" * 78)

    print("\n--- RAW HTML ---\n")
    html = str(table)
    print(html[:html_limit] + (f"\n... [{len(html):,} chars total]" if len(html) > html_limit else ""))

    print("\n--- CURRENT get_text() OUTPUT ---\n")
    print(" ".join(table.get_text(separator=" ", strip=True).split()))

    print("\n--- PROFILE ---\n")
    for key, value in asdict(profile).items():
        print(f"  {key:<26} {value}")

    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump one table for inspection.")
    parser.add_argument("doc_id", choices=sorted(DOCUMENTS))
    parser.add_argument("--table-id")
    parser.add_argument("--contains", nargs="+", help="strings that appear in the table's text")
    args = parser.parse_args()

    dump_table(args.doc_id, table_id=args.table_id, contains=args.contains)
