"""
Table converter: turns data tables into self-describing records.

Each emitted line carries table title, row label, resolved column header,
value and unit, so a line that survives a chunk boundary alone still says
what it is. Markdown would not: separate the header row from the data rows
and you are back to bare numbers, which is the bug being fixed.

Nothing is guessed. A table whose column headers cannot be resolved goes down
the ambiguous path, which preserves row structure but asserts no
header-to-value relationship. Layout tables keep today's get_text() output.
"""
import re
from dataclasses import dataclass, field

from table_profile import (cell_text, direct_cells, direct_rows, is_numeric_cell,
                           is_period_cell, span_of)

# Conversion outcomes
CONVERTED_DATA = "converted_data"
CONVERTED_SUPERHEADER = "converted_superheader"
LAYOUT_PASSTHROUGH = "layout_passthrough"
AMBIGUOUS_PRESERVED = "ambiguous_preserved"

FIELD_SEPARATOR = " | "
UNRESOLVED_MARKER = "(column headers unresolved)"

# A cell holding only a currency sign, percent sign or dash carries no value.
SYMBOL_ONLY = re.compile(r"^[\s$€£¥%\-–—()]*$")
# Footnote markers SEC filings append to figures: "57,347 (e)", "1,234 (b)(c)"
FOOTNOTE_SUFFIX = re.compile(r"(?:\s*\([a-z]{1,3}\))+$", re.IGNORECASE)
SCALE_PHRASE = re.compile(r"\bin\s+(thousands|millions|billions|trillions)\b", re.IGNORECASE)
# "(in millions, except per share, ratio, employee data and where otherwise
# noted)" - the filing states which rows the scale does NOT apply to.
SCALE_EXCEPTIONS = re.compile(
    r"\bin\s+(?:thousands|millions|billions|trillions)\s*,?\s*except\s+([^)]*)", re.IGNORECASE)
EXCEPTION_NOISE = ("where otherwise noted", "otherwise noted", "as noted", "where noted", "data")
PERCENT_UNIT = "percent"

# Cells that stand in for a missing figure rather than labelling anything.
PLACEHOLDER_CELLS = {"nm", "n/a", "na", "n.m.", "-", "—", "–", "*", "%", "$", "()"}
FOOTNOTE_ONLY = re.compile(r"^(?:\s*\([a-z]{1,3}\))+$", re.IGNORECASE)
TITLE_HAS_WORD = re.compile(r"[A-Za-z]{3}")
# Running heads repeat on every page of a filing and name the form, not the
# table. SEC-generic, not filer-specific.
RUNNING_HEAD = re.compile(r"\bForm\s*10-[KQ]\b", re.IGNORECASE)
MAX_INTERIOR_LABELS = 1             # more than this means a second stub column

MIN_NUMERIC_CELLS_IN_DATA_ROW = 2   # same "repeated shape" rule the census uses
MAX_HEADER_ROWS = 4                 # header stacks deeper than this are not headers
TITLE_MAX_CHARS = 100               # a heading, not a paragraph
TITLE_LOOKBACK_NODES = 8            # how far back to look for that heading
SHORT_FURNITURE_CHARS = 120         # page numbers, running heads


@dataclass
class TableConversion:
    status: str
    reason: str
    records: list = field(default_factory=list)
    fallback_text: str = ""
    title: str = ""
    unit: str = ""
    resolved_columns: int = 0
    unresolved_columns: int = 0

    def serialized_text(self):
        if self.records:
            return "\n".join(self.records)
        return self.fallback_text


@dataclass
class GridCell:
    text: str
    is_origin: bool     # False when this position is covered by a span


def expand_grid(table):
    """
    Expands colspan/rowspan into a rectangular grid, so a superheader spanning
    six physical columns is present in all six.
    """
    grid = []
    pending = {}   # column -> (text, rows_remaining)

    for row in direct_rows(table):
        cells = direct_cells(row)
        line = []
        column = 0

        def place_carried():
            nonlocal column
            while column in pending:
                text, remaining = pending[column]
                line.append(GridCell(text, False))
                if remaining <= 1:
                    del pending[column]
                else:
                    pending[column] = (text, remaining - 1)
                column += 1

        for cell in cells:
            place_carried()
            text = cell_text(cell)
            colspan, rowspan = span_of(cell, "colspan"), span_of(cell, "rowspan")

            for offset in range(colspan):
                line.append(GridCell(text, offset == 0))
                if rowspan > 1:
                    pending[column] = (text, rowspan - 1)
                column += 1

        place_carried()
        grid.append(line)

    width = max((len(line) for line in grid), default=0)
    for line in grid:
        line.extend(GridCell("", True) for _ in range(width - len(line)))

    return grid

def is_data_row(line):
    """
    A row of years ("2025 2024 2023") is a header, not data, even though every
    cell is numeric. Only figures that are not periods count as values.
    """
    values = [cell.text for cell in line
              if cell.is_origin and is_numeric_cell(cell.text) and not is_period_cell(cell.text)]
    return len(values) >= MIN_NUMERIC_CELLS_IN_DATA_ROW

def split_header_and_data(grid):
    """Leading non-data rows are the header stack; the rest are data."""
    first_data = next((i for i, line in enumerate(grid) if is_data_row(line)), None)
    if first_data is None:
        return [], []

    # A lone spanning label above the data ("Selected income statement data")
    # is a group heading, not a header row. Including it would staple that
    # phrase onto every column header in the table.
    header_rows = [line for line in grid[:first_data]
                   if len({cell.text for cell in line if cell.is_origin and cell.text}) >= 2]
    return header_rows[-MAX_HEADER_ROWS:], grid[first_data:]

def is_value_cell(cell):
    """Origin only: a spanned copy of 21.7 is the same figure, not a second one."""
    return (cell.is_origin and cell.text and not SYMBOL_ONLY.match(cell.text)
            and is_numeric_cell(cell.text) and not is_period_cell(cell.text))

def row_values(line, first_value_column):
    return [(column, clean_value(cell.text)) for column, cell in enumerate(line)
            if column >= first_value_column and is_value_cell(cell)]

def value_columns(data_rows):
    """Columns where a real figure originates."""
    columns = set()
    for line in data_rows:
        for column, cell in enumerate(line):
            if is_value_cell(cell):
                columns.add(column)
    return sorted(columns)

def first_header_column(header_rows, fallback):
    """
    Leftmost column a period header occupies. The first *value* column can sit
    further right, because filings put the currency sign in its own cell
    between the stub and the figure. Using the value column as the boundary
    pulls that "$" into the row label and the period column into the
    qualifier, which produces headers like "Year ended December 31, 2025 2024".
    """
    columns = [column for line in header_rows for column, cell in enumerate(line)
               if cell.is_origin and cell.text and is_period_cell(cell.text)]
    return min(columns) if columns else fallback

def label_columns(boundary):
    """Stub columns: everything left of the first header/value column."""
    return list(range(0, boundary))

def resolve_column_headers(header_rows, width):
    """
    Walks the header stack top to bottom, joining each column's non-empty
    header cells into one path: "Three months ended June 30," + "2026".
    Spans were already expanded, so a superheader is present in every column
    it covers.
    """
    resolved = {}

    for column in range(width):
        parts = []
        for line in header_rows:
            if column >= len(line):
                continue
            text = line[column].text.strip()
            if text and (not parts or parts[-1] != text):
                parts.append(text)

        resolved[column] = " ".join(parts).strip()

    return resolved

def header_qualifier(header_rows, first_value_column):
    """
    The stub cell of a header row often qualifies every period column:
    "Year ended December 31," above 2025 / 2024 / 2023. Without it the
    resolved header is a bare year. Scale parentheticals are dropped, since
    "(in millions)" is a unit, not part of the period.
    """
    parts = []

    for line in header_rows:
        for column in range(min(first_value_column, len(line))):
            text = line[column].text.strip()
            if not text:
                continue
            text = re.sub(r"\((?:[^()]*\b(?:in\s+(?:thousands|millions|billions|trillions)|except)\b[^()]*)\)",
                          "", text, flags=re.IGNORECASE).strip()
            if text and text not in parts:
                parts.append(text)

    return " ".join(parts).strip()

def apply_qualifier(headers, qualifier):
    if not qualifier:
        return headers
    return {column: (path if not path or qualifier in path else f"{qualifier} {path}").strip()
            for column, path in headers.items()}

def header_confidence(headers, data_rows, first_value_column, header_rows):
    """
    Checked per row, not per column: a spanned figure can originate in a
    different column than the row above it, so what matters is that within
    one row every value lands on a distinct, non-empty header path. Two
    values sharing a path would mean the span was never resolved.
    """
    if not header_rows:
        return False, "AMBIGUOUS_HEADER_UNRESOLVED"

    saw_values = False

    for line in data_rows:
        values = row_values(line, first_value_column)
        if not values:
            continue
        saw_values = True

        paths = [headers.get(column, "") for column, _ in values]
        if any(not path for path in paths):
            return False, "AMBIGUOUS_HEADER_UNRESOLVED"
        if len(set(paths)) != len(paths):
            return False, "AMBIGUOUS_HEADER_AMBIGUOUS_PATHS"

    if not saw_values:
        return False, "AMBIGUOUS_NO_VALUES"

    return True, ""


def clean_value(text):
    """
    Strips footnote markers and a leading currency sign. Filings put the sign
    in its own cell or inside the value cell depending on the table; the unit
    field records the currency either way, so values stay uniform.
    """
    value = FOOTNOTE_SUFFIX.sub("", text).strip()
    return re.sub(r"^[$€£¥]\s*", "", value).strip()

def find_scale(*texts):
    for text in texts:
        match = SCALE_PHRASE.search(text or "")
        if match:
            return match.group(1).lower()
    return ""

def parse_scale_exceptions(*texts):
    """Row kinds the stated scale explicitly does not cover."""
    for text in texts:
        match = SCALE_EXCEPTIONS.search(text or "")
        if not match:
            continue

        tokens = []
        for piece in re.split(r",|\band\b", match.group(1)):
            token = piece.strip().lower().rstrip(".")
            token = re.sub(r"\s+data$", "", token)
            if token and token not in EXCEPTION_NOISE and len(token) > 2:
                tokens.append(token)
        return tokens

    return []

def value_unit(line, column, row_label, header_path, scale, exceptions):
    """
    Units are read per value, not per row: a tax table puts expense in
    millions and its rate in percent on the same line, so a row-level verdict
    mislabels one of them.

    Filings put the sign in its own cell beside the figure ("$" then "1,800",
    or "0.73" then "%"), so the immediate neighbours are the evidence. A scale
    is never applied to a row the filing excluded from it: a capital ratio is
    not "14.2 millions". With no evidence the field is omitted.
    """
    cell_text_value = line[column].text if column < len(line) else ""
    left = " ".join(line[c].text for c in range(max(0, column - 2), column))
    right = " ".join(line[c].text for c in range(column + 1, min(len(line), column + 3)))

    if "%" in cell_text_value or "%" in right or "%" in header_path:
        return PERCENT_UNIT

    excluded = any(token in row_label.lower() for token in exceptions)

    if re.search(r"[$€£¥]", cell_text_value + left) or re.search(r"[$€£¥]", header_path):
        return f"USD {scale}".strip() if scale and not excluded else "USD"
    return "" if excluded else scale

def table_title(table):
    """
    Nearest preceding heading-like text. A paragraph is not a title, so long
    strings and sentences are skipped rather than truncated.
    """
    node = table
    for _ in range(TITLE_LOOKBACK_NODES):
        node = node.find_previous(string=True)
        if node is None:
            return ""

        text = " ".join(node.strip().split())
        if not text:
            continue
        # Page numbers, stray symbols and running heads are not titles.
        if not TITLE_HAS_WORD.search(text) or RUNNING_HEAD.search(text):
            continue
        if len(text) <= TITLE_MAX_CHARS and not text.endswith("."):
            return text

    return ""

def interior_label_count(data_rows, value_cols):
    """
    Text cells sitting between value columns mean the table has a second stub
    column - two independent tables printed side by side, like a headcount by
    region next to a headcount by business. One row label cannot describe both
    halves, so binding anything here would invent relationships.
    """
    if len(value_cols) < 2:
        return 0

    first, last = min(value_cols), max(value_cols)
    count = 0

    for line in data_rows:
        for column in range(first, min(last, len(line))):
            cell = line[column]
            text = cell.text.strip()
            if not cell.is_origin or not text or is_numeric_cell(text):
                continue
            if text.lower() in PLACEHOLDER_CELLS or FOOTNOTE_ONLY.match(text):
                continue
            if TITLE_HAS_WORD.search(text):
                count += 1

    return count


def build_records(title, header_rows, data_rows, value_cols, headers, scale,
                  exceptions=(), boundary=None):
    """One record per value cell, each carrying its full context."""
    records = []
    missing_context = 0
    group_label = ""
    first_value_column = min(value_cols) if value_cols else 0
    label_cols = label_columns(boundary if boundary is not None else first_value_column)

    for line in data_rows:
        texts = [cell.text for cell in line]
        label_parts = [line[c].text for c in label_cols
                       if c < len(line) and line[c].text and not SYMBOL_ONLY.match(line[c].text)]
        # De-duplicate the label repeated across spanned stub columns.
        row_label = " ".join(dict.fromkeys(label_parts)).strip()

        values = row_values(line, first_value_column)

        if not values:
            # A label with no figures is a group heading for the rows below it.
            if row_label:
                group_label = row_label
            continue

        full_label = f"{group_label} - {row_label}" if group_label and row_label else row_label

        for column, value in values:
            header_path = headers.get(column, "")
            # A record that cannot say which row and which column it came
            # from is not self-describing, so the table is not understood.
            if not full_label or not header_path:
                missing_context += 1

            fields = [part for part in (title, full_label, header_path, value) if part]
            unit = value_unit(line, column, full_label, headers.get(column, ""), scale, exceptions)
            if unit:
                fields.append(unit)
            records.append(FIELD_SEPARATOR.join(fields))

    return records, missing_context

def build_preserved_lines(title, header_rows, data_rows, value_cols):
    """
    Ambiguous path. Keeps each row's label with its own figures, which
    get_text() does not, but binds nothing to a column header. Header rows are
    reproduced verbatim and explicitly marked unresolved.
    """
    lines = []
    prefix = f"{title} {UNRESOLVED_MARKER}".strip() if title else UNRESOLVED_MARKER

    for line in header_rows:
        texts = [cell.text for cell in line if cell.is_origin and cell.text]
        if texts:
            lines.append(FIELD_SEPARATOR.join([prefix, "header row"] + texts))

    for line in data_rows:
        texts = [cell.text for cell in line if cell.is_origin and cell.text]
        if texts:
            lines.append(FIELD_SEPARATOR.join([prefix] + texts))

    return lines


def convert_table(table, profile):
    """Routes one table and produces its serialized form."""
    fallback = table.get_text(separator=" ", strip=True)
    title = table_title(table)

    if profile.primary_bucket == "likely_layout":
        reason = ("LAYOUT_SHORT_FURNITURE" if profile.total_chars <= SHORT_FURNITURE_CHARS
                  else "LAYOUT_PROSE" if "mostly_text" in profile.buckets
                  else "LAYOUT_CLASSIFIER")
        return TableConversion(LAYOUT_PASSTHROUGH, reason, fallback_text=fallback, title=title)

    grid = expand_grid(table)
    header_rows, data_rows = split_header_and_data(grid)
    value_cols = value_columns(data_rows)

    if not data_rows or not value_cols:
        return TableConversion(AMBIGUOUS_PRESERVED, "AMBIGUOUS_NO_DATA_ROWS",
                               records=build_preserved_lines(title, header_rows, data_rows, value_cols),
                               fallback_text=fallback, title=title)

    width = max(len(line) for line in grid)
    first_value_column = min(value_cols)
    boundary = first_header_column(header_rows, first_value_column)
    headers = resolve_column_headers(header_rows, width)
    headers = apply_qualifier(headers, header_qualifier(header_rows, boundary))
    confident, failure_reason = header_confidence(headers, data_rows, first_value_column, header_rows)

    if not confident:
        return TableConversion(
            AMBIGUOUS_PRESERVED, failure_reason,
            records=build_preserved_lines(title, header_rows, data_rows, value_cols),
            fallback_text=fallback, title=title,
            unresolved_columns=len(value_cols))

    if interior_label_count(data_rows, value_cols) > MAX_INTERIOR_LABELS:
        return TableConversion(
            AMBIGUOUS_PRESERVED, "AMBIGUOUS_MULTIPLE_STUB_COLUMNS",
            records=build_preserved_lines(title, header_rows, data_rows, value_cols),
            fallback_text=fallback, title=title,
            unresolved_columns=len(value_cols))

    if profile.primary_bucket == "ambiguous":
        return TableConversion(
            AMBIGUOUS_PRESERVED, "AMBIGUOUS_CLASSIFIER",
            records=build_preserved_lines(title, header_rows, data_rows, value_cols),
            fallback_text=fallback, title=title,
            unresolved_columns=len(value_cols))

    stub_texts = " ".join(cell.text for line in grid[:len(header_rows) + 1] for cell in line if cell.is_origin)
    scale = find_scale(stub_texts, title, fallback[:200])
    exceptions = parse_scale_exceptions(stub_texts, fallback[:400])
    records, missing_context = build_records(title, header_rows, data_rows, value_cols,
                                             headers, scale, exceptions, boundary)

    if missing_context:
        return TableConversion(
            AMBIGUOUS_PRESERVED, "AMBIGUOUS_ROW_CONTEXT_MISSING",
            records=build_preserved_lines(title, header_rows, data_rows, value_cols),
            fallback_text=fallback, title=title,
            unresolved_columns=len(value_cols))

    if not records:
        return TableConversion(AMBIGUOUS_PRESERVED, "AMBIGUOUS_NO_VALUES",
                               records=build_preserved_lines(title, header_rows, data_rows, value_cols),
                               fallback_text=fallback, title=title)

    status = CONVERTED_SUPERHEADER if len(header_rows) > 1 else CONVERTED_DATA
    reason = "DATA_SUPERHEADER_RESOLVED" if len(header_rows) > 1 else "DATA_NUMERIC_GRID"

    return TableConversion(status, reason, records=records, fallback_text=fallback,
                           title=title, unit=scale, resolved_columns=len(value_cols))
