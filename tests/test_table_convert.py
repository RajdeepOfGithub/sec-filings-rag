"""
Tests for the table converter.

The binding tests are the point: 21.7 must belong to "Three months ended
June 30, 2026" and never to 2025. That is the measured q01 failure.

Run from the project root:
    python tests/test_table_convert.py
"""
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup

from table_convert import (AMBIGUOUS_PRESERVED, CONVERTED_DATA, CONVERTED_SUPERHEADER,
                           LAYOUT_PASSTHROUGH, UNRESOLVED_MARKER, convert_table, expand_grid)
from table_profile import find_table, profile_table

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)

def make_table(html, title="Test table of figures"):
    """
    Fixtures carry a preceding heading, because a converted table needs a
    title to anchor its scope - without one the converter routes to ambiguous
    on purpose (see AMBIGUOUS_NO_TABLE_SCOPE).
    """
    soup = BeautifulSoup(f"<p>{title}</p>{html}" if title else html, "lxml")
    table = soup.find("table")
    return table, profile_table(table, "TEST_10-K_FY2025", "", 0)


def test_q01_binding():
    """21.7 belongs to 2026, 29.8 to 2025. The whole step exists for this."""
    print("\nq01: values bind to the correct superheader column")

    table, profile = find_table("JPMC_10-Q_Q2-2026", contains=["21.7", "29.8", "49.3", "59.8"])
    conversion = convert_table(table, profile)
    records = conversion.records

    check("status is converted_superheader", conversion.status == CONVERTED_SUPERHEADER, conversion.reason)

    def record_for(value):
        return next((r for r in records if f"| {value} |" in r or r.endswith(f"| {value}")), "")

    check("21.7 -> Three months ended June 30, 2026",
          "Three months ended June 30, 2026" in record_for("21.7"))
    check("21.7 is NOT bound to 2025", "2025" not in record_for("21.7").split("| 21.7")[0])
    check("29.8 -> Three months ended June 30, 2025",
          "Three months ended June 30, 2025" in record_for("29.8"))
    check("49.3 -> Six months ended June 30, 2026",
          "Six months ended June 30, 2026" in record_for("49.3"))
    check("6,703 -> Three months 2026 with USD unit",
          "Three months ended June 30, 2026" in record_for("6,703")
          and record_for("6,703").endswith("USD millions"), record_for("6,703"))
    check("every value cell became a record", len(records) == 8, f"{len(records)} records")


def test_control_not_degraded():
    print("\ncontrol table keeps its period context")

    table, profile = find_table("JPMC_10-K_FY2025", contains=["114.4", "91.7", "69.5"])
    conversion = convert_table(table, profile)

    check("status is converted_data", conversion.status == CONVERTED_DATA, conversion.reason)
    record = next((r for r in conversion.records if "114.4" in r), "")
    check("114.4 keeps the full period header", "Year ended December 31, 2025" in record, record)
    check("114.4 not bound to another year", "2024" not in record and "2023" not in record)
    check("row label survives", "Total number of shares of common stock repurchased" in record)


def test_q04_ratio_units():
    print("\nq04: a capital ratio is not measured in millions")

    table, profile = find_table("JPMC_10-Q_Q2-2026", contains=["2Q26", "1Q26", "4Q25"])
    conversion = convert_table(table, profile)
    cet1 = [r for r in conversion.records if "CET1" in r and "Standardized" in r]

    check("CET1 rows converted", len(cet1) >= 5, f"{len(cet1)} records")
    check("14.2 bound to 2Q26", any("2Q26" in r and "14.2" in r for r in cet1))
    check("ratio rows carry no millions unit", all(not r.endswith("millions") for r in cet1),
          next((r for r in cet1 if r.endswith("millions")), ""))
    # The group heading belongs in the row label, never in the column header.
    def column_header(record):
        fields = record.split(" | ")
        return fields[2] if len(fields) > 3 else ""

    check("group heading is not in any column header",
          all("Selected income statement data" not in column_header(r) for r in conversion.records))
    check("group heading is kept as row-label context",
          any("Selected income statement data - " in r for r in conversion.records))


def test_superheader_span_expansion():
    print("\nspans expand so a superheader reaches every column it covers")

    table, _ = make_table("""
        <table>
          <tr><td></td><td colspan="2">Three months ended June 30,</td><td colspan="2">Six months ended June 30,</td></tr>
          <tr><td>(in millions)</td><td>2026</td><td>2025</td><td>2026</td><td>2025</td></tr>
          <tr><td>Shares repurchased</td><td>21.7</td><td>29.8</td><td>49.3</td><td>59.8</td></tr>
        </table>""")
    grid = expand_grid(table)

    check("superheader present in both covered columns",
          grid[0][1].text == grid[0][2].text == "Three months ended June 30,")
    check("spanned copies are marked non-origin", grid[0][1].is_origin and not grid[0][2].is_origin)


def test_untitled_table_is_not_asserted():
    """q02: a segment total with no scope anchor read as a firmwide total."""
    print("\na table with no title cannot establish scope")

    table, profile = make_table("""
        <table>
          <tr><td>(in millions)</td><td>2026</td><td>2025</td></tr>
          <tr><td>Total net revenue</td><td>24,853</td><td>19,535</td></tr>
          <tr><td>Total noninterest expense</td><td>11,390</td><td>9,641</td></tr>
        </table>""", title="")
    conversion = convert_table(table, profile)

    check("routed to ambiguous", conversion.status == AMBIGUOUS_PRESERVED, conversion.reason)
    check("reason is the missing scope", conversion.reason == "AMBIGUOUS_NO_TABLE_SCOPE", conversion.reason)
    check("no record asserts the figure",
          not any(r.startswith("Total net revenue | ") for r in conversion.records))


def test_truncated_titles_are_rejected():
    print("\na sentence fragment is not a title")

    for bad in ["On July 1, 2025, the Firm announced that its Board had authorized a new $",
                "Shares repurchased during the year were as follows, and the",
                "Total capital of"]:
        table, profile = make_table("""
            <table>
              <tr><td>(in millions)</td><td>2026</td><td>2025</td></tr>
              <tr><td>Revenue</td><td>1,234</td><td>1,000</td></tr>
              <tr><td>Expense</td><td>567</td><td>500</td></tr>
            </table>""", title=bad)
        conversion = convert_table(table, profile)
        check(f"rejected: {bad[:40]}...", conversion.title != bad, repr(conversion.title))


def test_unresolved_headers_go_to_ambiguous():
    print("\nunresolvable headers are preserved, not guessed")

    table, profile = make_table("""
        <table>
          <tr><td>Portfolio</td><td>14.2</td><td>14.3</td><td>14.6</td></tr>
          <tr><td>Other</td><td>11.1</td><td>11.2</td><td>11.3</td></tr>
        </table>""")
    conversion = convert_table(table, profile)

    check("routed to ambiguous", conversion.status == AMBIGUOUS_PRESERVED, conversion.reason)
    check("marked unresolved", all(UNRESOLVED_MARKER in r for r in conversion.records))
    check("row label stays with its own figures",
          any("Portfolio" in r and "14.2" in r and "11.1" not in r for r in conversion.records))
    check("no header-to-value claim is made",
          not any("| 2026 |" in r or "| 2025 |" in r for r in conversion.records))


def test_layout_table_passthrough():
    print("\nlayout tables keep today's get_text behaviour")

    table, profile = make_table("<table><tr><td>110 JPMorgan Chase &amp; Co./2025 Form 10-K</td></tr></table>")
    conversion = convert_table(table, profile)

    check("routed to layout passthrough", conversion.status == LAYOUT_PASSTHROUGH, conversion.reason)
    check("no records emitted", conversion.records == [])
    check("text is unchanged", conversion.serialized_text() == "110 JPMorgan Chase & Co./2025 Form 10-K",
          conversion.serialized_text())


def test_year_header_row_is_not_data():
    print("\na row of years is a header, not a data row")

    table, profile = make_table("""
        <table>
          <tr><td>Year ended December 31, (in millions)</td><td>2025</td><td>2024</td></tr>
          <tr><td>Revenue</td><td>$ 1,234</td><td>$ 1,000</td></tr>
        </table>""")
    conversion = convert_table(table, profile)

    check("converted rather than treated as data rows", conversion.records != [])
    record = next((r for r in conversion.records if "1,234" in r), "")
    check("value bound to the year header", "Year ended December 31, 2025" in record, record)
    check("no record emitted for the year cells themselves",
          not any(r.rstrip().endswith("| 2025") for r in conversion.records))


if __name__ == "__main__":
    test_q01_binding()
    test_control_not_degraded()
    test_q04_ratio_units()
    test_superheader_span_expansion()
    test_unresolved_headers_go_to_ambiguous()
    test_layout_table_passthrough()
    test_year_header_row_is_not_data()
    test_untitled_table_is_not_asserted()
    test_truncated_titles_are_rejected()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("All checks passed.")
