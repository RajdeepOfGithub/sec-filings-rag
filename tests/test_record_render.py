"""
Tests for the scoring-time prose rendering.

The rule that matters: rendering changes only what the cross-encoder sees.
Stored chunk text, retrieval and the LLM context are untouched.

Run from the project root:
    python tests/test_record_render.py
"""
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from record_render import MAX_RENDER_CHARS, RENDER_STATS, render_chunk, render_record, scoring_text

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def test_target_format():
    print("\nrecords render as sentences")

    rendered = render_record(
        "Common stock repurchases | Total number of shares of common stock repurchased "
        "| Three months ended June 30, 2026 | 21.7 | millions")
    check("matches the specified rendering",
          rendered == "Common stock repurchases, total number of shares of common stock "
                      "repurchased, three months ended June 30, 2026: 21.7 million", rendered)

    check("currency renders with a symbol",
          render_record("Capital | Purchase price | 2026 | 6,703 | USD millions")
          == "Capital, purchase price, 2026: $6,703 million")
    check("percent stays percent",
          render_record("Capital | CET1 ratio | 2Q26 | 14.2 | percent")
          == "Capital, CET1 ratio, 2Q26: 14.2 percent")
    check("a record with no unit still renders",
          render_record("Highlights | CET1 ratio | 2Q26 | 14.2") == "Highlights, CET1 ratio, 2Q26: 14.2")
    check("mid-sentence capitals survive",
          "three months ended June 30, 2026" in rendered)


def test_figures_are_never_altered():
    print("\nfigures pass through untouched")

    for value in ["21.7", "6,703", "( 3,005 )", "14.2", "0.73"]:
        rendered = render_record(f"T | Row | 2026 | {value} | millions")
        check(f"{value} survives rendering", value in rendered, rendered)


def test_non_record_lines():
    print("\nnon-record lines degrade gracefully")

    check("continuation header flattens",
          render_record("Capital (continued) | values in millions unless the record states otherwise")
          == "Capital (continued), values in millions unless the record states otherwise")
    check("preserved header row flattens",
          "header row" in render_record("T (column headers unresolved) | header row | Region | Employees"))
    check("empty line renders empty", render_record("") == "")


def test_only_table_records_are_rendered():
    print("\nnarrative and v1 chunks pass through unchanged")

    prose = {"metadata": {"kind": "narrative"},
             "text": "Net interest income increased driven by higher Markets NII."}
    check("narrative untouched", scoring_text(prose) == prose["text"])

    v1 = {"metadata": {"company": "JPMC"}, "text": "a | b | c"}
    check("chunk without kind untouched", scoring_text(v1) == v1["text"])

    records = {"metadata": {"kind": "table_records"},
               "text": "Capital | Shares | 2026 | 21.7 | millions"}
    check("records rendered", scoring_text(records) != records["text"])
    check("stored text object not mutated", records["text"] == "Capital | Shares | 2026 | 21.7 | millions")


def test_multi_record_chunk_and_truncation():
    print("\nmulti-record chunks join, oversize renders are counted")

    chunk = "\n".join(f"Capital | Row {i} | 2026 | {i}.5 | millions" for i in range(6))
    rendered = render_chunk(chunk)
    check("every record appears", all(f"row {i}" in rendered for i in range(6)), rendered[:80])
    check("joined into one string", "\n" not in rendered)

    before = RENDER_STATS["truncated"]
    huge = "\n".join(f"Capital | {'very long row label ' * 20} | 2026 | {i} | millions"
                     for i in range(20))
    out = render_chunk(huge)
    check("oversize rendering is trimmed", len(out) == MAX_RENDER_CHARS, f"{len(out)} chars")
    check("and counted", RENDER_STATS["truncated"] == before + 1)


if __name__ == "__main__":
    test_target_format()
    test_figures_are_never_altered()
    test_non_record_lines()
    test_only_table_records_are_rendered()
    test_multi_record_chunk_and_truncation()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("All checks passed.")
