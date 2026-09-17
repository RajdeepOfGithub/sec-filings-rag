"""
Prose rendering of table records, for cross-encoder scoring only.

The cross-encoder was trained on natural-language passages scored against
natural-language queries. A pipe-delimited record is not that shape, and it
scores badly: q01's correct record scored 3.172 against 5.811 for a prose-ish
table of the wrong year.

So records are rendered into sentences for scoring. The chunk text that is
stored, retrieved and handed to the LLM is never changed by this module.

    Common stock repurchases | Total number of shares ... | Three months ended June 30, 2026 | 21.7 | millions
    ->
    Common stock repurchases, total number of shares ..., three months ended June 30, 2026: 21.7 million
"""
import re

FIELD_SEPARATOR = " | "

# Trailing unit field, as emitted by table_convert.value_unit().
UNIT_FIELD = re.compile(r"^(?:USD\s+)?(?:thousand|million|billion|trillion)s?$|^percent$|^USD$",
                        re.IGNORECASE)
SCALE_PLURAL = re.compile(r"\b(thousand|million|billion|trillion)s\b", re.IGNORECASE)

# Lines that are not records: continuation headers, preserved header rows.
NOT_A_RECORD = ("(continued)", "(continued from", "(column headers unresolved)")

RENDER_STATS = {"chunks": 0, "records": 0, "truncated": 0}

# ms-marco-MiniLM-L-6-v2 takes 512 tokens for query + passage together. Chunks
# are ~500 characters, so rendering stays well under; the cap is a guard, and
# every trim is counted so it can be reported rather than hidden.
MAX_RENDER_CHARS = 1800


def decapitalize(text):
    """
    Lowercase only the leading letter, so "June" keeps its capital. Acronyms
    are left alone: CET1, ROE, LCR and USD must not become cET1, rOE, lCR.
    """
    if not text:
        return text
    if len(text) > 1 and text[1].isupper():
        return text
    return text[0].lower() + text[1:]

def singular_scale(unit):
    return SCALE_PLURAL.sub(lambda m: m.group(1), unit)

def render_value(value, unit):
    if not unit:
        return value

    unit = singular_scale(unit.strip())
    if unit.lower() == "percent":
        return f"{value} percent"
    if unit.upper() == "USD":
        return f"${value}"
    if unit.upper().startswith("USD "):
        return f"${value} {unit[4:]}"
    return f"{value} {unit}"

def render_record(line):
    """One record line -> one sentence. Non-record lines are flattened."""
    fields = [f.strip() for f in line.split(FIELD_SEPARATOR) if f.strip()]

    if not fields:
        return ""
    if len(fields) == 1:
        return fields[0]
    if any(marker in line for marker in NOT_A_RECORD):
        return ", ".join(fields)

    unit = fields.pop() if UNIT_FIELD.match(fields[-1]) else ""
    if not fields:
        return unit

    value = fields.pop()
    if not fields:
        return render_value(value, unit)

    context = [fields[0]] + [decapitalize(f) for f in fields[1:]]
    return f"{', '.join(context)}: {render_value(value, unit)}"

def render_chunk(text):
    """Every record in a chunk, rendered and joined into running prose."""
    rendered = [render_record(line) for line in text.split("\n") if line.strip()]
    rendered = [r for r in rendered if r]

    RENDER_STATS["chunks"] += 1
    RENDER_STATS["records"] += len(rendered)

    out = " ".join(rendered)
    if len(out) > MAX_RENDER_CHARS:
        RENDER_STATS["truncated"] += 1
        out = out[:MAX_RENDER_CHARS]

    return out

def scoring_text(result):
    """
    What the cross-encoder sees. Table records are rendered; narrative chunks
    pass through untouched, as do chunks with no kind metadata (v1).
    """
    if result.get("metadata", {}).get("kind") == "table_records":
        return render_chunk(result["text"])
    return result["text"]
