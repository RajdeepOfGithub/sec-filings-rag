"""
Graders for the baseline questions.

Eyeball scoring disagreed with itself on q14/q15, so scoring is code now.
Every grader is deterministic string/metadata matching except claims_score,
which needs an LLM because the claims are paraphrasable facts.

The graders are kept separate on purpose. "The number is right" and "the
number came from the right place" are different questions: v2c's q03 quoted
4,908, a real figure from the CORPORATE segment, which a faithfulness metric
would happily call grounded. A tabular answer is only correct when value,
document and period are all right.

Nothing here depends on chunk id format, so it works on any run directory.
"""
import json
import re

from generator import call_llm

JUDGE_MODEL = "gpt-4o-mini"

# Phrases the generator uses when it refuses to answer from the context.
DECLINE_PHRASES = (
    "does not provide", "does not contain", "not provided", "cannot answer",
    "not specified", "is missing", "are missing", "lacks details", "lacks specific",
    "no information", "not included", "not disclosed", "unable to determine",
    "not detailed", "does not include", "does not specify", "not available",
)

# Unit spellings that mean the same thing. An answer may use any of them.
UNIT_EQUIVALENTS = {
    "usd millions": ["usd millions", "usd million", "$ million", "$million", "million dollars",
                     "millions of dollars", "million usd", "millions usd", "in millions",
                     "$ billion", "billion dollars", "billion usd", "usd billions", "usd billion",
                     "millions", "million", "billions", "billion"],
    "million shares": ["million shares", "millions of shares", "shares", "million common shares"],
    "percent": ["percent", "%", "percentage points", "pct"],
    "usd billions, full year": ["billion", "billions", "usd billion", "usd billions",
                                "$ billion", "billion dollars"],
}

# Scale words that multiply a figure in an answer.
SCALES = {"thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6,
          "billion": 1e9, "billions": 1e9, "trillion": 1e12, "trillions": 1e12}

# What each stated unit means in absolute terms.
UNIT_FACTORS = {"usd millions": 1e6, "million shares": 1e6, "percent": 1.0,
                "usd billions, full year": 1e9}

# The evidence names the period a figure BELONGS to; chunk metadata names the
# period of the FILING it sits in. A six-month figure and a quarter-over-quarter
# comparison both live in the Q2-2026 10-Q, so both are satisfied by it.
PERIOD_EQUIVALENTS = {
    "Q2-2026": {"Q2-2026"},
    "H1-2026": {"Q2-2026"},
    "Q2-2026 vs Q2-2025": {"Q2-2026"},
    "FY2025": {"FY2025"},
}

VALUE_TOLERANCE = 0.01      # 1%: "$57.3 billion" vs "57,347 million"
PERCENT_LOOKAHEAD = 14      # characters after a figure searched for % / "percent"
CLAIMS_PASS_MARK = 0.75

# Expected keys that are figures the answer may state. change_pct is tracked
# separately: q07 asks for two figures, and the percentage is derived.
PRIMARY_VALUE_KEYS = ("value", "shares", "cost", "value_2026", "value_2025", "value_from", "value_to")


def required_value_keys(expected):
    """
    Which figures the answer actually has to state.

    When the spec carries a "value", that is the figure the question asks for
    and any others are supplementary ground truth: q08 asks how much was
    spent, and its 114.4 million shares is context, not part of the ask.
    Without a "value" key every figure is part of the ask - q01 asks for share
    count AND cost, q07 for both years.
    """
    if "value" in expected:
        return ["value"]
    return [key for key in PRIMARY_VALUE_KEYS if key in expected]

def supplementary_value_keys(expected):
    return [key for key in PRIMARY_VALUE_KEYS
            if key in expected and key not in required_value_keys(expected)]

USAGE = {"judge_calls": 0, "claims_judged": 0}


def cited_metadata(run):
    return [c["metadata"] for c in run.get("resolved_citations", []) if c.get("metadata")]

def document_of(metadata):
    """Chunk metadata carries company/form/period, not a document id."""
    parts = [metadata.get("company"), metadata.get("form"), metadata.get("period")]
    return "_".join(p for p in parts if p)

def answer_text(run):
    return run.get("answer_text", "") or ""


def document_correct(run, expected_evidence):
    """True if at least one citation names the expected document."""
    expected = expected_evidence.get("document")
    if not expected:
        return None

    return any(document_of(m) == expected for m in cited_metadata(run))

def period_correct(run, expected_evidence):
    expected = expected_evidence.get("period")
    if not expected:
        return None

    acceptable = PERIOD_EQUIVALENTS.get(expected, {expected})
    return any(m.get("period") in acceptable for m in cited_metadata(run))


def stated_units(expected):
    return [(key, value) for key, value in expected.items()
            if key == "unit" or key.endswith("_unit")]

def unit_matches(text, unit):
    spellings = UNIT_EQUIVALENTS.get(unit.lower(), [unit.lower()])
    lowered = text.lower()
    return any(spelling in lowered for spelling in spellings)

def unit_correct(run, expected, category):
    """Tabular only. Every stated unit must appear, in any accepted spelling."""
    if category != "tabular":
        return None

    # Only the units of the figures the question asks for. q08 states a
    # shares_unit for a figure it does not ask the answer to give.
    units = [unit_for_key(expected, key) for key in required_value_keys(expected)]
    units = [u for u in units if u]
    if not units:
        units = [unit for _, unit in stated_units(expected)]
    if not units:
        return None

    return all(unit_matches(answer_text(run), unit) for unit in units)


def extract_numbers(text):
    """
    Every figure in the answer, as (number, scale_factor). A figure with no
    scale word gets factor 1, and the comparison tries both readings.
    """
    found = []

    for match in re.finditer(r"(-?\d[\d,]*\.?\d*)\s*(thousand|thousands|million|millions|"
                             r"billion|billions|trillion|trillions)?", text, re.IGNORECASE):
        raw = match.group(1).replace(",", "")
        try:
            number = float(raw)
        except ValueError:
            continue
        scale = SCALES.get((match.group(2) or "").lower(), 1.0)
        trailing = text[match.end():match.end() + PERCENT_LOOKAHEAD].lower()
        found.append((number, scale, trailing))

    return found

def reads_as_percent(trailing):
    return "%" in trailing or "percent" in trailing

def value_present(text, expected_value, unit):
    """
    True if some figure in the text equals the expected value. Presentation is
    allowed to differ: "$57.3 billion" and "57,347 million" are the same
    number, so both the written scale and the expected unit's scale are tried.
    """
    normalized = (unit or "").lower()
    factor = UNIT_FACTORS.get(normalized, 1.0)
    target = expected_value * factor
    wants_percent = normalized == "percent"

    for number, scale, trailing in extract_numbers(text):
        # "14.2" alone is not 14.2 percent: a $14.2 billion provision has the
        # same digits. A percentage has to read as one.
        if wants_percent and not reads_as_percent(trailing):
            continue

        for candidate in {number * scale, number * factor, number}:
            if target == 0:
                if abs(candidate) < 1e-9:
                    return True
            elif abs(candidate - target) / abs(target) <= VALUE_TOLERANCE:
                return True

    return False

def unit_for_key(expected, key):
    """q01 states shares_unit and cost_unit; a plain value uses "unit"."""
    specific = expected.get(f"{key}_unit")
    if specific:
        return specific
    if key in ("shares",):
        return expected.get("shares_unit", "million shares")
    return expected.get("unit", "")

def value_correct(run, expected, category):
    """
    Tabular only, and purely numeric: does the answer state the expected
    figure(s)? Whether the figure came from the right document and period is
    graded separately, so a wrong-scope match cannot pass as correct on its
    own.
    """
    if category != "tabular":
        return None

    keys = required_value_keys(expected)
    if not keys:
        return None

    text = answer_text(run)
    return all(value_present(text, expected[key], unit_for_key(expected, key)) for key in keys)

def supplementary_values_correct(run, expected, category):
    """Tracked, not required: figures in the ground truth the question did not ask for."""
    if category != "tabular":
        return None

    keys = supplementary_value_keys(expected)
    if not keys:
        return None

    text = answer_text(run)
    return all(value_present(text, expected[key], unit_for_key(expected, key)) for key in keys)

def change_pct_correct(run, expected, category):
    """Tracked, not required: q07's percentage is derived from its two figures."""
    if category != "tabular" or "change_pct" not in expected:
        return None
    return value_present(answer_text(run), expected["change_pct"], "percent")


JUDGE_SYSTEM = (
    "You check whether an answer states a given fact. You are strict about "
    "substance and lenient about wording: a paraphrase counts, a different "
    "figure or a different subject does not. You reply with JSON only."
)

def build_judge_prompt(question, answer, claims):
    numbered = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, start=1))
    return (
        f"Question asked:\n{question}\n\n"
        f"Answer given:\n{answer}\n\n"
        f"Facts to check:\n{numbered}\n\n"
        f"For each fact, decide whether the answer states or clearly implies it. "
        f"Reply with only a JSON array of {len(claims)} booleans, in order, "
        f'like [true, false, true].'
    )

def parse_judgements(reply, count):
    """Strict JSON first, then a lenient scan, so one odd reply is not silently 0."""
    try:
        parsed = json.loads(reply.strip().strip("`"))
        if isinstance(parsed, list) and len(parsed) == count:
            return [bool(v) for v in parsed]
    except (json.JSONDecodeError, AttributeError):
        pass

    tokens = re.findall(r"\b(true|false)\b", reply or "", re.IGNORECASE)
    if len(tokens) == count:
        return [t.lower() == "true" for t in tokens]

    return None

def claims_score(run, expected, category, model=JUDGE_MODEL):
    """Fraction of expected claims the answer supports. The only LLM grader."""
    if category not in ("narrative", "transcript"):
        return None

    claims = expected.get("claims")
    if not claims:
        return None

    reply = call_llm(JUDGE_SYSTEM, build_judge_prompt(run.get("question", ""),
                                                      answer_text(run), claims), model=model)
    USAGE["judge_calls"] += 1
    USAGE["claims_judged"] += len(claims)

    judgements = parse_judgements(reply, len(claims))
    if judgements is None:
        return None

    return sum(judgements) / len(claims)


def declined(run):
    """
    An answer declines when it says the context does not support an answer.
    An answer that gives part of the figure and declines the rest still counts
    as declining what was asked.
    """
    lowered = answer_text(run).lower()
    if any(phrase in lowered for phrase in DECLINE_PHRASES):
        return True
    return not run.get("resolved_citations")

def decline_correct(run, expected, category):
    if category != "no_answer":
        return None
    return declined(run) == bool(expected.get("should_decline"))

def false_decline(run, category):
    """Declining a question that does have an answer in the corpus."""
    if category == "no_answer":
        return False
    return declined(run)


def grade(run, question_spec, judge=True, model=JUDGE_MODEL):
    """All applicable graders plus the single pass/fail call."""
    category = question_spec["category"]
    expected = question_spec.get("expected", {})
    evidence = question_spec.get("evidence", {})

    result = {
        "question_id": question_spec["id"],
        "category": category,
        "document_correct": document_correct(run, evidence),
        "period_correct": period_correct(run, evidence),
        "unit_correct": unit_correct(run, expected, category),
        "value_correct": value_correct(run, expected, category),
        "change_pct_correct": change_pct_correct(run, expected, category),
        "supplementary_values_correct": supplementary_values_correct(run, expected, category),
        "claims_score": claims_score(run, expected, category, model) if judge else None,
        "decline_correct": decline_correct(run, expected, category),
        "false_decline": false_decline(run, category),
        "declined": declined(run),
        "citations": len(run.get("resolved_citations", [])),
    }

    if category == "tabular":
        result["correct"] = bool(result["value_correct"]
                                 and result["document_correct"]
                                 and result["period_correct"])
    elif category in ("narrative", "transcript"):
        score = result["claims_score"]
        result["correct"] = bool(score is not None and score >= CLAIMS_PASS_MARK)
    elif category == "no_answer":
        result["correct"] = bool(result["decline_correct"])
    else:
        result["correct"] = False

    return result
