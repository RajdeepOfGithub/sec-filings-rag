"""
Dynamic section detection over a block stream.

Replaces hardcoded character offsets. The offsets were positions in one
particular serialized output; any change to extraction silently repoints them
at different text, producing confident wrong labels. A wrong label is worse
than a crash, so this module prefers "section_unknown" and loud diagnostics
over a plausible guess.

Approach: find candidate headings, group them into monotonic runs, drop runs
that are packed too densely to be real sections (a table of contents is a
compressed replica of the section sequence), then require what remains to be
monotonic by item number.

Nothing here is company-specific: no offsets, no "Item 9B is last", no
assumptions about which items a filing contains.
"""
import re
from dataclasses import dataclass, replace

from ir import Document, TextBlock, assign_block_ids

FRONT_MATTER = "front_matter"
SECTION_UNKNOWN = "section_unknown"

@dataclass(frozen=True)
class ItemSpec:
    title: str   # shown in section_id, so it has to read well in an LLM prompt
    match: str   # prefix required after the item number when detecting


# SEC-mandated Form 10-K items. The match prefix is kept short enough to
# survive filer-specific title wording, long enough to reject a bare
# cross-reference like "Refer to Item 10."
ITEM_TITLES = {
    "1": ItemSpec("Business", "Business"),
    "1A": ItemSpec("Risk Factors", "Risk Factors"),
    "1B": ItemSpec("Unresolved Staff Comments", "Unresolved Staff Comments"),
    "1C": ItemSpec("Cybersecurity", "Cybersecurity"),
    "2": ItemSpec("Properties", "Properties"),
    "3": ItemSpec("Legal Proceedings", "Legal Proceedings"),
    "4": ItemSpec("Mine Safety Disclosures", "Mine Safety"),
    "5": ItemSpec("Market for Registrant's Common Equity", "Market for Registrant"),
    "6": ItemSpec("Reserved", "Reserved"),
    "7": ItemSpec("Management's Discussion and Analysis", "Management"),
    "7A": ItemSpec("Quantitative and Qualitative Disclosures About Market Risk",
                   "Quantitative and Qualitative"),
    "8": ItemSpec("Financial Statements and Supplementary Data", "Financial Statements"),
    "9": ItemSpec("Changes in and Disagreements with Accountants", "Changes in and Disagreements"),
    "9A": ItemSpec("Controls and Procedures", "Controls and Procedures"),
    "9B": ItemSpec("Other Information", "Other Information"),
    "9C": ItemSpec("Disclosure Regarding Foreign Jurisdictions", "Disclosure Regarding Foreign"),
    "10": ItemSpec("Directors, Executive Officers and Corporate Governance", "Directors"),
    "11": ItemSpec("Executive Compensation", "Executive Compensation"),
    "12": ItemSpec("Security Ownership of Certain Beneficial Owners", "Security Ownership"),
    "13": ItemSpec("Certain Relationships and Related Transactions", "Certain Relationships"),
    # filers write "Principal Accountant Fees" or "Principal Accounting Fees"
    "14": ItemSpec("Principal Accountant Fees and Services", "Principal Account"),
    "15": ItemSpec("Exhibits, Financial Statement Schedules", "Exhibit"),
}

# A run whose chars-per-candidate is this far below the widest run's is a
# table of contents, not a sequence of real sections. Relative, not absolute.
TOC_DENSITY_RATIO = 0.1
MIN_TOC_RUN = 3

# The final section has a heading but no closing evidence. If it runs this
# much longer than a typical detected section, its extent is an assumption
# rather than a measurement, so it is labelled section_unknown.
UNBOUNDED_TAIL_RATIO = 10

# A section that swallows most of the document was never interrupted by
# another heading, so its extent rests on one match rather than on evidence
# at both ends. This catches an item table applied to the wrong form: a 10-Q
# contains "Item 1A. Risk Factors" in Part II, and without this guard that
# one match labels the entire filing.
DOMINANT_SECTION_SHARE = 0.5


class SectionDetectionError(Exception):
    """Raised when detected headings cannot form a valid section sequence."""


@dataclass
class Diagnostic:
    severity: str   # "warning" | "error"
    code: str
    message: str

    def __str__(self):
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass
class Candidate:
    item_number: str
    start: int
    end: int
    matched_text: str


@dataclass
class SectionBoundary:
    section_id: str
    item_number: str
    title: str
    start: int
    end: int
    confidence: str = "high"


def item_sort_key(item_number):
    """"1A" sorts after "1" and before "1B"; "10" sorts after "9C"."""
    match = re.match(r"(\d+)([A-Z]*)", item_number.upper())
    return int(match.group(1)), match.group(2)

def section_id_for(item_number, title):
    return f"Item {item_number} - {title}"

def loose_pattern(literal):
    """
    Tolerates get_text() spacing artifacts: extraction produces things like
    "Item 1 C . C ybersecurity", so whitespace is optional between characters.
    """
    return r"\s*".join(re.escape(ch) for ch in literal if not ch.isspace())


def find_candidates(text, item_titles=ITEM_TITLES):
    """
    A candidate is an item number followed by that item's mandated title.
    Requiring the title filters out bare cross-references ("Refer to Item
    10."), and allowing only "." or whitespace between number and title
    filters out the colon form used by cross-references.
    """
    candidates = []

    for item_number, spec in item_titles.items():
        pattern = re.compile(
            loose_pattern("Item") + r"\s*" + loose_pattern(item_number) + r"[\s.]*" + loose_pattern(spec.match),
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            candidates.append(Candidate(item_number, match.start(), match.end(), match.group()))

    candidates.sort(key=lambda c: c.start)
    return candidates

def group_into_runs(candidates):
    """Splits the candidate list wherever the item sequence restarts."""
    runs = []
    current = []

    for candidate in candidates:
        if current and item_sort_key(candidate.item_number) <= item_sort_key(current[-1].item_number):
            runs.append(current)
            current = []
        current.append(candidate)

    if current:
        runs.append(current)
    return runs

def run_density(run):
    """Mean characters between candidates: a table of contents packs many
    headings into almost no text."""
    if len(run) < 2:
        return float("inf")
    span = run[-1].start - run[0].start
    return span / (len(run) - 1)

def drop_toc_runs(runs, diagnostics):
    """
    Discards dense runs, but only when their items also appear in a run that
    survives; a filing whose only copy of the sequence is dense keeps it,
    with a warning, rather than losing its sections entirely.
    """
    if len(runs) < 2:
        return runs

    densest = max(run_density(run) for run in runs)
    kept, dropped = [], []

    for run in runs:
        is_dense = (len(run) >= MIN_TOC_RUN and run_density(run) < TOC_DENSITY_RATIO * densest)
        (dropped if is_dense else kept).append(run)

    if not kept:
        diagnostics.append(Diagnostic("warning", "toc_only",
                                      "every candidate run looks like a table of contents; keeping all"))
        return runs

    kept_items = {c.item_number for run in kept for c in run}
    for run in dropped:
        missing = sorted({c.item_number for c in run} - kept_items, key=item_sort_key)
        diagnostics.append(Diagnostic(
            "warning", "toc_discarded",
            f"discarded dense run of {len(run)} candidates at {run[0].start}-{run[-1].start} "
            f"({run_density(run):.0f} chars/heading vs {densest:.0f} in the widest run)"))
        if missing:
            diagnostics.append(Diagnostic(
                "warning", "toc_only_items",
                f"items {missing} appear only in the discarded run and were lost"))

    return kept

def longest_monotonic(candidates):
    """Longest strictly increasing subsequence by item number, ties to the wider span."""
    if not candidates:
        return []

    best = [[c] for c in candidates]
    for i, candidate in enumerate(candidates):
        for j in range(i):
            if item_sort_key(candidates[j].item_number) < item_sort_key(candidate.item_number):
                if len(best[j]) + 1 > len(best[i]):
                    best[i] = best[j] + [candidate]

    return max(best, key=lambda seq: (len(seq), seq[-1].start - seq[0].start))

def enforce_monotonic(candidates, diagnostics, strict):
    kept = longest_monotonic(candidates)
    discarded = [c for c in candidates if c not in kept]

    if discarded:
        detail = ", ".join(f"Item {c.item_number} at {c.start}" for c in discarded)
        message = f"candidates out of sequence and discarded: {detail}"
        if strict:
            raise SectionDetectionError(message)
        diagnostics.append(Diagnostic("error", "non_monotonic", message))

    return kept

def warn_about_gaps(candidates, diagnostics, item_titles):
    seen = {}
    for candidate in candidates:
        seen.setdefault(candidate.item_number, []).append(candidate.start)

    for item_number, offsets in seen.items():
        if len(offsets) > 1:
            diagnostics.append(Diagnostic("warning", "duplicate_item",
                                          f"Item {item_number} detected {len(offsets)} times at {offsets}"))

    missing = sorted(set(item_titles) - set(seen), key=item_sort_key)
    if missing:
        diagnostics.append(Diagnostic("warning", "missing_items",
                                      f"no heading found for items {missing}"))

def build_boundaries(candidates, text_length, diagnostics):
    boundaries = []

    for i, candidate in enumerate(candidates):
        end = candidates[i + 1].start if i + 1 < len(candidates) else text_length
        title = ITEM_TITLES[candidate.item_number].title
        boundaries.append(SectionBoundary(
            section_id=section_id_for(candidate.item_number, title),
            item_number=candidate.item_number,
            title=title,
            start=candidate.start,
            end=end,
        ))

    flag_unbounded_tail(boundaries, diagnostics)
    flag_dominant_sections(boundaries, text_length, diagnostics)
    return boundaries

def flag_dominant_sections(boundaries, text_length, diagnostics):
    """Relabels any section covering most of the document (see DOMINANT_SECTION_SHARE)."""
    for boundary in boundaries:
        if boundary.section_id == SECTION_UNKNOWN or not text_length:
            continue

        share = (boundary.end - boundary.start) / text_length
        if share > DOMINANT_SECTION_SHARE:
            diagnostics.append(Diagnostic(
                "warning", "dominant_section",
                f"{boundary.section_id} would cover {share:.0%} of the document with no "
                f"intervening heading; labelling it {SECTION_UNKNOWN} instead"))
            boundary.section_id = SECTION_UNKNOWN
            boundary.confidence = "low"

def flag_unbounded_tail(boundaries, diagnostics):
    """
    Every section but the last is closed by the next heading. The last one is
    closed by the end of the document, which is an assumption: filings can
    append material that belongs to no item. If that tail is wildly longer
    than the measured sections, it is relabelled section_unknown.
    """
    if len(boundaries) < 2:
        return

    last = boundaries[-1]
    bounded_lengths = sorted(b.end - b.start for b in boundaries[:-1])
    median = bounded_lengths[len(bounded_lengths) // 2]
    tail_length = last.end - last.start

    if median > 0 and tail_length > UNBOUNDED_TAIL_RATIO * median:
        diagnostics.append(Diagnostic(
            "warning", "unbounded_tail",
            f"final section {last.section_id} would run {tail_length:,} chars "
            f"({tail_length // median}x the median section of {median:,}); "
            f"labelling it {SECTION_UNKNOWN} instead"))
        last.section_id = SECTION_UNKNOWN
        last.confidence = "low"


def detect_sections_report(blocks, strict=True, item_titles=ITEM_TITLES):
    """detect_sections() plus the diagnostics collected on the way."""
    text = "".join(block.serialized_text() for block in sorted(blocks, key=lambda b: b.source_order))
    diagnostics = []

    candidates = find_candidates(text, item_titles)
    if not candidates:
        diagnostics.append(Diagnostic("warning", "no_candidates", "no item headings detected"))
        return [], diagnostics

    runs = group_into_runs(candidates)
    kept_runs = drop_toc_runs(runs, diagnostics)
    kept = [c for run in kept_runs for c in run]
    kept = enforce_monotonic(kept, diagnostics, strict)

    warn_about_gaps(kept, diagnostics, item_titles)
    return build_boundaries(kept, len(text), diagnostics), diagnostics

def detect_sections(blocks, strict=True, item_titles=ITEM_TITLES):
    boundaries, diagnostics = detect_sections_report(blocks, strict=strict, item_titles=item_titles)
    for diagnostic in diagnostics:
        print(diagnostic)
    return boundaries


# Attribution

def section_id_at(offset, boundaries):
    """Text before the first heading is front matter (cover page, TOC)."""
    if not boundaries:
        return SECTION_UNKNOWN
    if offset < boundaries[0].start:
        return FRONT_MATTER

    for boundary in boundaries:
        if boundary.start <= offset < boundary.end:
            return boundary.section_id
    return SECTION_UNKNOWN

def block_offsets(blocks):
    """(block, start, end) in serialized order."""
    spans = []
    offset = 0
    for block in sorted(blocks, key=lambda b: b.source_order):
        length = len(block.serialized_text())
        spans.append((block, offset, offset + length))
        offset += length
    return spans

def split_points_within(start, end, boundaries):
    return [b.start for b in boundaries if start < b.start < end]

def attribute_sections_report(document, strict=True):
    """
    Attaches section_id to every block. A block that spans a boundary has no
    single correct label, so TextBlocks are split at the boundary; splitting
    is exact, and serialize_document() is unchanged by it. Blocks that cannot
    be split (tables, speaker turns) keep the label of their first character
    and raise a diagnostic.
    """
    boundaries, diagnostics = detect_sections_report(document.blocks, strict=strict)
    attributed = []

    for block, start, end in block_offsets(document.blocks):
        cuts = split_points_within(start, end, boundaries)

        if not cuts:
            attributed.append(replace(block, section_id=section_id_at(start, boundaries)))
            continue

        if not isinstance(block, TextBlock):
            diagnostics.append(Diagnostic(
                "warning", "block_spans_boundary",
                f"{type(block).__name__} {block.block_id} spans {len(cuts)} section boundary/ies; "
                f"labelled by its first character"))
            attributed.append(replace(block, section_id=section_id_at(start, boundaries)))
            continue

        for piece_start, piece_end in zip([start] + cuts, cuts + [end]):
            attributed.append(replace(
                block,
                text=block.text[piece_start - start:piece_end - start],
                section_id=section_id_at(piece_start, boundaries),
            ))

    for order, block in enumerate(attributed):
        block.source_order = order
    assign_block_ids(attributed)

    return Document(
        doc_id=document.doc_id,
        source_path=document.source_path,
        doc_type=document.doc_type,
        blocks=attributed,
    ), diagnostics

def attribute_sections(document, strict=True):
    attributed, diagnostics = attribute_sections_report(document, strict=strict)
    for diagnostic in diagnostics:
        print(diagnostic)
    return attributed
