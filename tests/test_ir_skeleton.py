"""
Acceptance test for the IR skeleton.

The point of step 1: the IR must not change behaviour. For every JPMC
document, serialize_document(load_document_ir(...)) must equal
load_document(...) character for character.

No pytest in this venv, so this runs as a plain script from the project root:
    python tests/test_ir_skeleton.py
Exit code 0 means everything passed.
"""
import os
import sys
import tempfile

sys.path.insert(0, "src")

from ir import (Document, HeadingBlock, SpeakerTurn, TableBlock, TextBlock,
                assign_block_ids, from_jsonl, make_block_id, normalize,
                serialize_document, to_jsonl)
from loader import load_document, load_document_ir

DOCUMENTS = [
    ("data/raw/jpmc/jpmc_10k_2025.htm", "sec_filing", "JPMC_10-K_FY2025"),
    ("data/raw/jpmc/jpmc_10q_q2_2026.htm", "sec_filing", "JPMC_10-Q_Q2-2026"),
    ("data/raw/jpmc/jpmc_earnings_call_q2_2026.htm", "earnings_transcript", "JPMC_earnings_call_Q2-2026"),
]

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def test_serialization_is_identical():
    """THE acceptance test: IR serialization equals the old loader exactly."""
    print("\nserialize_document(load_document_ir(...)) == load_document(...)")

    for path, doc_type, doc_id in DOCUMENTS:
        old = load_document(path, doc_type)
        doc = load_document_ir(path, doc_type, doc_id)
        new = serialize_document(doc)

        name = os.path.basename(path)
        if old == new:
            check(name, True, f"{len(new):,} chars identical")
        else:
            first_diff = next((i for i in range(min(len(old), len(new))) if old[i] != new[i]), min(len(old), len(new)))
            check(name, False, f"old {len(old):,} vs new {len(new):,}, first difference at {first_diff}")


def test_block_ids_are_content_derived():
    print("\nblock_id is derived from content, not position")

    text = "Total net revenue 57,347"
    a = TextBlock(doc_id="d", source_order=0, text=text)
    b = TextBlock(doc_id="d", source_order=99, text=text)
    assign_block_ids([a])
    assign_block_ids([b])
    check("same text at different source_order gets the same id", a.block_id == b.block_id, a.block_id)

    different = TextBlock(doc_id="d", source_order=0, text="Total net revenue 57,348")
    assign_block_ids([different])
    check("one digit difference changes the id", different.block_id != a.block_id, different.block_id)

    spaced = TextBlock(doc_id="d", source_order=0, text="  Total net   revenue\n57,347 ")
    assign_block_ids([spaced])
    check("whitespace differences do not change the id", spaced.block_id == a.block_id)

    cased = TextBlock(doc_id="d", source_order=0, text="total net revenue 57,347")
    assign_block_ids([cased])
    check("case differences DO change the id (no lowercasing)", cased.block_id != a.block_id)

    other_doc = TextBlock(doc_id="other", source_order=0, text=text)
    assign_block_ids([other_doc])
    check("doc_id namespaces the id", other_doc.block_id != a.block_id, other_doc.block_id)

    check("id format is doc_id/block_type/hash", a.block_id == f"d/text/{a.block_id.split('/')[-1]}", a.block_id)


def test_inserting_a_block_does_not_shift_ids():
    """The reason identity is content-derived: a better parser must not renumber."""
    print("\ninserting a newly-detected block leaves other ids unchanged")

    def turns(texts):
        blocks = [SpeakerTurn(doc_id="call", source_order=i, speaker="Jeremy Barnum", text=t, turn_index=i)
                  for i, t in enumerate(texts)]
        return {b.text: b.block_id for b in assign_block_ids(blocks)}

    before = turns(["First answer.", "Second answer.", "Third answer."])
    after = turns(["First answer.", "MISSED TURN.", "Second answer.", "Third answer."])

    unchanged = [t for t in before if before[t] == after[t]]
    check("all pre-existing turn ids survive an insertion", len(unchanged) == 3,
          f"{len(unchanged)}/3 unchanged")


def test_speaker_is_part_of_turn_identity():
    print("\nspeaker is part of SpeakerTurn identity")

    operator = SpeakerTurn(doc_id="call", source_order=0, speaker="Operator", text="Thank you.", turn_index=0)
    barnum = SpeakerTurn(doc_id="call", source_order=1, speaker="Jeremy Barnum", text="Thank you.", turn_index=1)
    assign_block_ids([operator, barnum])

    check("different speakers, same words -> different ids", operator.block_id != barnum.block_id,
          f"{operator.block_id} vs {barnum.block_id}")
    check("neither gets an ordinal", "#" not in operator.block_id and "#" not in barnum.block_id)

    same_speaker_elsewhere = SpeakerTurn(doc_id="call", source_order=99, speaker="Operator",
                                         text="Thank you.", turn_index=99)
    assign_block_ids([same_speaker_elsewhere])
    check("same speaker and words -> same id regardless of position",
          same_speaker_elsewhere.block_id == operator.block_id)

    spaced_speaker = SpeakerTurn(doc_id="call", source_order=0, speaker="  Operator  ",
                                 text="Thank you.", turn_index=0)
    assign_block_ids([spaced_speaker])
    check("speaker whitespace is normalized", spaced_speaker.block_id == operator.block_id)


def test_duplicate_text_gets_occurrence_ordinal():
    print("\nidentical text from the same speaker gets an occurrence ordinal")

    blocks = [
        SpeakerTurn(doc_id="call", source_order=0, speaker="Operator", text="Thank you.", turn_index=0),
        SpeakerTurn(doc_id="call", source_order=1, speaker="Operator", text="Next question.", turn_index=1),
        SpeakerTurn(doc_id="call", source_order=2, speaker="Operator", text="Thank you.", turn_index=2),
        SpeakerTurn(doc_id="call", source_order=3, speaker="Operator", text="Thank you.", turn_index=3),
    ]
    assign_block_ids(blocks)
    ids = [b.block_id for b in blocks]

    check("ids are unique", len(set(ids)) == len(ids))
    check("first occurrence has no ordinal", not ids[0].endswith("#2") and "#" not in ids[0], ids[0])
    check("second occurrence ends #2", ids[2].endswith("#2"), ids[2])
    check("third occurrence ends #3", ids[3].endswith("#3"), ids[3])
    check("ordinals share the same content hash", ids[0] == ids[2].split("#")[0] == ids[3].split("#")[0])


def test_table_block_serializes_like_the_old_loader():
    print("\nTableBlock serializes via get_text(), matching current behaviour")

    raw = "<table><tr><th>2026</th><th>2025</th></tr><tr><td>21.7</td><td>29.8</td></tr></table>"
    block = TableBlock(doc_id="d", source_order=0, raw_html=raw)
    check("flattens to spaced text", block.serialized_text() == "2026 2025 21.7 29.8",
          repr(block.serialized_text()))

    other = TableBlock(doc_id="d", source_order=1, raw_html=raw.replace("21.7", "21.8"))
    assign_block_ids([block, other])
    check("different figures get different ids", block.block_id != other.block_id)


def test_jsonl_round_trip():
    print("\nto_jsonl / from_jsonl round-trips")

    doc = Document(
        doc_id="mixed",
        source_path="fake.htm",
        doc_type="sec_filing",
        blocks=assign_block_ids([
            HeadingBlock(doc_id="mixed", source_order=0, text="Capital Risk Management", level=2),
            TextBlock(doc_id="mixed", source_order=1, text="Shares repurchased were as follows:", section_id="s1"),
            TableBlock(doc_id="mixed", source_order=2, raw_html="<table><tr><td>21.7</td></tr></table>"),
            SpeakerTurn(doc_id="mixed", source_order=3, speaker="Jeremy Barnum", text="Thanks.", turn_index=0),
        ]),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = to_jsonl(doc, os.path.join(tmp, "doc.jsonl"))
        restored = from_jsonl(path)

    check("blocks survive", [type(b).__name__ for b in restored.blocks] ==
          ["HeadingBlock", "TextBlock", "TableBlock", "SpeakerTurn"])
    check("block ids survive", [b.block_id for b in restored.blocks] == [b.block_id for b in doc.blocks])
    check("subclass fields survive", restored.blocks[3].speaker == "Jeremy Barnum"
          and restored.blocks[0].level == 2 and restored.blocks[1].section_id == "s1")
    check("serialization survives", serialize_document(restored) == serialize_document(doc))


def test_real_document_round_trips_through_jsonl():
    print("\nreal document round-trips through JSONL without changing serialization")

    path, doc_type, doc_id = DOCUMENTS[2]
    doc = load_document_ir(path, doc_type, doc_id)

    with tempfile.TemporaryDirectory() as tmp:
        restored = from_jsonl(to_jsonl(doc, os.path.join(tmp, "call.jsonl")))

    check("serialization identical after round-trip",
          serialize_document(restored) == load_document(path, doc_type))
    check("section_id is None everywhere (step 2 fills it)",
          all(b.section_id is None for b in restored.blocks))
    check("doc metadata survives", (restored.doc_id, restored.source_path, restored.doc_type) ==
          (doc.doc_id, doc.source_path, doc.doc_type), restored.doc_id)


def test_doc_id_is_required_and_matches_chunk_ids():
    print("\ndoc_id is required and uses the corpus naming scheme")

    try:
        load_document_ir(DOCUMENTS[2][0], DOCUMENTS[2][1])
        check("omitting doc_id raises TypeError", False, "no error raised")
    except TypeError as exc:
        check("omitting doc_id raises TypeError", True, str(exc).split(":")[-1].strip())

    from corpus import make_doc_id
    doc_id = make_doc_id("JPMC", "10-Q", "Q2-2026")
    check("make_doc_id matches indexed chunk id prefix", doc_id == "JPMC_10-Q_Q2-2026", doc_id)

    block = TextBlock(doc_id=doc_id, source_order=0, text="Total net revenue 57,347")
    assign_block_ids([block])
    check("block ids are namespaced by the same doc_id",
          block.block_id.startswith("JPMC_10-Q_Q2-2026/"), block.block_id)


if __name__ == "__main__":
    test_serialization_is_identical()
    test_block_ids_are_content_derived()
    test_inserting_a_block_does_not_shift_ids()
    test_speaker_is_part_of_turn_identity()
    test_duplicate_text_gets_occurrence_ordinal()
    test_doc_id_is_required_and_matches_chunk_ids()
    test_table_block_serializes_like_the_old_loader()
    test_jsonl_round_trip()
    test_real_document_round_trips_through_jsonl()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("All checks passed.")
