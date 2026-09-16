"""
Intermediate representation for loaded documents.

A Document is an ordered list of Blocks. Identity and position are separate
concerns: block_id is derived from the block's own content, source_order
records where it sat in the document. A parser improvement that finds one
extra block must not renumber every block after it.

    block_id = doc_id + "/" + block_type + "/" + short_hash(normalized text)

Normalization is whitespace collapsing only. Case, punctuation and digits are
left alone, so 6,703 and 7,500 can never hash to the same id.

Dataclasses are kw_only so subclasses can add required fields without
colliding with the base class's defaulted section_id.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup

HASH_LENGTH = 12


def normalize(text):
    """The only normalization applied to identity text."""
    return " ".join(text.split()).strip()

def short_hash(normalized_text):
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:HASH_LENGTH]

def make_block_id(doc_id, block_type, normalized_text, occurrence=1):
    block_id = f"{doc_id}/{block_type}/{short_hash(normalized_text)}"
    if occurrence > 1:
        block_id = f"{block_id}#{occurrence}"
    return block_id


@dataclass(kw_only=True)
class Block:
    doc_id: str
    block_type: str
    source_order: int
    section_id: str | None = None
    block_id: str = ""  # filled in by assign_block_ids()

    def identity_text(self):
        raise NotImplementedError

    def serialized_text(self):
        raise NotImplementedError


@dataclass(kw_only=True)
class TextBlock(Block):
    text: str
    block_type: str = "text"

    def identity_text(self):
        return normalize(self.text)

    def serialized_text(self):
        return self.text


@dataclass(kw_only=True)
class TableBlock(Block):
    raw_html: str
    block_type: str = "table"

    def identity_text(self):
        return normalize(self.raw_html)

    def serialized_text(self):
        """Matches the current loader: get_text(separator=" ", strip=True)."""
        soup = BeautifulSoup(self.raw_html, "lxml")
        return soup.get_text(separator=" ", strip=True)


@dataclass(kw_only=True)
class SpeakerTurn(Block):
    speaker: str
    text: str
    turn_index: int
    block_type: str = "speaker_turn"

    def identity_text(self):
        """Speaker is part of identity: two people saying "Thank you." are
        different blocks, not one block with an ordinal."""
        return f"{normalize(self.speaker)}: {normalize(self.text)}"

    def serialized_text(self):
        return self.text


@dataclass(kw_only=True)
class HeadingBlock(Block):
    text: str
    level: int | None = None
    block_type: str = "heading"

    def identity_text(self):
        return normalize(self.text)

    def serialized_text(self):
        return self.text


@dataclass
class Document:
    doc_id: str
    source_path: str
    doc_type: str
    blocks: list


BLOCK_CLASSES = {
    "text": TextBlock,
    "table": TableBlock,
    "speaker_turn": SpeakerTurn,
    "heading": HeadingBlock,
}


def assign_block_ids(blocks):
    """
    Sets block_id on each block from its own content. Blocks whose identity
    text repeats get an occurrence ordinal: the first stays bare, the second
    gets #2, and so on.
    """
    counts = {}

    for block in sorted(blocks, key=lambda b: b.source_order):
        identity = block.identity_text()
        key = (block.block_type, identity)
        counts[key] = counts.get(key, 0) + 1
        block.block_id = make_block_id(block.doc_id, block.block_type, identity, counts[key])

    return blocks

def serialize_document(document):
    """
    Concatenates block text in source_order.

    There is deliberately no separator parameter: blocks carry their own text
    exactly as extracted, including any leading or trailing whitespace the
    loader produced. Whitespace semantics live in the loader, in one place.
    """
    ordered = sorted(document.blocks, key=lambda b: b.source_order)
    return "".join(block.serialized_text() for block in ordered)


# JSONL round-tripping

def to_jsonl(document, path):
    """First line is the document header, then one line per block."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        header = {
            "record": "document",
            "doc_id": document.doc_id,
            "source_path": document.source_path,
            "doc_type": document.doc_type,
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        for block in sorted(document.blocks, key=lambda b: b.source_order):
            row = {"record": "block", **asdict(block)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return path

def from_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    header = rows[0]
    blocks = []
    for row in rows[1:]:
        fields = {k: v for k, v in row.items() if k != "record"}
        blocks.append(BLOCK_CLASSES[fields["block_type"]](**fields))

    return Document(
        doc_id=header["doc_id"],
        source_path=header["source_path"],
        doc_type=header["doc_type"],
        blocks=blocks,
    )
