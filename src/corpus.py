from loader import load_document_ir
from chunker import structure_aware_chunk_ir
from sections import attribute_sections_report


# Every document now goes through the same path: IR blocks, dynamic section
# attribution, record-aware chunking. The old per-document strategy labels
# ("structure_aware" / "fixed_size") no longer describe what happens.
STRATEGY = "record_aware"


def make_doc_id(company, form, period):
    """The one naming scheme: block ids and chunk ids both build on this."""
    return f"{company}_{form}_{period}"


def build_corpus(chunk_size=500, overlap=50):
    documents = [
        {
            "path": "data/raw/jpmc/jpmc_10k_2025.htm",
            "doc_type": "sec_filing",
            "company": "JPMC",
            "form": "10-K",
            "period": "FY2025",
        },
        {
            "path": "data/raw/jpmc/jpmc_10q_q2_2026.htm",
            "doc_type": "sec_filing",
            "company": "JPMC",
            "form": "10-Q",
            "period": "Q2-2026",
        },
        {
            "path": "data/raw/jpmc/jpmc_earnings_call_q2_2026.htm",
            "doc_type": "earnings_transcript",
            "company": "JPMC",
            "form": "earnings_call",
            "period": "Q2-2026",
        },
    ]

    corpus = []
    for doc in documents:
        doc_id = make_doc_id(doc["company"], doc["form"], doc["period"])
        document = load_document_ir(doc["path"], doc["doc_type"], doc_id)
        attributed, _ = attribute_sections_report(document, strict=False)
        pieces = structure_aware_chunk_ir(attributed, chunk_size, overlap)

        for i, piece in enumerate(pieces):
            corpus.append({
                "id": f"{doc_id}_{i}",
                "text": piece["text"],
                "company": doc["company"],
                "form": doc["form"],
                "period": doc["period"],
                "section": piece["section"],
                "strategy": STRATEGY,
                "kind": piece["kind"],
                "block_id": piece["block_id"],
            })

        kinds = {}
        for piece in pieces:
            kinds[piece["kind"]] = kinds.get(piece["kind"], 0) + 1
        print(f"{doc['form']}: {len(pieces)} chunks  {kinds}")

    return corpus


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"\nTotal chunks: {len(corpus)}")
    print(f"\nSample: {corpus[0]['id']}")
    print(f"  section: {corpus[0]['section']}")
    print(f"  text: {corpus[0]['text'][:100]}...")
