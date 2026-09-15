from loader import load_document
from chunker import structure_aware_chunk, fixed_size_chunk


def build_corpus():
    documents = [
        {
            "path": "data/raw/jpmc/jpmc_10k_2025.htm",
            "doc_type": "sec_filing",
            "company": "JPMC",
            "form": "10-K",
            "period": "FY2025",
            "strategy": "structure_aware",
        },
        {
            "path": "data/raw/jpmc/jpmc_10q_q2_2026.htm",
            "doc_type": "sec_filing",
            "company": "JPMC",
            "form": "10-Q",
            "period": "Q2-2026",
            "strategy": "fixed_size",
        },
        {
            "path": "data/raw/jpmc/jpmc_earnings_call_q2_2026.htm",
            "doc_type": "earnings_transcript",
            "company": "JPMC",
            "form": "earnings_call",
            "period": "Q2-2026",
            "strategy": "fixed_size",
        },
    ]

    corpus = []
    for doc in documents:
        text = load_document(doc["path"], doc["doc_type"])

        if doc["strategy"] == "structure_aware":
            pieces = structure_aware_chunk(text)
        else:
            pieces = [{"text": t, "section": None, "chunk_index_in_section": i}
                      for i, t in enumerate(fixed_size_chunk(text))]

        for i, piece in enumerate(pieces):
            corpus.append({
                "id": f"{doc['company']}_{doc['form']}_{doc['period']}_{i}",
                "text": piece["text"],
                "company": doc["company"],
                "form": doc["form"],
                "period": doc["period"],
                "section": piece["section"],
                "strategy": doc["strategy"],
            })

        print(f"{doc['form']}: {len(pieces)} chunks")

    return corpus


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"\nTotal chunks: {len(corpus)}")
    print(f"\nSample: {corpus[0]['id']}")
    print(f"  section: {corpus[0]['section']}")
    print(f"  text: {corpus[0]['text'][:100]}...")
