"""
Dense and sparse retrieval over the "financial_docs" collection.

Both functions return a list of dicts with the same shape:
    {"id", "text", "rank", "score", "score_type", "metadata"}

Scores are each retriever's native value, NOT normalized:
    dense  -> score_type "l2_distance"  (Chroma distance, lower is better)
    sparse -> score_type "bm25"         (BM25Okapi score, higher is better)

rank is 1-based within each result list.

filters is a simple equality dict over chunk metadata, e.g.
    {"form": "10-K", "period": "FY2025"}
Metadata values are as stored in Chroma (fixed-size chunks have section "none").
"""
import argparse

import chromadb

from embedder import embed_texts
from indexer import COLLECTION, load_bm25, tokenize

DEFAULT_COLLECTION = COLLECTION   # v1; pass collection= to query v2

# Loaded once per process and per collection, so a loop of queries doesn't
# reopen them every call and v1/v2 can both be queried in one session.
_collections = {}
_bm25 = {}


def get_collection(persist_dir="chroma_db", collection=DEFAULT_COLLECTION):
    key = (persist_dir, collection)
    if key not in _collections:
        client = chromadb.PersistentClient(path=persist_dir)
        _collections[key] = client.get_collection(name=collection)
    return _collections[key]

def get_bm25(rebuild=False, collection=DEFAULT_COLLECTION):
    if rebuild or collection not in _bm25:
        _bm25[collection] = load_bm25(rebuild=rebuild, collection_name=collection)
    return _bm25[collection]


def to_chroma_where(filters):
    if not filters:
        return None
    if len(filters) == 1:
        return dict(filters)
    return {"$and": [{key: value} for key, value in filters.items()]}

def matches_filters(metadata, filters):
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


def dense_search(query, n=20, filters=None, persist_dir="chroma_db",
                 collection=DEFAULT_COLLECTION):
    collection = get_collection(persist_dir, collection)

    qvec = embed_texts([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=n,
        where=to_chroma_where(filters),
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for i in range(len(res["ids"][0])):
        results.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "rank": i + 1,
            "score": res["distances"][0][i],
            "score_type": "l2_distance",
            "metadata": res["metadatas"][0][i],
        })
    return results

def sparse_search(query, n=20, filters=None, rebuild=False,
                  collection=DEFAULT_COLLECTION):
    bm25, chunks = get_bm25(rebuild=rebuild, collection=collection)
    scores = bm25.get_scores(tokenize(query))

    candidates = [
        i for i in range(len(chunks))
        if scores[i] > 0 and matches_filters(chunks[i]["metadata"], filters)
    ]
    top = sorted(candidates, key=lambda i: scores[i], reverse=True)[:n]

    results = []
    for rank, i in enumerate(top, start=1):
        results.append({
            "id": chunks[i]["id"],
            "text": chunks[i]["text"],
            "rank": rank,
            "score": float(scores[i]),
            "score_type": "bm25",
            "metadata": chunks[i]["metadata"],
        })
    return results


# CLI

COL_WIDTH = 62

def format_cell(result):
    if result is None:
        return ["", "", ""]
    meta = result["metadata"]
    snippet = " ".join(result["text"].split())
    return [
        f"#{result['rank']:<3} {result['id']}  {result['score_type']}={result['score']:.4f}",
        f"     {meta['form']} | {meta['period']} | {meta['section']}",
        f"     {snippet}",
    ]

def fit(line, width):
    if len(line) <= width:
        return line.ljust(width)
    return line[:width - 3] + "..."

def print_side_by_side(dense, sparse):
    print(fit("DENSE  (l2_distance, lower = better)", COL_WIDTH)
          + " | SPARSE (bm25, higher = better)")
    print("-" * COL_WIDTH + "-+-" + "-" * COL_WIDTH)

    for row in range(max(len(dense), len(sparse))):
        left = format_cell(dense[row] if row < len(dense) else None)
        right = format_cell(sparse[row] if row < len(sparse) else None)
        for l, r in zip(left, right):
            print(fit(l, COL_WIDTH) + " | " + fit(r, COL_WIDTH))
        print()

    print(f"dense returned {len(dense)}, sparse returned {len(sparse)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Dense vs sparse retrieval, side by side.")
    parser.add_argument("query")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--form", help='e.g. 10-K, 10-Q, earnings_call')
    parser.add_argument("--period", help='e.g. FY2025, Q2-2026')
    parser.add_argument("--rebuild-bm25", action="store_true")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    filters = {}
    if args.form:
        filters["form"] = args.form
    if args.period:
        filters["period"] = args.period

    dense = dense_search(args.query, n=args.n, filters=filters, collection=args.collection)
    sparse = sparse_search(args.query, n=args.n, filters=filters,
                           rebuild=args.rebuild_bm25, collection=args.collection)

    print(f"\nQ: {args.query}   filters={filters or None}\n")
    print_side_by_side(dense, sparse)
