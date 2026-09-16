"""
Builds the v2 Chroma collection from the record-aware corpus.

Creates a NEW collection. The v1 collection is left untouched and stays
queryable; v2 gets its own BM25 cache.

Usage (from the project root):
    python src/build_v2_index.py              # counts and cost estimate only
    python src/build_v2_index.py --confirm    # embeds and indexes
"""
import argparse
import warnings
from collections import Counter

import chromadb

import embedder
from corpus import build_corpus
from indexer import COLLECTION, COLLECTION_V2, bm25_cache_path, build_bm25, build_index, save_bm25, load_chunks_from_chroma

warnings.filterwarnings("ignore")

PRICE_PER_MILLION = 0.02
CHROMA_DIR = "chroma_db"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the v2 index.")
    parser.add_argument("--collection", default=COLLECTION_V2)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--force", action="store_true", help="index even if the collection already holds chunks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = {c.name for c in client.list_collections()}
    print(f"existing collections: {sorted(existing)}")
    if COLLECTION not in existing:
        raise SystemExit(f"v1 collection {COLLECTION!r} is missing - refusing to proceed")

    if args.collection in existing:
        count = client.get_collection(args.collection).count()
        print(f"{args.collection} already exists with {count:,} chunks")
        if count and not args.force:
            raise SystemExit("refusing to add to a populated collection; pass --force if that is intended")

    print("\nbuilding corpus...")
    corpus = build_corpus()

    per_doc = Counter(c["id"].rsplit("_", 1)[0] for c in corpus)
    kinds = Counter(c["kind"] for c in corpus)
    chars = sum(len(c["text"]) for c in corpus)
    empty = [c["id"] for c in corpus if not c["text"]]

    print(f"\nchunks: {len(corpus):,}")
    for doc_id, count in per_doc.items():
        print(f"  {doc_id:<30} {count:>7,}")
    print(f"  kinds: {dict(kinds)}")
    print(f"  characters: {chars:,}   estimated tokens: {chars / 4:,.0f}")
    print(f"  estimated cost: ${chars / 4 / 1e6 * PRICE_PER_MILLION:.4f}")

    if empty:
        raise SystemExit(f"{len(empty)} chunks are empty strings, which the embeddings API rejects: {empty[:5]}")

    if not args.confirm:
        print("\nNo API calls made. Re-run with --confirm to embed and index.")
        raise SystemExit(1)

    print(f"\nindexing into {args.collection} (v1 collection untouched)...")
    build_index(corpus, persist_dir=CHROMA_DIR, collection_name=args.collection)

    tokens = embedder.USAGE["tokens"]
    print(f"\nACTUAL embedding usage: {tokens:,} tokens in {embedder.USAGE['calls']} calls")
    print(f"ACTUAL embedding cost:  ${tokens / 1e6 * PRICE_PER_MILLION:.4f}")

    print("\nbuilding BM25 over the v2 collection...")
    chunks = load_chunks_from_chroma(CHROMA_DIR, args.collection)
    bm25 = build_bm25(chunks)
    save_bm25(bm25, chunks, bm25_cache_path(args.collection))

    print(f"\nv1 {COLLECTION}: {client.get_collection(COLLECTION).count():,} chunks (unchanged)")
    print(f"v2 {args.collection}: {client.get_collection(args.collection).count():,} chunks")
