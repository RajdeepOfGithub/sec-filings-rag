import chromadb
from embedder import embed_in_batches
from corpus import build_corpus
from rank_bm25 import BM25Okapi
import os
import pickle
import re

BM25_CACHE_PATH = "data/processed/bm25.pkl"
COLLECTION = "financial_docs"          # v1, still live and queryable
COLLECTION_V2 = "financial_docs_v2"
ADD_BATCH_SIZE = 2000


def bm25_cache_path(collection_name):
    """v1 keeps its original pickle path so its cache stays valid."""
    if collection_name == COLLECTION:
        return BM25_CACHE_PATH
    return f"data/processed/bm25_{collection_name}.pkl"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "much", "of", "on", "or", "our", "s", "so", "ت",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
}


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def build_index(corpus, persist_dir="chroma_db", collection_name=COLLECTION):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    texts = [c["text"] for c in corpus]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embed_in_batches(texts)

    metadatas = [
        {
            "company": c["company"],
            "form": c["form"],
            "period": c["period"],
            "section": c["section"] or "none",
            "strategy": c["strategy"],
            "kind": c.get("kind", "narrative"),
            "block_id": c.get("block_id") or "none",
        }
        for c in corpus
    ]

    # Chroma caps a single add() at 5,461 records.
    for start in range(0, len(corpus), ADD_BATCH_SIZE):
        stop = start + ADD_BATCH_SIZE
        collection.add(
            ids=[c["id"] for c in corpus[start:stop]],
            embeddings=vectors[start:stop],
            documents=texts[start:stop],
            metadatas=metadatas[start:stop],
        )
        print(f" added {min(stop, len(corpus))}/{len(corpus)}")

    print(f"Collection now holds {collection.count()} chunks")
    return collection

def test_query(question, n=3, persist_dir="chroma_db"):
    from embedder import embed_texts

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name="financial_docs")

    qvec = embed_texts([question])[0]
    results = collection.query(query_embeddings=[qvec], n_results=n)

    print(f"\nQ: {question}\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"[{i}] dist={dist:.4f}  {meta['form']} / {meta['section']}")
        print(f"    {results['documents'][0][i][:180]}...\n")



#BM25 Indexing

def build_bm25(corpus):
    tokenized = [tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built over {len(tokenized)} chunks")
    return bm25

def bm25_search(bm25, corpus, question, n=3):
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    print(f"\nQ: {question}\n")
    for rank, i in enumerate(ranked):
        c = corpus[i]
        print(f"[{rank}] score={scores[i]:.4f}  {c['form']} / {c['section']}")
        print(f"    {c['text'][:180]}...\n")

    return ranked

def load_chunks_from_chroma(persist_dir="chroma_db", collection_name=COLLECTION):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    got = collection.get(include=["documents", "metadatas"])

    return [
        {"id": i, "text": text, "metadata": meta}
        for i, text, meta in zip(got["ids"], got["documents"], got["metadatas"])
    ]

def save_bm25(bm25, chunks, path=BM25_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 cache written to {path}")

def load_bm25(rebuild=False, path=None, persist_dir="chroma_db", collection_name=COLLECTION):
    """
    Returns (bm25, chunks). chunks[i] lines up with bm25 score i.
    Built from the Chroma collection's stored text, never from raw files.
    """
    path = path or bm25_cache_path(collection_name)

    if rebuild or not os.path.exists(path):
        chunks = load_chunks_from_chroma(persist_dir, collection_name)
        bm25 = build_bm25(chunks)
        save_bm25(bm25, chunks, path)
        return bm25, chunks

    with open(path, "rb") as f:
        cached = pickle.load(f)

    client = chromadb.PersistentClient(path=persist_dir)
    chroma_count = client.get_collection(name=collection_name).count()
    if len(cached["chunks"]) != chroma_count:
        raise RuntimeError(
            f"Stale BM25 cache: {path} has {len(cached['chunks'])} chunks, "
            f"Chroma has {chroma_count}. Rebuild with rebuild=True."
        )

    return cached["bm25"], cached["chunks"]

def explain_match(bm25, corpus, question, doc_idx):
    q_tokens = tokenize(question)
    doc_tokens = tokenize(corpus[doc_idx]["text"])

    print(f"\nQuery tokens: {q_tokens}")
    print(f"Overlap with chunk {doc_idx}:")
    for t in q_tokens:
        count = doc_tokens.count(t)
        if count:
            idf = bm25.idf.get(t, 0)
            print(f"  '{t}'  appears {count}x  idf={idf:.3f}")


#De duplication function
def dedup_corpus(corpus, window=200):
    seen = {}
    kept = []
    dropped = 0

    for c in corpus:
        key = re.sub(r"\s+", " ", c["text"][:window]).strip().lower()

        if key in seen:
            dropped += 1
            continue

        seen[key] = c["id"]
        kept.append(c)

    print(f"Dedup: kept {len(kept)}, dropped {dropped}")
    return kept


if __name__ == "__main__":
    corpus = build_corpus()

    from embedder import embed_texts
    import chromadb

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="financial_docs")

    q = "How much did the firm spend on share repurchases?"
    qvec = embed_texts([q])[0]
    res = collection.query(query_embeddings=[qvec], n_results=3)

    for i in range(3):
        print(f"\n[{i}] id={res['ids'][0][i]}")
        print(res["documents"][0][i][:300])
