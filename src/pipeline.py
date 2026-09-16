"""
Full retrieval path:
    dense + sparse -> dedup each -> RRF -> dedup fused -> rerank -> top_k.

search() is the single entry point for downstream code (generation, eval).
Chroma collection, BM25 index and cross-encoder are cached per process
(see retriever.get_collection, retriever.get_bm25, reranker.get_model).
"""
from retriever import DEFAULT_COLLECTION, dense_search, sparse_search
from fusion import dedup_results, reciprocal_rank_fusion
from reranker import rerank


def search(query, n=20, top_k=5, filters=None, collection=DEFAULT_COLLECTION):
    dense = dense_search(query, n=n, filters=filters, collection=collection)
    sparse = sparse_search(query, n=n, filters=filters, collection=collection)

    dense_kept, _ = dedup_results(dense)
    sparse_kept, _ = dedup_results(sparse)

    fused = reciprocal_rank_fusion([dense_kept, sparse_kept], k=60)

    # Per-list passes can't see a near-copy that sits in the other list.
    fused_kept, fused_dropped = dedup_results(fused)
    for dropped_id, kept_id in fused_dropped:
        print(f"post-fusion dedup: dropped {dropped_id} (near-copy of {kept_id})")

    return rerank(query, fused_kept, top_n=top_k)
