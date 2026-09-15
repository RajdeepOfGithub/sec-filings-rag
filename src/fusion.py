from retriever import dense_search, sparse_search

def overlap_ratio(a, b, shingle=40):
    a_norm = " ".join(a.split()).lower()
    b_norm = " ".join(b.split()).lower()

    a_shingles = {a_norm[i:i+shingle] for i in range(len(a_norm) - shingle + 1)}
    b_shingles = {b_norm[i:i+shingle] for i in range(len(b_norm) - shingle + 1)}

    if not a_shingles or not b_shingles:
        return 0.0
    return len(a_shingles & b_shingles) / min(len(a_shingles), len(b_shingles))

def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    seen = {}
    contributions = {}

    for list_idx, results in enumerate(result_lists):
        for r in results:
            cid = r["id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + r["rank"])
            seen.setdefault(cid, {})[list_idx] = r["rank"]
            contributions.setdefault(cid, []).append(
                (list_idx, r["rank"], 1 / (k + r["rank"]))
            )
            if cid not in seen or "text" not in seen[cid]:
                pass

    lookup = {}
    for results in result_lists:
        for r in results:
            lookup[r["id"]] = r

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    out = []
    for rank, (cid, score) in enumerate(fused, start=1):
        out.append({
            "id": cid,
            "text": lookup[cid]["text"],
            "metadata": lookup[cid]["metadata"],
            "rank": rank,
            "rrf_score": score,
            "found_in": seen[cid],
        })
    return out


def dedup_results(results, threshold=0.85):
    kept = []
    dropped = []

    for r in results:
        is_dup = False
        for k in kept:
            if overlap_ratio(r["text"], k["text"]) >= threshold:
                dropped.append((r["id"], k["id"]))
                is_dup = True
                break
        if not is_dup:
            kept.append(r)

    # Re-rank copies so callers keep the original ranks on their own dicts.
    kept = [dict(r, rank=i) for i, r in enumerate(kept, start=1)]

    return kept, dropped



if __name__ == "__main__":
    from reranker import rerank

    q = "How much did the firm spend on share repurchases?"

    dense = dense_search(q, n=20)
    sparse = sparse_search(q, n=20)
    dense_d, _ = dedup_results(dense)
    sparse_d, _ = dedup_results(sparse)
    fused = reciprocal_rank_fusion([dense_d, sparse_d], k=60)

    reranked = rerank(q, fused, top_n=8)

    print(f"\nQ: {q}\n")
    print("rerank | was | score   | chunk")
    print("-" * 60)
    for r in reranked:
        print(f"  #{r['rerank_rank']:<4} | #{r['rank']:<3} | {r['rerank_score']:7.3f} | {r['id']}")

    print("\n" + "=" * 70)
    for r in reranked[:5]:
        print(f"\n--- #{r['rerank_rank']}  {r['id']}  score={r['rerank_score']:.3f} ---")
        print(" ".join(r["text"].split()))