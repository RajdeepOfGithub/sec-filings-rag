from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query, results, top_n=5):
    if not results:
        return []

    model = get_model()
    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    scored = []
    for r, s in zip(results, scores):
        item = dict(r)
        item["rerank_score"] = float(s)
        scored.append(item)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    for i, item in enumerate(scored, start=1):
        item["rerank_rank"] = i

    return scored[:top_n]

