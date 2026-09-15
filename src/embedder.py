import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "text-embedding-3-small"


def embed_texts(texts):
    response = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in response.data]

def embed_in_batches(texts, batch_size=100):
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)
        print(f" embedded {len(all_vectors)}/{len(texts)}")

    return all_vectors

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)

