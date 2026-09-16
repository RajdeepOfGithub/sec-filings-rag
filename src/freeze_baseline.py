"""
v1 baseline freeze: run every baseline question through the current pipeline
and capture the full intermediate state, plus the corpus, config and indexes.

Capture only. Nothing in the pipeline is modified, nothing is re-indexed.

run_question() mirrors pipeline.search() step by step, because search()
returns only the final top_k and the intermediate lists are the point of this
harness. If pipeline.search() changes, this must change with it.

Usage (from the project root):
    python src/freeze_baseline.py                  # cost estimate, then stops
    python src/freeze_baseline.py --confirm
    python src/freeze_baseline.py --confirm --ids q01
"""
import argparse
import inspect
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone

import chromadb

import embedder
import reranker
from chunker import fixed_size_chunk
from fusion import dedup_results, overlap_ratio, reciprocal_rank_fusion
from generator import (SYSTEM_PROMPT, build_context, call_llm, parse_citations,
                       resolve_citations)
from indexer import load_chunks_from_chroma
from reranker import rerank
from retriever import dense_search, get_collection, sparse_search

QUESTIONS_PATH = "data/baseline_questions.jsonl"
DEFAULT_OUT_DIR = "baseline/v1"
CHROMA_DIR = "chroma_db"
BM25_PATH = "data/processed/bm25.pkl"

N = 20
TOP_K = 5
GENERATION_MODEL = "gpt-4o-mini"
GENERATION_TEMPERATURE = 0  # hardcoded inside generator.call_llm

# USD per 1M tokens. Update if OpenAI pricing changes.
PRICE_EMBEDDING = 0.02
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60

# Rough per-question sizes for the estimate only.
EST_EMBEDDING_TOKENS = 30
EST_INPUT_TOKENS = 1100
EST_OUTPUT_TOKENS = 300


def read_questions(path=QUESTIONS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def estimate_cost(question_count):
    embedding = question_count * EST_EMBEDDING_TOKENS * PRICE_EMBEDDING / 1_000_000
    prompt = question_count * EST_INPUT_TOKENS * PRICE_INPUT / 1_000_000
    completion = question_count * EST_OUTPUT_TOKENS * PRICE_OUTPUT / 1_000_000
    return embedding + prompt + completion

def print_cost_estimate(question_count):
    total = estimate_cost(question_count)
    print(f"{question_count} questions x (1 query embedding + 1 {GENERATION_MODEL} call)")
    print(f"  assumes ~{EST_INPUT_TOKENS} input and ~{EST_OUTPUT_TOKENS} output tokens per question")
    print(f"  estimated cost: ${total:.4f}")
    print("  (estimate only - call_llm returns text, so actual token usage is not captured)")


# Capture

def label_found_in(found_in):
    """RRF reports list positions; we always pass [dense, sparse]."""
    names = {0: "dense", 1: "sparse"}
    return {names.get(idx, str(idx)): rank for idx, rank in found_in.items()}

def jsonable(value):
    """Chroma and numpy hand back scalar types json can't serialize."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value

def capture_retrieval_result(result):
    return {
        "id": result["id"],
        "rank": result["rank"],
        "score": jsonable(result["score"]),
        "score_type": result["score_type"],
        "metadata": result["metadata"],
        "text": result["text"],
    }

def capture_fused_result(result):
    return {
        "id": result["id"],
        "rank": result["rank"],
        "rrf_score": jsonable(result["rrf_score"]),
        "found_in": label_found_in(result["found_in"]),
        "metadata": result["metadata"],
        "text": result["text"],
    }

def capture_reranked_result(result):
    return {
        "id": result["id"],
        "rerank_rank": result["rerank_rank"],
        "rerank_score": jsonable(result["rerank_score"]),
        "rrf_rank": result["rank"],
        "rrf_score": jsonable(result["rrf_score"]),
        "found_in": label_found_in(result["found_in"]),
        "metadata": result["metadata"],
        "text": result["text"],
    }

def run_question(row, n=N, top_k=TOP_K, filters=None):
    """
    Mirrors pipeline.search(), keeping every intermediate list, then generates
    an answer. Returns the full record for one question.
    """
    question = row["question"]

    dense = dense_search(question, n=n, filters=filters)
    sparse = sparse_search(question, n=n, filters=filters)

    dense_kept, dense_dropped = dedup_results(dense)
    sparse_kept, sparse_dropped = dedup_results(sparse)

    fused = reciprocal_rank_fusion([dense_kept, sparse_kept], k=60)
    fused_kept, fused_dropped = dedup_results(fused)

    reranked = rerank(question, fused_kept, top_n=top_k)

    context, citation_map = build_context(reranked)
    user_prompt = f"{context}\n\nQuestion: {question}"
    answer_text = call_llm(SYSTEM_PROMPT, user_prompt, model=GENERATION_MODEL)

    numbers = parse_citations(answer_text)
    resolved = resolve_citations(numbers, citation_map, reranked)

    return {
        "question_id": row["id"],
        "question": question,
        "category": row["category"],
        "difficulty": row["difficulty"],
        "expected": row["expected"],
        "evidence": row["evidence"],
        "notes": row["notes"],
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n": n,
            "top_k": top_k,
            "filters": filters,
            "generation_model": GENERATION_MODEL,
            "generation_temperature": GENERATION_TEMPERATURE,
        },
        "dense": [capture_retrieval_result(r) for r in dense],
        "sparse": [capture_retrieval_result(r) for r in sparse],
        "dense_dedup_dropped": [list(pair) for pair in dense_dropped],
        "sparse_dedup_dropped": [list(pair) for pair in sparse_dropped],
        "fused": [capture_fused_result(r) for r in fused],
        "post_fusion_dedup_dropped": [list(pair) for pair in fused_dropped],
        "reranked": [capture_reranked_result(r) for r in reranked],
        "context": context,
        "user_prompt": user_prompt,
        "citation_map": citation_map,
        "answer_text": answer_text,
        "parsed_citations": numbers,
        "resolved_citations": resolved,
    }

def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def run_all_questions(rows, out_dir, n=N, top_k=TOP_K):
    """Writes each run file as it completes, so a later failure keeps earlier work."""
    runs_dir = os.path.join(out_dir, "runs")
    succeeded, failed = [], []

    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['id']} ({row['category']}): {row['question'][:60]}...")
        try:
            record = run_question(row, n=n, top_k=top_k)
            write_json(os.path.join(runs_dir, f"{row['id']}.json"), record)
            succeeded.append(row["id"])
            cited = [c["chunk_id"] for c in record["resolved_citations"]]
            print(f"      answered, cited {cited or '(none)'}")
        except Exception:
            failed.append(row["id"])
            write_json(os.path.join(runs_dir, f"{row['id']}.error.json"), {
                "question_id": row["id"],
                "question": row["question"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            })
            print(f"      FAILED, wrote {row['id']}.error.json")
            traceback.print_exc()

    return succeeded, failed


# Corpus snapshot, config, archive

def write_chunks_snapshot(out_dir, persist_dir=CHROMA_DIR):
    chunks = load_chunks_from_chroma(persist_dir)
    path = os.path.join(out_dir, "chunks.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"wrote {len(chunks)} chunks to {path}")
    return chunks

def git_provenance():
    def run(args):
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(run(["git", "status", "--porcelain"])),
        }
    except Exception as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}

def default_of(func, param):
    return inspect.signature(func).parameters[param].default

def document_breakdown(chunks):
    """Per-document chunk counts and strategy, read from what is actually indexed."""
    documents = {}
    for chunk in chunks:
        meta = chunk["metadata"]
        key = f"{meta['company']}_{meta['form']}_{meta['period']}"
        if key not in documents:
            documents[key] = {
                "company": meta["company"],
                "form": meta["form"],
                "period": meta["period"],
                "strategy": meta["strategy"],
                "chunks": 0,
            }
        documents[key]["chunks"] += 1
    return documents

def embedding_details(persist_dir=CHROMA_DIR):
    collection = get_collection(persist_dir)
    sample = collection.get(limit=1, include=["embeddings"])
    dimensions = len(sample["embeddings"][0])

    client = chromadb.PersistentClient(path=persist_dir)
    config = client.get_collection(name="financial_docs").configuration_json
    space = config.get("hnsw", {}).get("space")

    return {
        "model": embedder.MODEL,
        "dimensions": dimensions,
        "distance": space,
        "distance_note": "chroma hnsw 'l2' is squared L2; lower is better",
        "collection": "financial_docs",
    }

def write_config(out_dir, chunks):
    config = {
        "version": os.path.basename(out_dir.rstrip("/\\")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_provenance(),
        "chunking": {
            "chunk_size": default_of(fixed_size_chunk, "chunk_size"),
            "overlap": default_of(fixed_size_chunk, "overlap"),
            "documents": document_breakdown(chunks),
        },
        "embedding": embedding_details(),
        "retrieval": {
            "n": N,
            "top_k": TOP_K,
            "rrf_k": default_of(reciprocal_rank_fusion, "k"),
            "dedup_threshold": default_of(dedup_results, "threshold"),
            "dedup_shingle": default_of(overlap_ratio, "shingle"),
            "dedup_passes": ["dense", "sparse", "post_fusion"],
            "bm25_cache": BM25_PATH,
        },
        "reranker": {"model": reranker.MODEL_NAME},
        "generation": {
            "model": GENERATION_MODEL,
            "temperature": GENERATION_TEMPERATURE,
            "system_prompt": SYSTEM_PROMPT,
        },
        "corpus": {
            "total_chunks": len(chunks),
            "per_document": {k: v["chunks"] for k, v in document_breakdown(chunks).items()},
        },
    }

    path = os.path.join(out_dir, "config.json")
    write_json(path, config)
    print(f"wrote {path}")
    return config

def directory_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total / 1024 / 1024

def archive_indexes(out_dir):
    chroma_mb = directory_size_mb(CHROMA_DIR)
    bm25_mb = os.path.getsize(BM25_PATH) / 1024 / 1024
    print(f"archiving indexes: chroma_db {chroma_mb:.0f} MB, bm25.pkl {bm25_mb:.0f} MB")

    shutil.copytree(CHROMA_DIR, os.path.join(out_dir, "chroma_db"), dirs_exist_ok=True)
    shutil.copy2(BM25_PATH, os.path.join(out_dir, "bm25.pkl"))
    print(f"archived to {out_dir}/chroma_db and {out_dir}/bm25.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze the v1 baseline.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--ids", nargs="+", help="run only these question ids")
    parser.add_argument("--confirm", action="store_true", help="required; without it only the cost estimate prints")
    parser.add_argument("--skip-archive", action="store_true", help="skip copying chroma_db and bm25.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rows = read_questions(args.questions)
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids]

    print_cost_estimate(len(rows))
    if not args.confirm:
        print("\nNo API calls made. Re-run with --confirm to execute.")
        raise SystemExit(1)

    print(f"\nloading cross-encoder ({reranker.MODEL_NAME})...")
    reranker.get_model()

    succeeded, failed = run_all_questions(rows, args.out)

    chunks = write_chunks_snapshot(args.out)
    write_config(args.out, chunks)
    if args.skip_archive:
        print("skipped index archive (--skip-archive)")
    else:
        archive_indexes(args.out)

    print(f"\ndone: {len(succeeded)} succeeded, {len(failed)} failed -> {args.out}")
    if failed:
        print(f"failed ids: {failed}")
