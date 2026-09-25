#!/usr/bin/env python3
"""
Smoke test for the RAG ingestion pipeline (Steps 9-10 in README.md): runs
a handful of sample financial questions through
src/retrieval/vector_store.similarity_search() and prints what comes back,
so you can eyeball whether retrieval is actually finding relevant text -
not just whether the row count in pgvector matches (that only proves
nothing was lost/duplicated during loading, not that the content is any
good for answering questions).

This does NOT call an LLM - it only tests the retrieval half (embed query
-> find nearest chunks by cosine similarity). Answer generation isn't
built yet (src/models/, src/retrieval/rag_pipeline.py).

Run:
  python3 scripts/test_retrieval.py
  python3 scripts/test_retrieval.py "câu hỏi tuỳ ý của bạn"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.vector_store import similarity_search

DEFAULT_QUERIES = [
    "rủi ro pháp lý của công ty",
    "chiến lược kinh doanh năm 2025",
    "tổng tài sản và nợ phải trả",
    "kết quả kinh doanh quý",
]


def run_query(query: str, top_k: int = 3):
    print(f"\n=== \"{query}\" ===")
    results = similarity_search(query, top_k=top_k)
    if not results:
        print("  (no results - is disclosure_chunks empty? check the COUNT(*) from README Step 10)")
        return
    for r in results:
        preview = r["chunk_text"][:100].replace("\n", " ")
        print(f"  [{r['similarity']:.3f}] {r['ticker']} / {r['doctype']} / p{r['page']} - {preview}...")


def main():
    queries = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES
    print(f"Running {len(queries)} test quer{'y' if len(queries) == 1 else 'ies'} against disclosure_chunks...")
    for q in queries:
        run_query(q)

    print(
        "\nWhat to look for: similarity scores usually fall roughly in the "
        "0.3-0.7 range for real matches with bge-m3 (1.0 = identical text, "
        "0 = unrelated) - there's no fixed 'good' threshold, judge by "
        "whether the printed chunk text actually relates to the query. "
        "If every query returns the same handful of chunks regardless of "
        "topic, or scores are all clustered near 0, something's off with "
        "the embeddings and is worth flagging before building the Router "
        "on top of this."
    )


if __name__ == "__main__":
    main()
