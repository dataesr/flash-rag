import argparse
from typing import Literal, Optional
from chromadb import Knn, Rrf, Search, K
from datetime import datetime
from src.chromadb import get_collection
from src.bm25 import bm25_search, rrf_fusion

MAX_TIMESTAMP = datetime((datetime.now().year - 4), 1, 1).timestamp()  # 3 years ago + 1 year buffer
MIN_CHUNK_LEN = 50  # Minimum chunk length to consider for querying
MAX_K = 50  # Max docs to retrieves
K_MULTIPLIER = 5  # Multiplier for candidate retrieval before RRF and reranking


def lightweight_rerank(query_text: str, sources: list) -> list:
    """
    Combine dense + sparse signals without ML models.
    Zero dependencies, microsecond latency.
    """
    query_terms = set(query_text.lower().split())

    for source in sources:
        text_lower = source["document"].lower()

        # Signal 1: How many query terms exact match?
        term_match = len([t for t in query_terms if t in text_lower]) / len(query_terms)

        # Signal 2: Is query phrase found contiguously?
        exact_phrase = 1.0 if query_text.lower() in text_lower else 0.0

        # Signal 3: Chunk length penalty (shorter = less fluff)
        length_penalty = min(1.0, 300 / len(text_lower))

        # Combine signals
        rerank_score = (term_match * 0.5) + (exact_phrase * 0.35) + (length_penalty * 0.15)
        source["rerank_score"] = rerank_score

    return sorted(sources, key=lambda x: x["rerank_score"], reverse=True)


def query(
    query_text: str,
    source: Literal["all", "eesr", "ssmesr"] = "all",
    k: int = 5,
    use_reranker: bool = True,
    use_hybrid_search: bool = True,
) -> tuple:
    """
    Query the RAG collection with optional hybrid search (dense + BM25) and reranking.

    Args:
        query_text: The query string
        source: Which source to query from
        k: Number of final results to return (1-50)
        use_reranker: Whether to use CrossEncoder reranking
        use_hybrid_search: Whether to combine vector + BM25 search (RRF fusion)

    Returns:
        (answer, sources)
    """

    # Validate k
    k = max(1, min(k, MAX_K))

    # Get collection
    collection = get_collection()

    # Build filters
    where_filter = {
        "$and": [
            {"publication_epoch": {"$gte": MAX_TIMESTAMP}},
            {"chunk_len": {"$gte": MIN_CHUNK_LEN}},
        ]
    }
    if source != "all":
        where_filter["$and"].append({"source": {"$eq": source}})  # ty: ignore[invalid-argument-type]

    # Retrieve more candidates than the final k because
    # RRF + reranking need a larger candidate pool
    retrieval_k = min(k * K_MULTIPLIER, MAX_K) if (use_hybrid_search or use_reranker) else k

    print(f"[search] k={k}, retrieval_k={retrieval_k}, " f"hybrid={use_hybrid_search}, reranker={use_reranker}")

    # ========== DENSE SEARCH (Vector/Mistral) ==========
    print(f"[search] Dense search: retrieving top {retrieval_k}")
    dense_results = collection.query(query_texts=[query_text], n_results=retrieval_k, where=where_filter)

    ids = dense_results["ids"][0]
    documents = (dense_results.get("documents") or [[]])[0]
    metadatas = (dense_results.get("metadatas") or [[]])[0]
    distances = (dense_results.get("distances") or [[]])[0]

    dense_sources = []
    for i in range(len(ids)):
        dense_sources.append(
            {
                "id": ids[i],
                "distance": distances[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }
        )

    # ========== HYBRID SEARCH: BM25 Fusion ==========
    if use_hybrid_search:
        print("[search] Running BM25 sparse search")
        bm25_sources = bm25_search(query_text, k=retrieval_k)

        # Fuse dense + sparse via RRF
        sources = rrf_fusion(dense_sources, bm25_sources)
        print(f"[search] After RRF fusion: {len(sources)} results")

    # ========== RERANKING ==========
    if use_reranker and sources:
        sources = lightweight_rerank(query_text, sources)

    # Keep only top-k final results
    sources = sources[:k]
    answer = "AI answer is not implemented yet..."

    return answer, sources


def query_cli():
    parser = argparse.ArgumentParser(description="Query the ChromaDB collection")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--source", choices=["all", "eesr", "ssmesr"], default="all", help="Source to query")
    parser.add_argument("--k", type=int, default=5, help=f"Number of results to return (1-{MAX_K})", metavar=f"1-{MAX_K}")
    parser.add_argument("--no-rerank", action="store_true", help="Disable CrossEncoder reranking")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search")
    args = parser.parse_args()

    answer, sources = query(
        args.query,
        source=args.source,
        k=args.k,
        use_reranker=not args.no_rerank,
        use_hybrid_search=not args.no_hybrid,
    )

    print(f"Answer: {answer}")
    print(f"\nTop {len(sources)} sources:")
    for i, src in enumerate(sources, 1):
        print(f"\n{i}. {src['metadata'].get('title', 'N/A')} (distance: {src['distance']:.4f})")
        print(f"   {src['document'][:200]}...")


if __name__ == "__main__":
    query_cli()
