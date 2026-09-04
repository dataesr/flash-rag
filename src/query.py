from src.mistral import mistral_rag_answer, RagCitation
import re
import argparse
from time import perf_counter
from typing import Literal
from datetime import datetime
from src.utils import parse_key_value_pair
from src.chromadb import get_collection
from src.bm25 import bm25_search, rrf_fusion

CURRENT_DATE = datetime.now()
CURRENT_YEAR = CURRENT_DATE.year
MAX_TIMESTAMP = datetime((CURRENT_YEAR - 4), 1, 1).timestamp()  # 3 years ago + 1 year buffer
MIN_CHUNK_LEN = 50  # Minimum chunk length to consider for querying
MAX_K = 50  # Max docs to retrieves
K_MULTIPLIER = 5  # Multiplier for candidate retrieval before RRF and reranking


def lightweight_rerank(query_text: str, sources: list[dict]) -> list[dict]:
    """
    Rerank ChromaDB results by combining:
    1. Semantic score (from ChromaDB)
    2. Title relevance (keyword overlap)
    3. Temporal proximity (year matching/recency bias)
    """

    # Parse query
    query_lower = query_text.lower().strip()
    query_words = set(query_lower.split())
    query_years = {int(year) for year in re.findall(r"\b(20\d{2})\b", query_lower)}

    for source in sources:
        semantic_score = 1 - source["distance"]

        # Title relevance: how much query keywords overlap with title
        title_words = set(source["metadata"]["title"].lower().split())
        title_score = len(title_words & query_words) / (len(title_words | query_words) + 1e-6)

        # Temporal: does doc year match query years?
        publication_epoch = source["metadata"]["publication_epoch"]
        temporal_score = 0.0
        if query_years:
            min_year = min(query_years)
            min_epoch = datetime(min_year, 1, 1).timestamp()
            if publication_epoch >= min_epoch:
                max_epoch = CURRENT_DATE.timestamp()
                temporal_score = min(1.5, 1.0 + 0.5 * ((publication_epoch - min_epoch) / (max_epoch - min_epoch)))

        # Combine with weights
        final_score = 0.5 * semantic_score + 0.25 * title_score + 0.25 * temporal_score
        source["rerank_score"] = final_score

    return sorted(sources, key=lambda x: x["rerank_score"], reverse=True)


def query(
    query_text: str,
    source: Literal["all", "eesr", "ssmesr"] = "all",
    k: int = 5,
    use_reranker: bool = False,
    use_hybrid_search: bool = False,
    use_mistral: bool = False,
    filters: dict = {},
) -> tuple[list, str, list[RagCitation]]:
    """
    Query the RAG collection with optional hybrid search (dense + BM25) and reranking.

    Args:
        query_text: The query string
        source: Which source to query from
        filters: Additionnal filters
        k: Number of final results to return (1-50)
        use_reranker: Whether to use reranking
        use_hybrid_search: Whether to combine vector + BM25 search (RRF fusion)
        use_mistral: Whether to use Mistral to get a LLM answer
    Returns:
        (sources, answer, citations)
    """

    query_started = perf_counter()

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

    if filters.get("chunk_type"):
        where_filter["$and"].append({"chunk_type": {"$eq": filters["chunk_type"]}})

    if source != "all":
        where_filter["$and"].append({"source": {"$eq": source}})  # ty: ignore[invalid-argument-type]

    # Retrieve more candidates than the final k because
    # RRF + reranking need a larger candidate pool
    retrieval_k = min(k * K_MULTIPLIER, MAX_K) if (use_hybrid_search or use_reranker) else k

    print(f"[search] k={k}, retrieval_k={retrieval_k}, " f"hybrid={use_hybrid_search}, reranker={use_reranker}")

    # ========== DENSE SEARCH (Vector/Mistral) ==========
    print(f"[search] Dense search: retrieving top {retrieval_k}")
    stage_started = perf_counter()
    dense_results = collection.query(query_texts=[query_text], n_results=retrieval_k, where=where_filter)
    print(f"[timing] dense search + embedding: {perf_counter() - stage_started:.3f}s")

    ids = dense_results["ids"][0]
    documents = (dense_results.get("documents") or [[]])[0]
    metadatas = (dense_results.get("metadatas") or [[]])[0]
    distances = (dense_results.get("distances") or [[]])[0]

    sources = []
    for i in range(len(ids)):
        sources.append(
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
        stage_started = perf_counter()
        bm25_sources = bm25_search(query_text, k=retrieval_k)
        print(f"[timing] BM25 search: {perf_counter() - stage_started:.3f}s")

        # Fuse dense + sparse via RRF
        stage_started = perf_counter()
        sources = rrf_fusion(sources, bm25_sources)
        print(f"[search] After RRF fusion: {len(sources)} results")
        print(f"[timing] RRF fusion: {perf_counter() - stage_started:.3f}s")

    # ========== RERANKING ==========
    if use_reranker and sources:
        stage_started = perf_counter()
        sources = lightweight_rerank(query_text, sources)
        print(f"[timing] reranking: {perf_counter() - stage_started:.3f}s")

    # Keep only top-k final results
    sources = sources[:k]

    # Mistral answer
    answer = "mistral_not_enabled"
    citations = []
    if use_mistral:
        answer_started = perf_counter()
        rag_answer = mistral_rag_answer(query_text, documents=sources)
        answer = rag_answer.answer
        citations = rag_answer.citations
        print(f"[timing] mistral answer: {perf_counter() - answer_started}")
        print(f"[debug] citations = {citations}")

    print(f"[timing] total query: {perf_counter() - query_started:.3f}s")
    return sources, answer, citations


def query_cli():
    parser = argparse.ArgumentParser(description="Query the ChromaDB collection")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--source", choices=["all", "eesr", "ssmesr"], default="all", help="Source to query")
    parser.add_argument("--k", type=int, default=5, help=f"Number of results to return (1-{MAX_K})", metavar=f"1-{MAX_K}")
    parser.add_argument("--use-rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--use-hybrid", action="store_true", help="Enable hybrid search")
    parser.add_argument("--use-mistral", action="store_true", help="Enable Mistral answer")
    parser.add_argument(
        "-f",
        "--filter",
        action="append",
        type=parse_key_value_pair,
        help="Filters in format: key=value (can be used multiple times)",
    )
    args = parser.parse_args()

    sources, answer, citations = query(
        args.query,
        source=args.source,
        k=args.k,
        use_reranker=args.use_rerank,
        use_hybrid_search=args.use_hybrid,
        use_mistral=args.use_mistral,
        filters=dict(args.filter) if args.filter else {},
    )

    print(f"Answer: {answer}")
    print(f"\nCitations:")
    for citation in citations:
        print(f"\n[{citation.citation}] {citation.source_title} (source_index: {citation.source_index - 1})")
    print(f"\nTop {len(sources)} sources:")
    for i, src in enumerate(sources, 1):
        print(f"\n{i}. {src['metadata'].get('title', 'N/A')} (distance: {src['distance']:.4f})")
        print(f"   {src['document'][:200]}...")


if __name__ == "__main__":
    query_cli()
