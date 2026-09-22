import os
import httpx
import re
import logging
import argparse
from time import perf_counter
from typing import Literal, Any
from datetime import datetime
from chromadb.types import Where
from src.utils import parse_key_value_pair, fetch_data
from src.chromadb import get_collection
from src.bm25 import bm25_search, rrf_fusion
from src.mistral import mistral_rag_answer, RagCitation
from src.rerank import rerank

logger = logging.getLogger(__name__)

CURRENT_DATE = datetime.now()
CURRENT_YEAR = CURRENT_DATE.year
MAX_TIMESTAMP = datetime((CURRENT_YEAR - 4), 1, 1).timestamp()  # 3 years ago + 1 year buffer
MIN_CHUNK_LEN = 50  # Minimum chunk length to consider for querying
MAX_K = 50  # Max docs to retrieves
K_MULTIPLIER = 5  # Multiplier for candidate retrieval before RRF and reranking


def query(
    query_text: str,
    k: int = 5,
    use_reranker: bool = False,
    use_cross_encoder: bool = False,
    use_hybrid_search: bool = False,
    use_mistral: bool = False,
    filters: dict[str, str | list[str]] = {},
) -> tuple[list, str, list[RagCitation]]:
    """
    Query the RAG collection with optional hybrid search (dense + BM25) and reranking.

    Args:
        query_text: The query string
        k: Number of final results to return (1-50)
        use_reranker: Whether to use reranking
        use_hybrid_search: Whether to combine vector + BM25 search (RRF fusion)
        use_mistral: Whether to use Mistral to get a LLM answer
        filters: Additionnal filters
    Returns:
        (sources, answer, citations)
    """

    query_started = perf_counter()

    # Validate k
    k = max(1, min(k, MAX_K))

    # Get collection
    collection = get_collection()

    # Build filters
    where_filter: Where = {
        "$and": [
            {"publication_epoch": {"$gte": MAX_TIMESTAMP}},
            {"chunk_len": {"$gte": MIN_CHUNK_LEN}},
        ]
    }

    for key, value in filters.items():
        if key in ["reference", "publication_type", "chunk_type"]:
            val = value[0] if isinstance(value, list) else value
            where_filter["$and"].append({key: {"$eq": val}})
        elif key == "keywords":
            val = value if isinstance(value, list) else [value]
            keywords = [k.strip() for k in val if k.strip()]
            if len(keywords) == 1:
                where_filter["$and"].append({key: {"$contains": keywords[0]}})
            elif len(keywords) > 1:
                where_filter["$and"].append({"$or": [{key: {"$contains": k}} for k in keywords]})
        else:
            logger.warning(f"Query filter {key}={value} skipped")

    # Retrieve more candidates than the final k because
    # RRF + reranking need a larger candidate pool
    retrieval_k = min(k * K_MULTIPLIER, MAX_K) if (use_hybrid_search or use_reranker) else k

    logger.debug(f"Search with k={k}, retrieval_k={retrieval_k}, " f"hybrid={use_hybrid_search}, reranker={use_reranker}")

    # ========== DENSE SEARCH (Vector/Mistral) ==========
    logger.debug(f"Dense search: retrieving top {retrieval_k}")
    stage_started = perf_counter()
    dense_results = collection.query(query_texts=[query_text], n_results=retrieval_k, where=where_filter)
    logger.debug(f"[timing] dense search + embedding: {perf_counter() - stage_started:.3f}s")

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
        logger.debug("Running BM25 sparse search")
        stage_started = perf_counter()
        bm25_sources = bm25_search(query_text, k=retrieval_k)
        logger.debug(f"[timing] BM25 search: {perf_counter() - stage_started:.3f}s")

        # Fuse dense + sparse via RRF
        stage_started = perf_counter()
        sources = rrf_fusion(sources, bm25_sources)
        logger.debug(f"After RRF fusion: {len(sources)} results")
        logger.debug(f"[timing] RRF fusion: {perf_counter() - stage_started:.3f}s")

    # ========== RERANKING ==========
    if (use_reranker or use_cross_encoder) and sources:
        stage_started = perf_counter()
        sources = rerank(query_text, sources, use_cross_encoder=use_cross_encoder)
        logger.debug(f"[timing] reranking: {perf_counter() - stage_started:.3f}s")

    # Keep only top-k final results
    sources = sources[:k]

    # ========== LLM GENERATION ==========
    answer = "mistral_not_enabled"
    citations = []
    if use_mistral:
        answer_started = perf_counter()
        rag_answer = mistral_rag_answer(query_text, documents=sources)
        answer = rag_answer.answer
        citations = rag_answer.citations
        logger.debug(f"[timing] mistral answer: {perf_counter() - answer_started}")
        logger.debug(f"Mistral citations = {citations}")

    logger.debug(f"[timing] total query: {perf_counter() - query_started:.3f}s")
    return sources, answer, citations


def query_cli():
    parser = argparse.ArgumentParser(description="Query the ChromaDB collection")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--k", type=int, default=5, help=f"Number of results to return (1-{MAX_K})", metavar=f"1-{MAX_K}")
    parser.add_argument("--use-rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--use-cross-encoder", action="store_true", help="Enable cross-encoder reranking")
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
        k=args.k,
        use_reranker=args.use_rerank,
        use_cross_encoder=args.use_cross_encoder,
        use_hybrid_search=args.use_hybrid,
        use_mistral=args.use_mistral,
        filters=dict(args.filter) if args.filter else {},
    )

    print(f"Answer: {answer}")
    print(f"\nCitations:")
    for citation in citations:
        print(f"\n[{citation.source_index}] {citation.source_title}")
    print(f"\nTop {len(sources)} sources:")
    for i, src in enumerate(sources, 1):
        print(f"\n{i}. {src['metadata'].get('title', 'N/A')} (distance: {src['distance']:.4f})")
        print(f"   {src['document'][:200]}...")


if __name__ == "__main__":
    query_cli()
