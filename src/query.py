from typing import Literal
import argparse
from datetime import datetime
from src.chromadb import get_collection

MAX_TIMESTAMP = datetime((datetime.now().year - 4), 1, 1).timestamp()  # 3 years ago + 1 year buffer
MIN_CHUNK_LEN = 50  # Minimum chunk length to consider for querying
MAX_K = 50  # Maximum docs to retrieve

RERANKER = None  # Load once at first use


def get_reranker():
    """Load CrossEncoder reranker"""
    global RERANKER
    if RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder

            RERANKER = CrossEncoder(
                "cross-encoder/mmarco-mMiniLMv2L12H384",  # https://www.sbert.net/docs/pretrained-models/ce-msmarco.html
                max_length=512,
                device="cpu",  # "cuda" if GPU
            )
            print("[reranker] CrossEncoder loaded successfully")
        except Exception as error:
            print(f"[reranker] Failed to load reranker: {error}. Continuing without reranking.")
            RERANKER = False  # Flag to not retry
    return RERANKER if RERANKER is not False else None


def rerank_sources(query_text: str, sources: list) -> list:
    """
    Rerank sources using CrossEncoder for better relevance ordering.
    Falls back to original ranking if reranker unavailable.
    """
    reranker = get_reranker()
    if not reranker or not sources:
        return sources

    try:
        # Get pairs and compute CrossEncoder scores
        pairs = [(query_text, source["document"]) for source in sources]
        scores = reranker.predict(pairs, batch_size=32)

        # Update distances with reranker scores (normalized)
        for i, source in enumerate(sources):
            source["distance"] = -scores[i]

        # Resort by distances
        reranked = sorted(sources, key=lambda x: x["distance"])
        print(f"[rerank_sources] Reranked {len(sources)} sources")
        return reranked

    except Exception as error:
        print(f"[rerank_sources] Reranking failed: {error}. Returning original order.")
        return sources


def query(query_text: str, source: Literal["all", "eesr", "ssmesr"] = "all", k: int = 5, use_reranker: bool = True):

    # Get collection
    collection = get_collection()

    # Build where filter
    where_filter = {
        "$and": [
            {"publication_epoch": {"$gte": MAX_TIMESTAMP}},
            {"chunk_len": {"$gte": MIN_CHUNK_LEN}},
        ]
    }
    if source != "all":
        where_filter["$and"].append({"source": {"$eq": source}})  # ty:ignore[invalid-argument-type]

    # Validate k
    k = max(1, min(k, MAX_K))
    retrieval_k = min(k * 3, MAX_K) if use_reranker else k  # Retrieve more results if reranking is enabled

    # Query collection
    results = collection.query(
        query_texts=[query_text],
        n_results=retrieval_k,
        where=where_filter,
    )

    # Get results
    ids = results["ids"][0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    sources = []
    for i in range(len(ids)):
        sources.append(
            {
                "distance": distances[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }
        )

    # Rerank if enabled and we have sources
    if use_reranker and sources:
        sources = rerank_sources(query_text, sources)

    # Return sources and answer
    sources = sources[:k]
    answer = "AI answer is not implemented yet..."

    return answer, sources


def query_cli():
    parser = argparse.ArgumentParser(description="Query the ChromaDB collection")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--source", choices=["all", "eesr", "ssmesr"], default="all", help="Source to query")
    parser.add_argument("--k", type=int, default=5, help=f"Number of results to return (1-{MAX_K})", metavar=f"1-{MAX_K}")
    parser.add_argument("--no-rerank", action="store_true", help="Disable CrossEncoder reranking")
    args = parser.parse_args()

    answer, sources = query(args.query, source=args.source, k=args.k, use_reranker=not args.no_rerank)

    print(f"Answer: {answer}")
    print(f"\nTop {len(sources)} sources:")
    for i, src in enumerate(sources, 1):
        print(f"\n{i}. {src['metadata'].get('title', 'N/A')} (distance: {src['distance']:.4f})")
        print(f"   {src['document'][:200]}...")


if __name__ == "__main__":
    query_cli()
