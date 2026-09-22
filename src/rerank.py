import os
import re
import logging
from datetime import datetime
from src.utils import fetch_data

logger = logging.getLogger(__name__)


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
        title_score = len(title_words & query_words) / max(1, len(title_words))

        # Temporal: does doc year match query years?
        publication_epoch = source["metadata"]["publication_epoch"]
        temporal_score = 0.0
        if query_years:
            min_year = min(query_years)
            min_epoch = datetime(min_year, 1, 1).timestamp()
            if publication_epoch >= min_epoch:
                # max_epoch = CURRENT_DATE.timestamp()
                # temporal_score = min(1.5, 1.0 + 0.5 * ((publication_epoch - min_epoch) / (max_epoch - min_epoch)))
                temporal_score = 1.0

        # Combine with weights
        final_score = 0.85 * semantic_score + 0.05 * title_score + 0.1 * temporal_score
        source["rerank_score"] = final_score

    return sorted(sources, key=lambda x: x["rerank_score"], reverse=True)


def cross_encoder_rerank(query_text: str, sources: list[dict]) -> list[dict]:
    """
    Rerank ChromaDB results using sentence-transformers cross-encoder model
    """
    url = f"{os.getenv("ML_HUB_API_URL")}/tools/cross-encoder"
    headers = {"Authorization": os.getenv("ML_HUB_API_KEY")}

    if not len(sources):
        return sources

    try:
        pairs = [[query_text, source["document"]] for source in sources]
        data = fetch_data(url, method="POST", headers=headers, json={"pairs": pairs})
        for source, score in zip(sources, data):
            source["rerank_score"] = score
        return sorted(sources, key=lambda x: x["rerank_score"], reverse=True)

    except Exception as error:
        logger.error(f"Error while fetching cross encoder tool: {str(error)}")
        raise error


def rerank(
    query_text: str, sources: list[dict], use_cross_encoder: bool = False, fallback_lightweight: bool = True
) -> list[dict]:
    """
    Rerank ChromaDB results using either cross-encoder or lightweight reranker.
    If cross-encoder fails, fallback to lightweight reranker if enabled.
    """
    if use_cross_encoder:
        try:
            return cross_encoder_rerank(query_text, sources)
        except Exception as error:
            logger.error(f"Cross encoder reranking failed: {str(error)}")
            if fallback_lightweight:
                logger.info("Falling back to lightweight reranker")
                return lightweight_rerank(query_text, sources)
            return sources

    return lightweight_rerank(query_text, sources)
