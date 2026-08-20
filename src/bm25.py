# BM25 config
# To remove when BM25 supported by chromadb
import os
import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from src.chromadb import get_collection

BM25_DIR = "./data/bm25"
BM25_PATH = f"{BM25_DIR}/index.pkl"

_bm25_index = None


def get_bm25_index():
    """Lazy-load or create BM25 index from persisted pickle file"""
    global _bm25_index
    if _bm25_index is None:
        try:
            # Try to load persisted index
            if os.path.exists(BM25_PATH):
                print(f"[bm25] Loading BM25 index from {BM25_PATH}")
                with open(BM25_PATH, "rb") as f:
                    _bm25_index = pickle.load(f)
                print("[bm25] BM25 index loaded successfully")
            else:
                print(f"[bm25] BM25 index not found at {BM25_PATH}. Run build_bm25_index() first.")
                _bm25_index = False  # Flag to skip
        except Exception as error:
            print(f"[error] Failed to load BM25 index: {error}. Continuing with vector-only search.")
            _bm25_index = False

    return _bm25_index if _bm25_index is not False else None


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25
    """
    # TODO: better tokenization ?
    return text.lower().split()


def build_bm25_index() -> bool:
    """
    Build and persist the BM25 index from the Chroma collection.

    Returns:
        {
            "bm25": BM25Okapi,
            "ids": list[str],
            "documents": list[str],
            "tokenized_documents": list[list[str]],
        }
    """

    os.makedirs(BM25_DIR, exist_ok=True)
    print("[bm25] Building BM25 index from ChromaDB collection...")

    collection = get_collection()

    # Retrieve the complete corpus.
    # Chroma's get() returns documents and IDs
    results = collection.get(include=["documents"])
    ids = results["ids"] or []
    documents = results["documents"] or []

    print(f"[bm25] Indexing {len(documents)} documents")

    tokenized_documents = [tokenize(document) for document in documents]
    bm25 = BM25Okapi(tokenized_documents)
    index = {
        "bm25": bm25,
        "ids": ids,
        "documents": documents,
        "tokenized_documents": tokenized_documents,
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"[bm25] Index saved to {BM25_PATH}")

    # Clear global to reload new index
    global _bm25_index
    _bm25_index = None

    return True


def bm25_search(
    query: str,
    k: int = 20,
) -> list[dict]:
    """
    Search the BM25 index.

    Returns list of
        {
            "id": "...",
            "document": "...",
            "bm25_score": 12.34,
        }
    """

    index = get_bm25_index()
    if not index:
        return []

    try:
        bm25: BM25Okapi = index["bm25"]
        ids = index["ids"]
        documents = index["documents"]

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        # Highest BM25 score = most relevant.
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue  # Don't return documents with zero BM25 relevance.
            results.append(
                {
                    "id": ids[idx],
                    "document": documents[idx],
                    "bm25_score": score,
                }
            )
        print(f"[bm25] BM25 search returned {len(results)} results")
        return results

    except Exception as error:
        print(f"[error] BM25 search failed: {error}")
        return []


def rrf_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Fuse dense and BM25 rankings using Reciprocal Rank Fusion.
    RRF score: weight / (rrf_k + rank)
    Higher score = better result.
    """
    if not bm25_results:
        return dense_results

    fused = {}

    # Dense ranking
    for rank, result in enumerate(dense_results, start=1):
        doc_id = result["id"]
        if doc_id not in fused:
            fused[doc_id] = {
                "id": doc_id,
                "document": result["document"],
                "metadata": result.get("metadata"),
                "distance": result.get("distance"),
                "bm25_score": None,
                "rrf_score": 0.0,
            }
        fused[doc_id]["rrf_score"] += dense_weight / (rrf_k + rank)  # ty: ignore[unsupported-operator]

    # Add BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id in fused:
            fused[doc_id]["bm25_score"] = result["bm25_score"]
            fused[doc_id]["rrf_score"] += bm25_weight / (rrf_k + rank)  # ty: ignore[unsupported-operator]

    # Sort by rrf score
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
