import argparse
from src.chromadb import get_collection


def query(args=None):
    parser = argparse.ArgumentParser(description="Query the ChromaDB collection")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args(args)

    # Get collection
    collection = get_collection()

    # Query collection
    results = collection.query(
        query_texts=[args.query],
        n_results=args.k,
    )

    # class QueryResult(TypedDict):
    #     ids: List[IDs]
    #     embeddings: Optional[List[Embeddings]]
    #     documents: Optional[List[List[Document]]]
    #     uris: Optional[List[List[URI]]]
    #     metadatas: Optional[List[List[Metadata]]]
    #     distances: Optional[List[List[float]]]
    #     included: Include

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

    answer = "AI answer is not implemented yet..."

    return answer, sources


if __name__ == "__main__":
    query()
