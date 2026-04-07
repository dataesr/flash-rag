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

    # Print results
    for i, (doc, metadata, score) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])  # ty:ignore[not-subscriptable]
    ):
        print(f"Result {i + 1} (score: {score:.4f}):")
        print(f"    Document: {doc}")
        print(f"    Metadata: {metadata}")
        print()


if __name__ == "__main__":
    query()
