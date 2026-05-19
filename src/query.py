import argparse
from datetime import datetime
from src.chromadb import get_collection

MAX_TIMESTAMP = datetime((datetime.now().year - 4), 1, 1).timestamp()  # 3 years ago + 1 year buffer


def sources_to_publications(sources: list):
    publications = {}
    for source in sources:
        publication = {
            "file_name": source["metadata"].pop("file_name"),
            "file_format": source["metadata"].pop("file_format"),
            "doc_type": source["metadata"].pop("doc_type"),
            "title": source["metadata"].pop("title"),
            "created": source["metadata"].pop("created"),
            "modified": source["metadata"].pop("modified"),
            "publication_date": source["metadata"].pop("publication_date"),
            "publication_epoch": source["metadata"].pop("publication_epoch"),
            "keywords": source["metadata"].pop("keywords"),
        }
        publication_id = publication["file_name"]
        if publication_id not in publications:
            publications[publication_id] = publication
            publications[publication_id]["sources"] = []

        # Add source chunk
        publications[publication_id]["sources"].append(source)
        # Mean distance from chunks
        publications[publication_id]["distance"] = sum(
            [s["distance"] for s in publications[publication_id]["sources"]]
        ) / len(publications[publication_id]["sources"])
        # Sort sources by page and section
        publications[publication_id]["sources"] = sorted(
            publications[publication_id]["sources"],
            key=lambda x: (x["metadata"]["page_index"], x["metadata"]["section_level"]),
        )

    # Sort publications by mean distance
    return sorted(publications.values(), key=lambda x: x["distance"])


def query(query_text: str, k: int = 5):
    # Get collection
    collection = get_collection()

    # Query collection
    results = collection.query(
        query_texts=[query_text],
        n_results=k,
        where={"publication_epoch": {"$gte": MAX_TIMESTAMP}},
    )

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

    return answer, sources, sources_to_publications(sources)


def query_cli():
    parser = argparse.ArgumentParser(description="Query the ChromaDB collection")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    query(args.query, args.k)


if __name__ == "__main__":
    query_cli()
