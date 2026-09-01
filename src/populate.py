from src.bm25 import build_bm25_index
import argparse
from typing import Any, Literal
from src.chromadb import get_collection
from src.pipelines.transform_eesr import transform as transform_eesr
from src.pipelines.transform_ssmesr import transform as transform_ssmesr

MAX_ITEMS_PER_BATCH = 5000


def batch_chroma_payload(
    ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], max_items_per_batch: int = MAX_ITEMS_PER_BATCH
):
    """Split collection payloads into smaller batches for ChromaDB."""
    if not ids:
        return []

    if max_items_per_batch <= 0:
        raise ValueError("max_items_per_batch must be greater than zero")

    return [
        (ids[i : i + max_items_per_batch], documents[i : i + max_items_per_batch], metadatas[i : i + max_items_per_batch])
        for i in range(0, len(ids), max_items_per_batch)
    ]


def populate(
    source: Literal["all", "ssmesr", "eesr"] = "all",
    use_cache: bool = True,
    reset: bool = False,
    override: bool = False,
    build_bm25: bool = True,
):
    collection = get_collection(reset)
    all_chunks = []

    if source in ["all", "ssmesr"]:
        print("[populate] Running SSMESR transform")
        ssmesr_chunks = transform_ssmesr(use_cache)
        print(f"[populate] SSMESR chunks: {len(ssmesr_chunks)}")
        if len(ssmesr_chunks):
            all_chunks += ssmesr_chunks

    if source in ["all", "eesr"]:
        print("[populate] Running EESR transform")
        eesr_chunks = transform_eesr(use_cache)
        print(f"[populate] EESR chunks: {len(eesr_chunks)}")
        if len(eesr_chunks):
            all_chunks += eesr_chunks

    if not all_chunks:
        print("[populate] No chunks to ingest")
        return

    ids = [chunk["id"] for chunk in all_chunks]
    documents = [chunk["document"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    batches = batch_chroma_payload(ids, documents, metadatas)

    for batch_index, (batch_ids, batch_documents, batch_metadatas) in enumerate(batches, start=1):
        print(f"[populate] Writing batch {batch_index}/{len(batches)} ({len(batch_ids)} chunks) \
            into collection '{collection.name}'")

        if override:
            collection.upsert(ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas)
        else:
            collection.add(ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas)

    print(f"[populate] Indexed {len(all_chunks)} chunks")

    if build_bm25:
        build_bm25_index()


def populate_cli():
    parser = argparse.ArgumentParser(description="Populate ChromaDB with EESR and SSMESR chunks")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection")
    parser.add_argument("--override", action="store_true", help="Override existing documents")
    args = parser.parse_args()
    populate(reset=args.reset, override=args.override)


if __name__ == "__main__":
    populate_cli()
