from chromadb import Collection
import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from src.chromadb import get_collection
from src.load import OCR_DIR, get_records, get_files

load_dotenv()

BATCH_SIZE = 50  # Batch size for upserts


def chunk_document(ocr_path: str) -> list[dict]:
    """Load an OCR JSON file and produce chunks from its parsed sections.

    Each parsed section becomes one chunk. The document text is built
    from the section title + its paragraphs. Metadata includes file name,
    page index, section title, and section level.

    Returns a list of dicts with keys: id, document, metadata.
    """
    file_name = os.path.splitext(os.path.basename(ocr_path))[0]

    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as error:
        print(f"[populate] Error loading {ocr_path}: {error}")
        return []

    pages = data.get("pages", [])
    if not pages:
        return []

    chunks = []
    for page in pages:
        page_index = page.get("index", 0)
        parsed_sections = page.get("parsed", [])

        if not parsed_sections:
            continue

        for section_idx, section in enumerate(parsed_sections):
            title = section.get("title", "")
            level = section.get("level", 0)
            paragraphs = section.get("paragraphs", [])
            # tables = section.get("tables", []) #TODO: implement table parsing

            if not paragraphs:
                print(f"[populate] No paragraphs found in section {section_idx} of page {page_index} of {ocr_path}")
                continue

            for para_index, para in enumerate(paragraphs):
                chunk_id = f"{file_name}_p{page_index}_s{section_idx}_p{para_index}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "document": para,
                        "metadata": {
                            "file_name": file_name,
                            "page_index": page_index,
                            "paragraph_index": para_index,
                            "section_title": title[:200],  # Truncate for metadata
                            "section_level": level,
                            "chunk_type": "paragraph",
                        },
                    }
                )

    return chunks


def upsert_chunks(chunks: list[dict], collection: Collection, override: bool = False):
    """Upsert chunks into the collection in batches."""
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = chunks[start:end]

        ids = [c["id"] for c in batch]
        documents = [c["document"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        if override:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[populate] Upserted {total} chunks")


def upsert_one_document(file: pd.Series, collection: Collection, override: bool = False) -> int:
    path = file["ocr_path"]
    print(f"[populate] Upserting document {path}")
    chunks = chunk_document(path)
    if not chunks or not isinstance(chunks, list) or len(chunks) == 0:
        return 0
    try:
        upsert_chunks(chunks, collection, override)
    except Exception as error:
        print(f"[populate] Error while upserting {path}: {error}")
        return 0
    return len(chunks)


def upsert_documents(files: pd.DataFrame, collection: Collection, override: bool = False):

    stats = files.apply(upsert_one_document, collection=collection, override=override, axis=1)
    total_chunks = stats.sum()

    print(f"[populate] Populated {len(files)} files with {total_chunks} chunks")
    return


def populate(args=None):
    parser = argparse.ArgumentParser(description="Populate ChromaDB with OCR documents")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection")
    parser.add_argument("--override", action="store_true", help="Override existing documents")
    args = parser.parse_args(args)

    # Get records
    records = get_records()

    # Get files
    files = get_files(records)
    files_with_ocr = files[files["ocr_path"].apply(os.path.exists)]
    print(f"[populate] Found {len(files_with_ocr)} files with OCR")
    if not len(files_with_ocr):
        return

    # Get collection
    collection = get_collection(args.reset)

    # Chunk and add documents
    print(f"[populate] Loading OCR files from {OCR_DIR}")
    upsert_documents(files_with_ocr, collection, args.override)


if __name__ == "__main__":
    populate()
