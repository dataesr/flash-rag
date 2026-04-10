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
CHUNK_MAX_CHARS = 8000


def chunk_document(ocr_path: str, document_metadata: dict) -> list[dict]:
    file_name = document_metadata["file_name"]

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

        for section_index, section in enumerate(parsed_sections):
            title = section.get("title", "")
            level = section.get("level", 0)
            paragraphs = section.get("paragraphs", [])
            # tables = section.get("tables", []) #TODO: implement table parsing

            if not paragraphs:
                # print(f"[populate] {file_name}: No paragraphs found in section {section_index} of page {page_index}")
                continue

            current_doc = ""
            current_chunks = []
            for para in paragraphs:
                next_doc = current_doc + "\n" + para if current_doc else para
                if len(next_doc) <= CHUNK_MAX_CHARS:
                    current_doc = next_doc
                else:
                    current_chunks.append(current_doc)
                    current_doc = para
            current_chunks.append(current_doc)

            if len(current_chunks) > 1:
                print(f"[populate] {file_name}: Section={section_index}, page={page_index} --> {len(current_chunks)} chunks")

            for chunk_index, chunk in enumerate(current_chunks):
                chunks.append(
                    {
                        "id": f"{file_name}_p{page_index}_s{section_index}_{chunk_index}",
                        "document": chunk,
                        "metadata": {
                            **document_metadata,
                            "page_index": page_index,
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
    document_metadata = {
        "file_id": file["file_id"],
        "file_name": file["file_name"],
        "file_format": file["file_format"],
        "doc_type": file["subtype"],
        "publication_date": file["publication_date"],
        "created": str(file["created"]),
        "modified": str(file["modified"]),
        "title": file["title"],
    }
    print(f"[populate] Upserting document {path}")
    chunks = chunk_document(path, document_metadata)
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


def populate(reset: bool = False, override: bool = False):

    # Get records
    records = get_records()

    # Get files
    files = get_files(records)
    files_with_ocr = files[files["ocr_path"].apply(os.path.exists)]
    print(f"[populate] Found {len(files_with_ocr)} files with OCR")
    if not len(files_with_ocr):
        return

    # Get collection
    collection = get_collection(reset)

    # Chunk and add documents
    print(f"[populate] Loading OCR files from {OCR_DIR}")
    upsert_documents(files_with_ocr, collection, override)


def populate_cli():
    parser = argparse.ArgumentParser(description="Populate ChromaDB with OCR documents")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection")
    parser.add_argument("--override", action="store_true", help="Override existing documents")
    args = parser.parse_args()
    populate(reset=args.reset, override=args.override)


if __name__ == "__main__":
    populate_cli()
