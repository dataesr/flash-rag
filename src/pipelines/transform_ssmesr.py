import os
import json
import argparse
import pandas as pd
from src.pipelines.load_ssmesr import OCR_DIR, get_records, get_files
from src.utils import save_jsonl, to_unix_epoch

OUTPUT_DIR = "./data"
OUTPUT_CHUNKS = f"{OUTPUT_DIR}/ssmesr_chunks.jsonl"
CHUNK_MAX_CHARS = 8000


def parse_table(table: dict) -> tuple[str, str, str]:
    """
    Parse SSMESR table

    Returns: (markdown, csv, headers_text)
    """
    headers = table.get("headers", [])
    data = table.get("data", [])

    if not headers or not data:
        return "", "", ""

    # Markdown format
    md = "| " + " | ".join(str(h) for h in headers) + " |\n"
    md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for row in data:
        md += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    # CSV format
    lines = [",".join(str(h) for h in headers)]
    for row in data:
        lines.append(",".join(str(cell) for cell in row))
    csv = "\n".join(lines)

    # Extract column names for better BM25 matching
    headers_text = " | ".join(str(h) for h in headers)

    return md, csv, headers_text


def chunk_document(ocr_path: str, document_metadata: dict) -> list[dict]:
    file_name = document_metadata["file_name"]
    file_name_no_ext = file_name.split(".")[0] if "." in file_name else file_name

    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as error:
        print(f"[transform_ssmesr] Error loading {ocr_path}: {error}")
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
            tables = section.get("tables", [])

            # ========== PARAGRAPHS ==========
            if paragraphs:
                current_doc = ""
                current_chunks = []
                for para in paragraphs:
                    # Skip table captions and image references
                    if para.startswith("TABLEAU") or para.startswith("![") or para.startswith("GRAPHIQUE"):
                        continue

                    next_doc = current_doc + "\n\n" + para if current_doc else para
                    if len(next_doc) <= CHUNK_MAX_CHARS:
                        current_doc = next_doc
                    else:
                        current_chunks.append(current_doc)
                        current_doc = para

                if current_doc:
                    current_chunks.append(current_doc)

                # if len(current_chunks) > 1:
                #     print(
                #         f"[transform_ssmesr] {file_name}: page={section_index}, section={page_index} --> {len(current_chunks)} paragraph chunks"
                #     )

                for chunk_index, chunk in enumerate(current_chunks):
                    chunks.append(
                        {
                            "id": f"ssmesr_{file_name_no_ext}_p{page_index}_s{section_index}_p{chunk_index}",
                            "document": chunk,
                            "metadata": {
                                **document_metadata,
                                "page_index": page_index,
                                "section_title": title[:200],
                                "section_level": level,
                                "chunk_type": "paragraph",
                                "chunk_len": len(chunk),
                            },
                        }
                    )

            # ========== TABLES ==========
            if tables:
                # print(f"[transform_ssmesr] {file_id}: page={section_index}, section={page_index} --> {len(tables)} table(s)")

                for table_index, table in enumerate(tables):
                    if not isinstance(table, dict):
                        print(f"[warn] Empty table found for document {file_name} ({ocr_path})")
                        continue

                    # Convert table to searchable markdown
                    markdown_table, csv_table, headers_text = parse_table(table)

                    if not markdown_table:
                        print(f"[warn] No markdown table for document {file_name} ({ocr_path})")
                        continue

                    chunks.append(
                        {
                            "id": f"ssmesr_{file_name_no_ext}_p{page_index}_s{section_index}_t{table_index}",
                            "document": markdown_table,
                            "metadata": {
                                **document_metadata,
                                "page_index": page_index,
                                "section_title": title[:200],
                                "section_level": level,
                                "chunk_type": "table",
                                "table_index": table_index,
                                # "table_rows": len(table.get("data", [])),
                                # "table_cols": len(headers),
                                "table_headers": headers_text[:500],  # For BM25 keyword matching
                                "table_csv": csv_table,  # Raw data for post-retrieval processing
                                "chunk_len": len(markdown_table),
                            },
                        }
                    )

    return chunks


def build_document_metadata(file: pd.Series) -> dict:
    return {
        "source": "ssmesr",
        "record_id": file["id"],
        "file_id": file["file_id"],
        "file_name": file["file_name"],
        "file_format": file["file_format"],
        "doc_type": file["subtype"],
        "publication_date": str(file["publication_date"]),
        "publication_epoch": to_unix_epoch(str(file["publication_date"])) if file["publication_date"] else 0,
        # "created": str(file.get("created", "")),
        # "modified": str(file.get("modified", "")),
        "title": file["title"],
        "keywords": (
            ", ".join(file.get("keywords", [])) if isinstance(file.get("keywords"), list) else str(file.get("keywords", ""))
        ),
    }


def transform(use_cache: bool = True) -> list[dict]:
    if use_cache and os.path.exists(OUTPUT_CHUNKS):
        print(f"[transform_ssmesr] Chunks already exist in {OUTPUT_CHUNKS}, skipping")
        return pd.read_json(OUTPUT_CHUNKS, lines=True, encoding="utf-8").to_dict(orient="records")

    records = get_records()
    if records.empty:
        print("[transform_ssmesr] No SSMESR records found")
        return []

    # Get files
    files = get_files(records)
    if files.empty:
        print("[transform_ssmesr] No SSMESR files found")
        return []

    files_with_ocr = files[files["ocr_path"].apply(os.path.exists)]
    print(f"[transform_ssmesr] Found {len(files_with_ocr)} files with OCR")
    if files_with_ocr.empty:
        return []

    chunks: list[dict] = []
    for _, file in files_with_ocr.iterrows():
        metadata = build_document_metadata(file)
        chunks.extend(chunk_document(file["ocr_path"], metadata))

    print(f"[transform_ssmesr] Generated {len(chunks)} SSMESR chunks")
    print(f"  - Paragraphs: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'paragraph')}")
    print(f"  - Tables: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'table')}")

    save_jsonl(chunks, OUTPUT_CHUNKS)
    return chunks


def transform_cli():
    parser = argparse.ArgumentParser(description="Transform SSMESR OCR results into chunked documents (paragraphs + tables)")
    parser.add_argument("--no-cache", action="store_true", help="Force reload of chunks")
    args = parser.parse_args()
    transform(use_cache=not args.no_cache)


if __name__ == "__main__":
    transform_cli()
