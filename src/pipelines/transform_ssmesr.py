from src.pipelines.load_eesr import EESR_TITLE
import os
import re
import json
import logging
import argparse
import pandas as pd
from src.pipelines.load_ssmesr import OCR_DIR, get_records, get_files
from src.query import MIN_CHUNK_LEN, MAX_CHUNK_LEN
from src.utils import save_jsonl, to_unix_epoch, split_text

logger = logging.getLogger(__name__)

OUTPUT_DIR = "./data"
OUTPUT_CHUNKS = f"{OUTPUT_DIR}/ssmesr_chunks.jsonl"


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


def chunk_one_page(page: dict, document_metadata: dict) -> list[dict]:
    file_name = document_metadata["file_name"]
    file_name_no_ext = file_name.split(".")[0] if "." in file_name else file_name

    # mistral ocr 4 patterns
    table_pattern = re.compile(r"\[(tbl-[^]]+\.md)\]\([^)]+\.md\)")
    image_pattern = re.compile(r"!\[[^\]]+\.jpeg\]\([^)]+\.jpeg\)")
    superscript_pattern = re.compile(r"\$\^\{\d+\}\$")
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)")

    page_index = page.get("index", 0) + 1  # Start at 1
    markdown = page.get("markdown", "")

    if not markdown:
        logger.debug(f"[{file_name}] No markdown found for page={page_index} → skipping")
        return []

    chunks = []
    tables = {tbl["id"]: tbl for tbl in page.get("tables", [])}

    current_doc = ""
    section_title = ""
    section_level = 0
    section_index = 0
    chunk_index = 0
    has_table = False

    def flush_chunk():
        """Flush current buffer into a chunk"""
        nonlocal current_doc, chunk_index, section_title, section_level, has_table
        current_doc = current_doc.strip()
        if not current_doc:
            return

        if len(current_doc) < MIN_CHUNK_LEN:
            logger.warning(f"[{file_name}] Small chunk at page={page_index}, section={section_index} → skipping")
            logger.debug(f"chunk={current_doc}")
            return

        next_chunks = split_text(current_doc, MAX_CHUNK_LEN)
        for next_doc in next_chunks:
            chunk_index += 1
            chunks.append(
                {
                    "id": f"ssmesr_{file_name_no_ext}_p{page_index}_s{section_index}_c{chunk_index}",
                    "document": next_doc,
                    "metadata": {
                        **document_metadata,
                        "page_index": page_index,
                        "section_index": section_index,
                        "section_title": section_title[:200],
                        "section_level": section_level,
                        "chunk_len": len(next_doc),
                        "chunk_type": "table" if has_table else "paragraph",
                    },
                }
            )

        # Reset buffer
        current_doc = ""
        has_table = False

    for para in markdown.split("\n\n"):
        para = para.strip()
        if not para:
            continue

        lines = para.split("\n")
        first_line = lines[0]
        heading_match = heading_pattern.match(first_line)

        if heading_match:
            # New section: flush current chunk and reset
            flush_chunk()
            chunk_index = 0

            section_index += 1
            section_level = len(heading_match.group(1))
            title_parts = [heading_match.group(2)]

            # Collect multi-line section title
            for line in lines[1:]:
                line = line.strip()
                if line:
                    title_parts.append(line)
                else:
                    break
            section_title = " ".join(title_parts)

        else:

            # Check for tables in this paragraph
            if table_pattern.search(para):
                has_table = True

            # Process paragraph and add to chunk
            processed = para
            processed = table_pattern.sub(lambda m: tables.get(m.group(1), {}).get("content", ""), processed)
            processed = image_pattern.sub("", processed)
            processed = superscript_pattern.sub("", processed)
            processed = re.sub(r"\n{2,}", "\n\n", processed).strip()

            if not processed:
                continue

            if has_table and len(processed) > MAX_CHUNK_LEN:
                logger.warning(f"[{file_name}] Large table at page={page_index}, section={section_index} → skipping")
                continue

            # Try to add to current chunk
            next_doc = (current_doc + "\n\n" + processed) if current_doc else processed
            if len(next_doc) <= MAX_CHUNK_LEN:
                current_doc = next_doc
            else:
                # Chunk is full: flush and start new
                flush_chunk()
                current_doc = processed

    # Flush final chunk
    flush_chunk()
    return chunks


def chunk_document(ocr_path: str, document_metadata: dict) -> list[dict]:

    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as error:
        logger.error(f"Error loading {ocr_path}: {error}")
        return []

    pages = data.get("pages", [])
    if not pages:
        logger.debug(f"No pages found in {ocr_path}")
        return []

    chunks = []
    for page in pages:
        page_chunks = chunk_one_page(page, document_metadata)
        if page_chunks:
            chunks.extend(page_chunks)

    return chunks


def build_document_metadata(file: pd.Series) -> dict:
    keywords = file.get("keywords") or []
    keywords = [k.lower() for k in keywords if k] if isinstance(keywords, list) else []

    metadata = {
        "title": file["title"],
        "reference": "ssmesr",
        "record_id": file["id"],
        "publication_type": file["subtype"],
        "publication_date": str(file["publication_date"]),
        "publication_epoch": to_unix_epoch(str(file["publication_date"])) if file["publication_date"] else 0,
        "file_id": file["file_id"],
        "file_name": file["file_name"],
        "file_format": file["file_format"],
        "file_url": file["doi_url"],
        "file_access": file.get("access_right", ""),
    }
    if len(keywords):
        metadata["keywords"] = keywords

    return metadata


def transform() -> list[dict]:
    records = get_records()
    if records.empty:
        logger.info("No SSMESR records found")
        return []

    # Get files
    files = get_files(records)
    if files.empty:
        logger.info("No SSMESR files found")
        return []

    files_with_ocr = files[files["ocr_path"].apply(os.path.exists)]
    logger.info(f"Found {len(files_with_ocr)} files with OCR")
    if files_with_ocr.empty:
        return []

    chunks: list[dict] = []
    for _, file in files_with_ocr.iterrows():
        metadata = build_document_metadata(file)
        chunks.extend(chunk_document(file["ocr_path"], metadata))

    logger.info(f"Generated {len(chunks)} SSMESR chunks")
    logger.info(f"  - Paragraphs: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'paragraph')}")
    # logger.info(f"  - Tables: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'table')}")

    save_jsonl(chunks, OUTPUT_CHUNKS)
    return chunks


def transform_cli():
    parser = argparse.ArgumentParser(description="Transform SSMESR OCR results into chunked documents (paragraphs + tables)")
    # parser.add_argument("--no-cache", action="store_true", help="Force reload of chunks")
    args = parser.parse_args()
    transform()


if __name__ == "__main__":
    transform_cli()
