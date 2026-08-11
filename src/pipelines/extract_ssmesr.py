import os
import re
import argparse
import pandas as pd
from src.mistral import mistral_ocr
from src.utils import save_jsonl, load_jsonl
from src.pipelines.load_ssmesr import get_records, get_files


def parse_table(md: str) -> dict | None:
    lines = [line.strip() for line in md.strip().splitlines()]
    # Remove separator line
    lines = [line for line in lines if not re.match(r"^\|[-:\s|]+\|$", line)]

    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return None

    headers = rows[0]
    data = rows[1:]

    return {"headers": headers, "data": data}


def parse_markdown(md: str) -> list[dict] | None:
    sections: list[dict] = []
    current: dict | None = None
    buffer: list[str] = []

    if not md:
        return None

    def flush_buffer():
        if current is None or not buffer:
            return
        block = "\n".join(buffer).strip()
        if not block:
            return

        # Split block into chunks separated by blank lines (2 or more \n)
        for chunk in re.split(r"\n{2,}", block):
            chunk = chunk.strip()
            if not chunk:
                continue

            lines = chunk.splitlines()
            # A table has at least one | and a separator line with --- or : between two |
            is_table = any("|" in line for line in lines) and any(re.match(r"^\s*\|[\s\-|:]+\|\s*$", line) for line in lines)

            if is_table:
                current["tables"].append(parse_table(chunk))
            else:
                if chunk:
                    current["paragraphs"].append(chunk)
        buffer.clear()

    for line in md.splitlines():
        heading = re.match(r"^(#{1,4})\s+(.*)", line)  # 1 to 4 # followed by a space and the title
        if heading:
            flush_buffer()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            current = {"level": level, "title": title, "paragraphs": [], "tables": []}
            sections.append(current)
        else:
            if current is not None:
                buffer.append(line)

    flush_buffer()
    return sections


def parse_one_ocr(file: pd.Series, use_cache: bool = True) -> pd.Series:
    file_name = file["file_name"]
    file_path = file["file_path"]
    file_format = file["file_format"]
    ocr_path = file["ocr_path"]

    results = pd.Series({"parsed": 0, "skipped": 0, "failed": 0, "empty": 0, "total": 0, "file": "failed"})

    # Only pdf
    if not file_format == "pdf":
        print(f"[parse-ssmesr] Skipping {file_name} ({file_format=})")
        results["file"] = "skipped"
        return results

    if not ocr_path:
        print(f"[parse-ssmesr] No ocr_path found for {file_name} ({file_path=})")
        return results

    ocr_data = load_jsonl(ocr_path)
    if not ocr_data:
        print(f"[parse-ssmesr] No data found in {ocr_path}")
        return results

    if not isinstance(ocr_data, dict):
        print(f"[parse-ssmesr] Invalid data type in {ocr_path} ({type(ocr_data)=})")
        return results

    ocr_pages = ocr_data.get("pages")
    if not ocr_pages:
        print(f"[parse-ssmesr] No pages found in {ocr_path}")
        return results

    if not isinstance(ocr_pages, list):
        print(f"[parse-ssmesr] Invalid data type in {ocr_path} ({type(ocr_pages)=})")
        return results

    # print(f"[debug] ocr_pages: {len(ocr_pages)}")
    results["total"] = len(ocr_pages)

    for page in ocr_pages:
        md = page.get("markdown")
        parsed = page.get("parsed")

        if use_cache and parsed:
            results["skipped"] += 1
            continue

        if not md:
            print(f"[parse-ssmesr] No markdown found in page {page['index']} of {ocr_path}")
            results["empty"] += 1
            continue

        try:
            sections = parse_markdown(md)
            if sections:
                page["parsed"] = sections
                results["parsed"] += 1
            else:
                results["empty"] += 1
        except Exception as error:
            print(f"[error] Failed to parse page {page['index']} of {ocr_path}: {error}")
            results["failed"] += 1
            continue

    if results["parsed"] == 0:
        if results["skipped"] > 0 and results["skipped"] == results["total"]:
            results["file"] = "skipped"
        return results

    results["file"] = "parsed"
    ocr_data["pages"] = ocr_pages
    save_jsonl(ocr_data, ocr_path)

    return results


def parse_ocr(files: pd.DataFrame, use_cache: bool = True):
    if not len(files):
        print("[parse-ssmesr] Found 0 files to parse")
        return

    # Parse pdf files
    stats = files.apply(parse_one_ocr, use_cache=use_cache, axis=1)

    # Count stats
    parsed = int(stats["parsed"].sum())
    skipped = int(stats["skipped"].sum())
    failed = int(stats["failed"].sum())
    empty = int(stats["empty"].sum())
    total = int(stats["total"].sum())
    print(f"[parse-ssmesr] Parsed {len(files)} files")
    print(f"[parse-ssmesr] Parsed {parsed}/{total} pages ({skipped=}, {failed=}, {empty=})")


def extract_one(file: pd.Series, use_cache: bool = True) -> str:
    file_name = file["file_name"]
    file_path = file["file_path"]
    ocr_path = file["ocr_path"]

    if not ocr_path or not file_path:
        return "failed"

    if use_cache and os.path.exists(ocr_path):
        return "skipped"

    try:
        data = mistral_ocr(file_path, file_name)
        save_jsonl(data, ocr_path)
        return "extracted"
    except Exception as error:
        print(f"[error] Failed to extract {file_name}: {error}")
        print(f"[debug] {ocr_path=}, {file_path=}")
        return "failed"


def extract_pdf(files: pd.DataFrame, use_cache: bool = True):
    if not len(files):
        print("[extract-ssmesr] Found 0 files to extract")
        return

    pdfs = files[files["file_format"].isin(["pdf"])]
    print(f"[extract-ssmesr] Found {len(pdfs)} pdf from {len(files)} files")

    if not len(pdfs):
        print(f"[extract-ssmesr] Found 0 pdf files from {len(files)} files to extract")
        return

    # Extract pdf files
    stats = pdfs.apply(extract_one, use_cache=use_cache, axis=1)

    # Count stats
    stats_counts = stats.value_counts()
    extracted = int(stats_counts.get("extracted", 0))
    skipped = int(stats_counts.get("skipped", 0))
    failed = int(stats_counts.get("failed", 0))

    print(f"[extract-ssmesr] Extracted {extracted}/{len(pdfs)} pdf files ({skipped=}, {failed=})")


def extract(use_cache: bool = True):
    # Get records
    print("[warn] Only 'article' publications will be extracted")
    records = get_records()
    records = records[records["metadata"].apply(lambda x: x.get("resource_type", {}).get("subtype") == "article")]
    print(f"[extract-ssmesr] Found {len(records)} 'article' records")

    # Get files from records
    files = get_files(records)

    # Extract pdf files
    print("[warn] Only pdf files will be extracted")
    extract_pdf(files, use_cache)

    # Parse ocr results
    parse_ocr(files, use_cache)


def extract_cli():
    parser = argparse.ArgumentParser(description="Extract data from records files using OCR")
    parser.add_argument("--no-cache", action="store_true", help="Force extract and parsing")
    args = parser.parse_args()

    # Extract and parse pdf files
    extract(use_cache=not args.use_cache)


if __name__ == "__main__":
    extract_cli()
