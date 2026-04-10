import re
import argparse
import pandas as pd
from src.load import get_records, get_files
from src.utils import save_jsonl, load_jsonl


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


def parse_one_ocr(file: pd.Series, force_parse: bool = False) -> pd.Series:
    file_name = file["file_name"]
    file_path = file["file_path"]
    file_format = file["file_format"]
    ocr_path = file["ocr_path"]

    results = pd.Series({"parsed": 0, "skipped": 0, "failed": 0, "empty": 0, "total": 0, "file": "failed"})

    if not file_format == "pdf":
        print(f"[parse] Skipping {file_name} ({file_format=})")
        results["file"] = "skipped"
        return results

    if not ocr_path:
        print(f"[parse] No ocr_path found for {file_name} ({file_path=})")
        return results

    ocr_data = load_jsonl(ocr_path)
    if not ocr_data:
        print(f"[parse] No data found in {ocr_path}")
        return results

    if not isinstance(ocr_data, dict):
        print(f"[parse] Invalid data type in {ocr_path} ({type(ocr_data)=})")
        return results

    ocr_pages = ocr_data.get("pages")
    if not ocr_pages:
        print(f"[parse] No pages found in {ocr_path}")
        return results

    if not isinstance(ocr_pages, list):
        print(f"[parse] Invalid data type in {ocr_path} ({type(ocr_pages)=})")
        return results

    # print(f"[debug] ocr_pages: {len(ocr_pages)}")
    results["total"] = len(ocr_pages)

    for page in ocr_pages:
        md = page.get("markdown")
        parsed = page.get("parsed")

        if not force_parse and parsed:
            results["skipped"] += 1
            continue

        if not md:
            print(f"[parse] No markdown found in page {page['index']} of {ocr_path}")
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


def parse_ocr(files: pd.DataFrame, force_parse: bool = False):
    if not len(files):
        print("[parse] Found 0 files to parse")
        return

    # Parse pdf files
    stats = files.apply(parse_one_ocr, force_parse=force_parse, axis=1)

    # Count stats
    parsed = int(stats["parsed"].sum())
    skipped = int(stats["skipped"].sum())
    failed = int(stats["failed"].sum())
    empty = int(stats["empty"].sum())
    total = int(stats["total"].sum())
    print(f"[parse] Parsed {len(files)} files")
    print(f"[parse] Parsed {parsed}/{total} pages ({skipped=}, {failed=}, {empty=})")


# def get_parsed() -> pd.DataFrame:
#     if os.path.exists(OUTPUT_PARSED):
#         parsed = pd.read_json(OUTPUT_PARSED, lines=True, encoding="utf-8")
#         print(f"[parse] Found {len(parsed)} existing parsed files")
#         return parsed
#     print("[parse] No existing parsed files found")
#     return pd.DataFrame()


def parse(force_parse: bool = False):

    # Get records
    records = get_records()

    # Get files from records
    files = get_files(records)

    # Parse pdf files
    parse_ocr(files, force_parse)


def parse_cli():
    parser = argparse.ArgumentParser(description="Parse data from ocr files")
    parser.add_argument("--force-parse", action="store_true", help="Force parse")
    args = parser.parse_args()
    parse(force_parse=args.force_parse)


if __name__ == "__main__":
    parse_cli()
