import os
import argparse
import pandas as pd
from src.mistral import mistral_ocr
from src.utils import save_jsonl
from src.load import get_records, get_files


def extract_one(file: pd.Series, force_extract: bool = False) -> str:
    file_name = file["file_name"]
    file_path = file["file_path"]
    ocr_path = file["ocr_path"]

    if not ocr_path or not file_path:
        return "failed"

    if not force_extract and os.path.exists(ocr_path):
        return "skipped"

    try:
        data = mistral_ocr(file_path, file_name)
        save_jsonl(data, ocr_path)
        return "extracted"
    except Exception as error:
        print(f"[error] Failed to extract {file_name}: {error}")
        print(f"[debug] {ocr_path=}, {file_path=}")
        return "failed"


def extract_pdf(files: pd.DataFrame, force_extract: bool = False):
    if not len(files):
        print("[extract] Found 0 files to extract")
        return

    pdfs = files[files["file_format"].isin(["pdf"])]
    print(f"[extract] Found {len(pdfs)} pdf from {len(files)} files")

    if not len(pdfs):
        print(f"[extract] Found 0 pdf files from {len(files)} files to extract")
        return

    # Extract pdf files
    stats = pdfs.apply(extract_one, force_extract=force_extract, axis=1)

    # Count stats
    stats_counts = stats.value_counts()
    extracted = int(stats_counts.get("extracted", 0))
    skipped = int(stats_counts.get("skipped", 0))
    failed = int(stats_counts.get("failed", 0))

    print(f"[extract] Extracted {extracted}/{len(pdfs)} pdf files ({skipped=}, {failed=})")


def extract(force_extract: bool = False):
    # Get records
    print("[warn] Only 'article' publications will be extracted")
    records = get_records()
    records = records[records["metadata"].apply(lambda x: x.get("resource_type", {}).get("subtype") == "article")]
    print(f"[extract] Found {len(records)} 'article' records")

    # Get files from records
    files = get_files(records)

    # Extract pdf files
    print("[warn] Only pdf files will be extracted")
    extract_pdf(files, force_extract)


def extract_cli():
    parser = argparse.ArgumentParser(description="Extract data from records files")
    parser.add_argument("--force-extract", action="store_true", help="Force extract")
    args = parser.parse_args()
    extract(force_extract=args.force_extract)


if __name__ == "__main__":
    extract_cli()
