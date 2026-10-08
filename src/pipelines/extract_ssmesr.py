import os
import re
import logging
import argparse
import pandas as pd
from src.mistral import mistral_ocr
from src.utils import save_jsonl, load_jsonl
from src.pipelines.load_ssmesr import get_records, get_files
from src.pipelines.load_eesr import EESR_TITLE

logger = logging.getLogger(__name__)


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
        if not data:
            logger.error(f"No OCR data returned for {file_name}")
            return "failed"
        save_jsonl(data, ocr_path)
        return "extracted"
    except Exception as error:
        logger.error(f"Failed to extract {file_name}: {error}")
        logger.debug(f"{ocr_path=}, {file_path=}")
        return "failed"


def extract_pdf(files: pd.DataFrame, use_cache: bool = True):
    if not len(files):
        logger.info("Found 0 files to extract")
        return

    pdfs = files[files["file_format"].isin(["pdf"])]

    if not len(pdfs):
        logger.info(f"Found 0 pdf files from {len(files)} files to extract")
        return

    logger.info(f"Found {len(pdfs)} pdf from {len(files)} files")

    # Extract pdf files
    stats = pdfs.apply(extract_one, use_cache=use_cache, axis=1)

    # Count stats
    stats_counts = stats.value_counts()
    extracted = int(stats_counts.get("extracted", 0))
    skipped = int(stats_counts.get("skipped", 0))
    failed = int(stats_counts.get("failed", 0))

    logger.info(f"Extracted {extracted}/{len(pdfs)} pdf files ({skipped=}, {failed=})")


def extract(use_cache: bool = True):
    # Get records
    records = get_records()

    # logger.warning("Only 'article' publications will be extracted")
    # records = records[records["metadata"].apply(lambda x: x.get("resource_type", {}).get("subtype") == "article")]
    # logger.info(f"Found {len(records)} 'article' records")
    records = records[~records["title"].apply(lambda title: EESR_TITLE.lower() in title.lower())]
    logger.debug(f"Skip EESR records - added separately (remaining records={len(records)})")

    # Get files from records
    files = get_files(records)

    # Extract pdf files
    logger.warning("Only pdf files will be extracted")
    extract_pdf(files, use_cache=use_cache)


def extract_cli():
    parser = argparse.ArgumentParser(description="Extract data from records files using OCR")
    parser.add_argument("--no-cache", action="store_true", help="Force mistral OCR")
    args = parser.parse_args()

    # Extract and parse pdf files
    extract(use_cache=not args.no_cache)


if __name__ == "__main__":
    extract_cli()
