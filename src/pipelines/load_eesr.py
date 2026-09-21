import os
import logging
import argparse
import pandas as pd
from zipfile import ZipFile
from src.utils import load_jsonl

logger = logging.getLogger(__name__)

OUTPUT_DIR = "./data"
EESR_PUBLICATIONS_CODES = ["PAGE_EESR19"]
OUTPUT_PAGES = f"{OUTPUT_DIR}/eesr_pages.jsonl"


def publication_get_pages(code: str):

    # Check the data
    dir_path = f"{OUTPUT_DIR}/{code}"
    if not os.path.exists(dir_path):
        # Check if zip file exists
        zip_path = f"{code}.zip"
        if not os.path.exists(zip_path):
            logger.error(f"{zip_path} does not exist")
            return

        # Unzip the file
        try:
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dir_path)
        except Exception as error:
            logger.error(f"Failed to unzip {zip_path}: {error}")
            return

        if not os.path.isdir(dir_path):
            logger.error(f"{dir_path} is not a directory after unzipping")
            return

    if not os.path.isdir(dir_path):
        logger.error(f"{dir_path} is not a directory")
        return

    # Load the publications pages
    data = []
    skipped_pages = 0
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(dir_path, file_name)
            try:
                page_data = load_jsonl(file_path)
                # logger.debug(f"page_data: {page_data}")
                if not isinstance(page_data, dict):
                    logger.error(f"Expected a dict in {file_path}, got {type(page_data)}")
                    continue

                page_data["PAGE_FILE_NAME"] = file_name  # Add the file name to the page data for reference
                data.append(page_data)

            except Exception as error:
                logger.error(f"Failed to load JSON from {file_path}: {error}")

    if not data:
        logger.info(f"No data loaded from {dir_path}")
        return

    logger.info(f"Loaded {len(data)} pages from {dir_path} (skipped={skipped_pages})")
    return data


def get_pages() -> pd.DataFrame:
    if os.path.exists(OUTPUT_PAGES):
        pages = pd.read_json(OUTPUT_PAGES, lines=True, encoding="utf-8")
        logger.info(f"Found {len(pages)} pages")
        return pages
    logger.info("No pages found")
    return pd.DataFrame()


def load(use_cache: bool = True, use_fetch: bool = True, force_download: bool = False):

    if use_cache and os.path.exists(OUTPUT_PAGES):
        logger.info(f"Pages already loaded in {OUTPUT_PAGES}, skipping")
        return

    all_pages = []
    for code in EESR_PUBLICATIONS_CODES:
        pages = publication_get_pages(code)
        if pages:
            all_pages.extend(pages)

    if not all_pages:
        logger.info("No pages loaded, nothing to save")
        return

    pages_df = pd.DataFrame(all_pages)
    pages_df.to_json(OUTPUT_PAGES, orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved {len(pages_df)} pages to {OUTPUT_PAGES}")


def load_cli():
    parser = argparse.ArgumentParser(description="Load pages")
    parser.add_argument("--no-cache", action="store_true", help="Force reload of data")
    args = parser.parse_args()
    load(use_cache=not args.no_cache)


if __name__ == "__main__":
    load_cli()
