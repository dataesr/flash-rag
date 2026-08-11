import os
import argparse
import pandas as pd
from zipfile import ZipFile
from src.utils import load_jsonl

OUTPUT_DIR = "./data"
EESR_PUBLICATIONS_CODES = ["PAGE_EESR19"]
OUTPUT_PAGES = f"{OUTPUT_DIR}/eesr_pages.jsonl"


def publication_get_pages(code: str):

    # Check the data
    dir_path = f"{OUTPUT_DIR}/{code}"
    if not os.path.exists(dir_path):
        # Check if zip file exists
        zip_path = f"{dir_path}.zip"
        if not os.path.exists(zip_path):
            print(f"[error] {zip_path} does not exist")
            return

        # Unzip the file
        try:
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dir_path)
        except Exception as error:
            print(f"[error] Failed to unzip {zip_path}: {error}")
            return

        if not os.path.isdir(dir_path):
            print(f"[error] {dir_path} is not a directory after unzipping")
            return

    if not os.path.isdir(dir_path):
        print(f"[error] {dir_path} is not a directory")
        return

    # Load the publications pages
    data = []
    skipped_pages = 0
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(dir_path, file_name)
            try:
                page_data = load_jsonl(file_path)
                # print(f"[debug] page_data: {page_data}")
                if not isinstance(page_data, dict):
                    print(f"[error] Expected a dict in {file_path}, got {type(page_data)}")
                    continue

                # page_courant_id = page_data.get("PAGE_COURANTE_ID", 0)
                # if page_courant_id <= 1:
                #     # Skip pages with PAGE_COURANTE_ID <= 1 (annexes or resumes)
                #     skipped_pages += 1
                #     continue
                page_data["PAGE_FILE_NAME"] = file_name  # Add the file name to the page data for reference

                data.append(page_data)
            except Exception as error:
                print(f"[error] Failed to load JSON from {file_path}: {error}")

    if not data:
        print(f"[load-eesr] No data loaded from {dir_path}")
        return

    print(f"[load-eesr] Loaded {len(data)} pages from {dir_path} (skipped={skipped_pages})")
    return data


def get_pages() -> pd.DataFrame:
    if os.path.exists(OUTPUT_PAGES):
        pages = pd.read_json(OUTPUT_PAGES, lines=True, encoding="utf-8")
        print(f"[load-eesr] Found {len(pages)} pages")
        return pages
    print("[load-eesr] No pages found")
    return pd.DataFrame()


def load(use_cache: bool = True):

    if use_cache and os.path.exists(OUTPUT_PAGES):
        print(f"[load-eesr] Pages already loaded in {OUTPUT_PAGES}, skipping")
        return

    all_pages = []
    for code in EESR_PUBLICATIONS_CODES:
        pages = publication_get_pages(code)
        if pages:
            all_pages.extend(pages)

    if not all_pages:
        print("[load-eesr] No pages loaded, nothing to save")
        return

    pages_df = pd.DataFrame(all_pages)
    pages_df.to_json(OUTPUT_PAGES, orient="records", lines=True, force_ascii=False)
    print(f"[load-eesr] Saved {len(pages_df)} pages to {OUTPUT_PAGES}")


def load_cli():
    parser = argparse.ArgumentParser(description="Load pages")
    parser.add_argument("--no-cache", action="store_true", help="Force reload of data")
    args = parser.parse_args()
    load(use_cache=not args.no_cache)


if __name__ == "__main__":
    load_cli()
