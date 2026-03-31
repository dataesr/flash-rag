import os
import argparse
import pandas as pd
from src.utils import fetch_data, download_file

BASE_URL = "https://zenodo.org/api/records?communities=ssm-esr&size=25&page=1"
OUTPUT_DIR = "./data"
OUTPUT_RECORDS = f"{OUTPUT_DIR}/records.jsonl"


def fetch_records(url: str) -> pd.DataFrame:
    all_records = []
    page = 0
    while url:
        page += 1
        print(f"[load] Fetching page {page}: {url}")
        data = fetch_data(url)
        hits = data.get("hits", {}).get("hits", [])
        all_records.extend(hits)
        print(f"[load]  → Got {len(hits)} hits (total: {len(all_records)})")
        # use pagination
        url = data.get("links", {}).get("next")

    df = pd.DataFrame(all_records)
    df[["created", "modified"]] = df[["created", "modified"]].apply(pd.to_datetime, utc=True, errors="coerce")
    return df


def get_records() -> pd.DataFrame:
    if os.path.exists(OUTPUT_RECORDS):
        records = pd.read_json(OUTPUT_RECORDS, lines=True, encoding="utf-8")
        print(f"[load] Found {len(records)} existing records")
        return records
    print("[load] No existing records found")
    return pd.DataFrame()


def merge_records(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if len(existing) == 0 and len(new) == 0:
        raise ValueError("[load] No existing nor new records found")
    if len(existing) == 0:
        print(f"[load] No existing records found -- using new {len(new)} records")
        return new
    if len(new) == 0:
        print(f"[load] No new records found -- continue with existing {len(existing)} records")
        return existing

    all = pd.concat([existing, new])
    all["modified"] = all["modified"].apply(pd.to_datetime, utc=True, errors="coerce")
    merged = all.sort_values("modified").drop_duplicates(subset="id", keep="last").reset_index(drop=True)

    print(f"[load] Merged records: {len(merged)} ({len(existing)=}, {len(new)=})")
    return merged


def download_files(records: pd.DataFrame, force_download: bool = True):
    downloaded = 0
    skipped = 0
    failed = 0
    total = 0
    total_records = len(records)

    print(f"[load] Starting to download files from {total_records} records")

    for index, record in records.iterrows():
        files = record.get("files", [])
        for file in files:
            total += 1
            file_name = file.get("key")
            file_link = file.get("links", {}).get("self")
            if not file_name or not file_link:
                failed += 1
                continue

            extension = file_name.split(".")[-1]
            file_path = f"{OUTPUT_DIR}/{extension.lower()}/{file_name}"
            if force_download or not os.path.exists(file_path):
                try:
                    download_file(file_link, file_path)
                    downloaded += 1
                except Exception as error:
                    print(f"[error] Failed to download {file_name}: {error}")
                    failed += 1
                    continue
            else:
                skipped += 1
                continue

    print(f"[load] Downloaded {downloaded}/{total} files ({skipped=}, {failed=})")


def load():
    parser = argparse.ArgumentParser(description="Load records")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching records")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading files")
    parser.add_argument("--force-download", action="store_true", help="Force download files")
    args = parser.parse_args()

    # Get existing records
    existing = get_records()

    # Fetch new records
    new = pd.DataFrame()
    if not args.skip_fetch:
        new = fetch_records(BASE_URL)

    # Merge with existing
    records = merge_records(existing, new)

    # Download files if needed
    if not args.skip_download:
        download_files(records, args.force_download)

    # Save records
    records.to_json(OUTPUT_RECORDS, orient="records", lines=True)
    print(f"[load] Saved {len(records)} records to {OUTPUT_RECORDS}")


if __name__ == "__main__":
    load()
