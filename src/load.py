import os
import argparse
import pandas as pd
from src.utils import fetch_data, download_file

BASE_URL = "https://zenodo.org/api/records?communities=ssm-esr&size=25&page=1"
OUTPUT_DIR = "./data"
OCR_DIR = f"{OUTPUT_DIR}/ocr"
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
    df[["created", "modified"]] = df[["created", "modified"]].apply(
        pd.to_datetime, format="%Y-%m-%d", utc=True, errors="coerce"
    )
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
    all["modified"] = all["modified"].apply(pd.to_datetime, format="%Y-%m-%d", utc=True, errors="coerce")
    merged = all.sort_values("modified").drop_duplicates(subset="id", keep="last").reset_index(drop=True)

    print(f"[load] Merged records: {len(merged)} ({len(existing)=}, {len(new)=})")
    return merged


def get_files(records: pd.DataFrame) -> pd.DataFrame:
    if not len(records):
        print("[load] Records dataframe is empty")
        return pd.DataFrame()

    if "files" not in records.columns:
        print("[load] No column 'files' found on records dataframe")
        return pd.DataFrame()

    # Explode df on files column
    try:
        exploded = records.explode("files", ignore_index=True)
        files_data = (
            pd.json_normalize(exploded["files"], sep="_")
            .add_prefix("file_")
            .rename(columns={"file_key": "file_name", "file_links_self": "file_url"})
            .reset_index(drop=True)
        )
        # Build file paths
        files_data["file_format"] = files_data["file_name"].apply(lambda x: x.split(".")[-1])
        files_data["file_path"] = OUTPUT_DIR + "/" + files_data["file_format"] + "/" + files_data["file_name"]
        files_data["ocr_name"] = files_data["file_name"].apply(lambda x: x.split(".")[0]) + ".json"
        files_data["ocr_path"] = OCR_DIR + "/" + files_data["file_format"] + "/" + files_data["ocr_name"]

        # Get metadata
        metadata = exploded["metadata"]
        metadata_data = pd.json_normalize(metadata)[["title", "publication_date", "description", "keywords"]].reset_index(drop=True)

        # Get resource types
        resource_types = metadata.apply(lambda x: x.get("resource_type") if isinstance(x, dict) else None)
        types_data = pd.json_normalize(resource_types).rename(columns={"title": "type_title"}).reset_index(drop=True)

        # Merge files
        files = pd.concat([exploded[["id", "created", "modified"]], files_data, metadata_data, types_data], axis=1)
    except Exception as error:
        print(f"[error] Error while exploding files: {error}")
        raise error

    print(f"[load] Found {len(files)} files from {len(records)} records")
    return files


def download_one_file(file: pd.Series, force_download: bool = True) -> str:
    url = file["file_url"]
    path = file["file_path"]
    name = file["file_name"]

    if not url or not path:
        return "failed"
    if name == "empty_file.txt":
        return "skipped"
    if not force_download and os.path.exists(path):
        return "skipped"
    try:
        download_file(url, path)
        return "downloaded"
    except Exception as error:
        print(f"[error] Failed to download {name}: {error}")
        print(f"[debug] {url=}, {path=}")
        return "failed"


def download_files(records: pd.DataFrame, force_download: bool = True, formats: list[str] = []):
    total_records = len(records)

    print(f"[load] Starting to download files from {total_records} records")

    # Get files from records
    files = get_files(records)
    if formats:
        files = files[files["file_format"].isin(formats)]

    if not len(files):
        print(f"[load] Found 0 files from {len(records)} records to download")
        return

    # Download files
    stats = files.apply(download_one_file, force_download=force_download, axis=1)

    # Count stats
    stats_counts = stats.value_counts()
    downloaded = int(stats_counts.get("downloaded", 0))
    skipped = int(stats_counts.get("skipped", 0))
    failed = int(stats_counts.get("failed", 0))

    print(f"[load] Downloaded {downloaded}/{len(files)} files ({skipped=}, {failed=})")


def load(skip_fetch: bool = False, skip_download: bool = False, force_download: bool = False):

    # Get existing records
    existing = get_records()

    # Fetch new records
    new = pd.DataFrame()
    if not skip_fetch:
        new = fetch_records(BASE_URL)

    # Merge with existing
    records = merge_records(existing, new)

    # Download files if needed
    if not skip_download:
        download_files(records, force_download)

    # Save records
    records.to_json(OUTPUT_RECORDS, orient="records", lines=True)
    print(f"[load] Saved {len(records)} records to {OUTPUT_RECORDS}")


def load_cli():
    parser = argparse.ArgumentParser(description="Load records")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching records")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading files")
    parser.add_argument("--force-download", action="store_true", help="Force download files")
    args = parser.parse_args()
    load(skip_fetch=args.skip_fetch, skip_download=args.skip_download, force_download=args.force_download)


if __name__ == "__main__":
    load_cli()
