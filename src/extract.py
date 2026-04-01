import argparse
import pandas as pd
from src.load import get_records


def get_files(records: pd.DataFrame) -> pd.DataFrame:
    if not len(records):
        print("[extract] Records dataframe is empty")
        return pd.DataFrame()

    if "files" not in records.columns:
        print("[extract] No column 'files' found on records dataframe")
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
        files_data["format"] = files_data["file_name"].apply(lambda x: x.split(".")[-1])
        resource_types = exploded["metadata"].apply(lambda x: x.get("resource_type") if isinstance(x, dict) else None)
        types_data = pd.json_normalize(resource_types).rename(columns={"title": "type_title"}).reset_index(drop=True)
        files = pd.concat([exploded[["id", "modified"]], files_data, types_data], axis=1)
    except Exception as error:
        print(f"[error] Error while exploding files: {error}")
        raise error

    print(f"[extract] Found {len(files)} files from {len(records)} records")
    return files


def extract_pdf(files: pd.DataFrame):
    pdfs = files[files["format"].isin(["pdf"])]
    print(f"[extract] Found {len(pdfs)} pdf from {len(files)} files")

    if len(pdfs):
        print("[debug] dry run")
        pass

    print(f"[extract] Extracted {len(pdfs)} pdf files")


def extract(args=None):
    parser = argparse.ArgumentParser(description="Extract data from records files")
    parser.add_argument("--force-extract", action="store_true", help="Force extract")
    args = parser.parse_args(args)

    # Get records
    records = get_records()
    print("[warn] Only publications will be extracted")
    records = records[records["type"].isin(["publication"])]

    # Get files from records
    files = get_files(records)

    # Extract pdf files
    print("[warn] Only pdf files will be extracted")
    extract_pdf(files)


if __name__ == "__main__":
    extract()
