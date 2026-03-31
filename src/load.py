import argparse
from src.utils import fetch_data, save_jsonl

BASE_URL = "https://zenodo.org/api/records?communities=ssm-esr&size=25&page=1"
OUTPUT_FILE = "data/records.jsonl"


def fetch_all_records(url: str) -> list[dict]:
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

    return all_records


def load():
    parser = argparse.ArgumentParser(description="Load records")
    parser.add_argument("--output", "-o", help="Override output path (default: data/records.jsonl)")
    args = parser.parse_args()

    output_path = args.output or OUTPUT_FILE
    records = fetch_all_records(BASE_URL)
    save_jsonl(records, output_path)


if __name__ == "__main__":
    load()
