import os
import json
import httpx


def save_jsonl(records: list[dict], output_path: str):
    """Save a list of dicts as a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {output_path}")


def load_jsonl(input_path: str) -> list[dict]:
    """Load a list of dicts from a JSONL file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fetch_data(url: str) -> dict:
    """Fetch a URL and return the JSON response."""
    with httpx.Client() as client:
        try:
            response = client.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            print(f"[error] An error occurred while requesting {url}: {error}")
            raise error


def download_file(url: str, output_path: str):
    """Download a file from a URL and save it."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with httpx.Client() as client:
        with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
