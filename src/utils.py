import os
import json
import httpx


def save_jsonl(data: list[dict] | dict | None, output_path: str):
    """Save a list of dicts as a JSON or JSONL file."""
    if not data:
        print("[save_jsonl] No data to save")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if isinstance(data, dict):
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[save_jsonl] Saved data to {output_path}")
        elif isinstance(data, list):
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[save_jsonl] Saved {len(data)} records to {output_path}")
        else:
            print(f"[save_jsonl] Unsupported type: {type(data)}")


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
