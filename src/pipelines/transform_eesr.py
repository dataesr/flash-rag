import os
import argparse
from typing import Any
import pandas as pd
from src.pipelines.load_eesr import get_pages
from src.utils import save_jsonl, to_unix_epoch

OUTPUT_DIR = "./data"
OUTPUT_CHUNKS = f"{OUTPUT_DIR}/eesr_chunks.jsonl"
CHUNK_MAX_CHARS = 3000
CONTENT_FIELDS = ["PAGE_CHAPEAU_FR", "PAGE_TEXTE_FR", "PAGE_METHODE_FR", "PAGE_NOTES_FR"]


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    if not text or not isinstance(text, str):
        return []

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        if not current_chunk:
            current_chunk = paragraph
            continue

        candidate = current_chunk + "\n\n" + paragraph
        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_eesr_metadata(page: dict[str, Any]) -> dict[str, Any]:
    publication = page.get("PUBLICATION") or {}
    publication_date = publication.get("PUBLICATION_DATE_TRI") or publication.get("PUBLICATION_DATE_ANNEE") or ""
    publication_date = publication_date + "-01" if publication_date and len(publication_date) == 7 else publication_date
    publication_epoch = to_unix_epoch(publication_date) if publication_date else 0

    page_id = page["PAGE_NOM_DE_CODE"].lower().replace("eesr", "")

    return {
        "source": "eesr",
        "page_id": page_id,
        "file_name": page.get("PAGE_FILE_NAME", ""),
        "file_format": "json",
        "doc_type": page.get("PAGE_TYPE_NOM", page.get("PAGE_TYPE_ID", "page")),
        "title": page.get("PAGE_TITRE_FR") or page.get("PAGE_TITRE_EN") or "",
        # "created": "",
        # "modified": "",
        "publication_date": publication_date,
        "publication_epoch": publication_epoch,
        # "keywords": "",
        "chapitre": page.get("PAGE_CHAPITRE_FR") or page.get("PAGE_CHAPITRE_EN") or "",
    }


def build_page_text(page: dict[str, Any]) -> str:
    parts: list[str] = []
    title = page.get("PAGE_TITRE_FR") or page.get("PAGE_TITRE_EN")
    if title:
        parts.append(title)

    for field in CONTENT_FIELDS:
        value = page.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if not parts:
        print(f"[error] No content found for page {page['PAGE_ID']}")
        return ""

    return "\n\n".join(parts)


def page_to_chunks(page: dict[str, Any]) -> list[dict[str, Any]]:
    text = build_page_text(page)
    if not text:
        return []

    metadata = build_eesr_metadata(page)
    page_id = metadata["page_id"]
    chunk_texts = chunk_text(text)

    chunks: list[dict[str, Any]] = []
    for chunk_idx, document in enumerate(chunk_texts):
        chunks.append(
            {
                "id": f"eesr_{page_id}_{chunk_idx}",
                "document": document,
                "metadata": {
                    **metadata,
                    "chunk_len": len(document),
                    "chunk_type": "chunk_page_text",
                },
            }
        )

    return chunks


def transform(use_cache: bool = True) -> list[dict[str, Any]]:
    if use_cache and os.path.exists(OUTPUT_CHUNKS):
        print(f"[transform-eesr] Chunks already exist in {OUTPUT_CHUNKS}, skipping")
        return pd.read_json(OUTPUT_CHUNKS, lines=True, encoding="utf-8").to_dict(orient="records")

    pages = get_pages()
    if pages.empty:
        print("[transform-eesr] No EESR pages found")
        return []

    # Skip annexes, resumes, and other non-content pages based on PAGE_COURANTE_ID
    print(f"[warn] Skipping pages PAGE_COURANTE_ID <= 1 (annexes, resumes, etc.)")
    pages = pages[pages["PAGE_COURANTE_ID"] > 1]

    chunks: list[dict[str, Any]] = []
    for _, row in pages.iterrows():
        page = row.to_dict()
        chunks.extend(page_to_chunks(page))

    print(f"[transform-eesr] Generated {len(chunks)} EESR chunks")
    save_jsonl(chunks, OUTPUT_CHUNKS)
    return chunks


def transform_cli():
    parser = argparse.ArgumentParser(description="Transform EESR pages into chunked documents")
    parser.add_argument("--no-cache", action="store_true", help="Force reload of chunks")
    args = parser.parse_args()
    transform(use_cache=not args.no_cache)


if __name__ == "__main__":
    transform_cli()
