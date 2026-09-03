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


def parse_illustration(illustration: dict) -> tuple[str, str, str]:
    """
    Parse EESR illustration

    Returns: (markdown, csv, headers_text)
    """

    columns = illustration.get("ILLUSTRATION_TABLEAU_COLONNES_FR", "").split("|")
    rows = illustration.get("ILLUSTRATION_TABLEAU_LIGNES_FR", "").split("§")
    values = illustration.get("ILLUSTRATION_TABLEAU_VALEURS_FR", "").split("§")

    if not columns or not rows or not values:
        return "", "", ""

    # Clean empty values
    columns = [c.strip() for c in columns if c.strip()]
    rows = [r.strip() for r in rows if r.strip()]

    # Build markdown with row labels as first column
    md = "| " + " | ".join(columns) + " |\n"
    md += "|" + "|".join(["---"] * len(columns)) + "|\n"

    # Build CSV
    csv_lines = [",".join(columns)]

    for row_idx, row_label in enumerate(rows):
        if row_idx < len(values):
            row_values = values[row_idx].split("|")
            row_values = [v.strip() for v in row_values if v.strip()]
            md += "| " + " | ".join([row_label] + row_values) + " |\n"
            csv_lines.append(",".join([row_label] + row_values))

    csv_table = "\n".join(csv_lines)
    headers_text = " | ".join(columns)

    return md, csv_table, headers_text


def build_page_metadata(page: dict[str, Any]) -> dict[str, Any]:
    publication = page.get("PUBLICATION") or {}
    publication_date = publication.get("PUBLICATION_DATE_TRI") or publication.get("PUBLICATION_DATE_ANNEE") or ""
    publication_date = publication_date + "-01" if publication_date and len(publication_date) == 7 else publication_date
    publication_epoch = to_unix_epoch(publication_date) if publication_date else 0

    page_id = page["PAGE_NOM_DE_CODE"].lower().replace("eesr", "")
    page_keywords = [page.get("PAGE_CHAPITRE_FR"), page.get("PAGE_CHAPITRE_EN")]

    return {
        "title": page.get("PAGE_TITRE_FR") or page.get("PAGE_TITRE_EN", ""),
        "source": "eesr",
        "publication_date": publication_date,
        "publication_epoch": publication_epoch,
        "publication_type": page.get("PAGE_TYPE_NOM", page.get("PAGE_TYPE_ID", "page")),
        "keywords": ", ".join([k for k in page_keywords if k]),
        "file_id": page_id,
        "file_name": page["PAGE_FILE_NAME"],
        "file_format": "json",
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
        else:
            if field == "PAGE_TEXTE_FR":
                print(f"[warn] No PAGE_TEXTE_FR found for page {page['PAGE_NOM_DE_CODE']} ({page.get('PAGE_TITRE_FR')})")
                return ""  # Skip pages without main text

    if not parts:
        print(f"[error] No content found for page {page['PAGE_NOM_DE_CODE']}")
        return ""

    return "\n\n".join(parts)


def page_to_chunks(page: dict[str, Any]) -> list[dict[str, Any]]:

    metadata = build_page_metadata(page)
    text = build_page_text(page)
    page_id = metadata["page_id"]

    chunks: list[dict[str, Any]] = []

    # ========== PARAGRAPHS ==========
    if text and isinstance(text, str):
        text_chunks: list[str] = []
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            text_chunks = [text.strip()] if text.strip() else []

        current_chunk = ""
        for paragraph in paragraphs:
            if not current_chunk:
                current_chunk = paragraph
                continue

            candidate = current_chunk + "\n\n" + paragraph
            if len(candidate) <= CHUNK_MAX_CHARS:
                current_chunk = candidate
            else:
                text_chunks.append(current_chunk)
                current_chunk = paragraph

        if current_chunk:
            text_chunks.append(current_chunk)

        for chunk_idx, document in enumerate(text_chunks):
            chunks.append(
                {
                    "id": f"eesr_{page_id}_p{chunk_idx}",
                    "document": document,
                    "metadata": {
                        **metadata,
                        "chunk_len": len(document),
                        "chunk_type": "paragraph",
                    },
                }
            )

    # ========== TABLES ==========
    illustrations = page.get("ILLUSTRATIONS", [])
    if not illustrations:
        return chunks

    for illust_index, illust in enumerate(illustrations):
        # Only process tableaux
        if illust.get("ILLUSTRATION_TYPE") != "Tableau":
            continue

        title = illust.get("ILLUSTRATION_TITRE_FR", "")
        sous_type = illust.get("ILLUSTRATION_SOUS_TYPE", "")

        # Parse table
        markdown_table, csv_table, headers_text = parse_illustration(illust)

        if not markdown_table:
            continue

        # Create document with context for BM25
        doc_parts = [f"{sous_type}: {title}", f"Colonnes: {headers_text}", "", markdown_table]
        document = "\n".join(doc_parts)

        chunks.append(
            {
                "id": f"eesr_{page_id}_t{illust_index}",
                "document": document,
                "metadata": {
                    **metadata,
                    "chunk_type": "table",
                    "chunk_len": len(document),
                    "table_index": illust_index,
                    "table_headers": headers_text[:500],
                    "table_csv": csv_table,
                    "table_title": title,
                    "table_type": sous_type,
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
    print(f"  - Paragraphs: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'paragraph')}")
    print(f"  - Tables: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'table')}")

    save_jsonl(chunks, OUTPUT_CHUNKS)
    return chunks


def transform_cli():
    parser = argparse.ArgumentParser(description="Transform EESR pages into chunked documents")
    parser.add_argument("--no-cache", action="store_true", help="Force reload of chunks")
    args = parser.parse_args()
    transform(use_cache=not args.no_cache)


if __name__ == "__main__":
    transform_cli()
