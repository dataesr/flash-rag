from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.query import query as run_query
from src.load import load as run_load
from src.extract import extract as run_extract
from src.parse import parse as run_parse
from src.populate import populate as run_populate

app = FastAPI(title="Flash Notes RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/query")
async def query(payload: QueryRequest):
    question = payload.question
    print(f"[query] Received query: {question}")
    answer, sources = run_query(question, payload.top_k)
    return {"question": question, "answer": answer, "sources": sources}


class UpdateRequest(BaseModel):
    load_skip_fetch: bool = True
    load_skip_download: bool = True
    load_force_download: bool = False
    extract_force: bool = False
    parse_force: bool = False
    populate_override: bool = False
    populate_reset: bool = False


@app.post("/update")
async def update(payload: UpdateRequest):
    print(f"[update] Received update request: {payload}")

    # load new documents
    print(f"\n{'='*60}")
    print("=== Loading documents ===")
    print(f"{'='*60}")
    run_load(
        skip_fetch=payload.load_skip_fetch,
        skip_download=payload.load_skip_download,
        force_download=payload.load_force_download,
    )

    # extract documents (OCR)
    print(f"\n{'='*60}")
    print("=== Extracting documents ===")
    print(f"{'='*60}")
    run_extract(force_extract=payload.extract_force)

    # parse documents
    print(f"\n{'='*60}")
    print("=== Parsing documents ===")
    print(f"{'='*60}")
    run_parse(force_parse=payload.parse_force)

    # populate collection
    print(f"\n{'='*60}")
    print("=== Populating collection ===")
    print(f"{'='*60}")
    run_populate(reset=payload.populate_reset, override=payload.populate_override)

    print(f"\n{'='*60}")
    print("=== Update Complete ===")
    print(f"{'='*60}\n")

    return {"status": "success"}


app.mount("/", StaticFiles(directory="./static", html=True), name="static")
