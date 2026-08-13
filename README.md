# Flash Notes RAG

Natural-language Q&A over a collection of corporate "flash notes" (PDF, XLSX, spreadsheets). Implements a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB for vector storage, a language model backend for generation, and FastAPI for the HTTP API.

**Goals**
- Provide fast, accurate answers over a document collection
- Reproducible ingestion pipeline (fetch → extract → parse → embed → index)
- Simple HTTP API and lightweight frontend for demos

## Repository layout

```text
flash-notes-rag/
├── data/               # Raw and processed document JSON/line-delimited files
├── db/                 # ChromaDB persisted storage (sqlite + collection dir)
├── src/                # Python source (ingest, populate, query, utils)
├── notebooks/          # Exploration and repro notebooks
├── static/             # Tiny demo frontend (HTML/CSS)
├── main.py             # FastAPI app entrypoint
├── pyproject.toml      # Dependencies and metadata
├── makefile            # Build/release helpers
└── dockerfile          # Container image definition
```

## Quickstart

Prerequisites: Python 3.10+, `uv` (optional helper used here), and the environment variables required by your model backend and ChromaDB configuration.

1. Install dependencies and sync the workspace (with `uv`):

```bash
uv sync
```

2. Create a local `.env` file with required variables (example):

```
# .env
# MODEL_ENDPOINT=...        # model API url or inference server
# MODEL_API_KEY=...         # model API key if required
# CHROMA_DIR=./db          # chroma persistence path
# OTHER_ENV=...
```

3. Populate the ChromaDB index (ingest & embed documents):

```bash
# option A: using uv
uv run --env-file .env python -m src.populate

# option B: direct Python
python -m src.populate
```

The `populate` module supports flags to control each stage (fetch, extract, parse, embed). Run `python -m src.populate --help` for details.

4. Run the API server locally:

```bash
uv run --env-file .env fastapi dev main.py
# or
python -m main
```

The API is available at http://localhost:8000 by default.

## API

- POST `/query` — run a RAG query (retrieve + generate)
  - Body example: `{"question":"What does the report say about X?","top_k":5}`
- POST `/update` — run the ingestion/update orchestration (same options as `src.populate`)

Example `curl`:

```bash
curl -sS -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key findings about Y?","top_k":3}'
```

## Development notes

- Code lives under `src/`. Key modules:
  - `src/populate.py` — ingestion and indexing pipeline
  - `src/query.py` — retrieval + generation orchestration used by the API
  - `src/chromadb.py` — helper for collection creation and persistence
  - `src/mistral.py` — model integration (replaceable with any LLM client)

- Data and DB:
  - Raw/processed documents: `data/`
  - Chroma persistence: `db/`

- Notebooks: see `notebooks/` for interactive experiments and repros.

## Docker & CI

- Build locally via `make build` (image tagged from `pyproject.toml` version).
- Push with `make push` (pushes to configured registry).

## Release

Tag and release using the `make release VERSION=X.Y.Z` helper; it updates `pyproject.toml` and creates a git tag.

## Contributing

Feel free to open issues or PRs. If you change ingestion formats or the embedding model, include a short note in `README.md` describing required env vars and the expected data flow.

## License

See the repository `LICENSE` file for license information.
