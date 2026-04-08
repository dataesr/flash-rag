# Flash Notes RAG

Natural-language Q&A over a collection of company flash notes (PDF, XLSX). This project implements a RAG (Retrieval-Augmented Generation) pipeline using ChromaDB, Mistral AI, and FastAPI.

## Architecture

The project is structured as follows:

```text
flash-notes-rag/
├── data/               # Raw and processed document data
├── db/                 # ChromaDB vector database storage
├── src/                # Project source code
│   ├── load.py         # Handles fetching records from Zenodo
│   ├── extract.py      # Performs OCR and text extraction
│   ├── parse.py        # Parses documents into a unified format
│   ├── populate.py     # Embeds and indexes documents in ChromaDB
│   ├── query.py        # Main retrieval and generation logic
│   ├── chromadb.py     # ChromaDB client and collection utilities
│   ├── mistral.py      # Mistral AI model integration
│   └── utils.py        # Shared utility functions
├── static/             # Frontend assets (HTML, CSS, JS)
├── main.py             # FastAPI entry point
├── makefile            # Build and release orchestration
├── dockerfile          # Container configuration
└── pyproject.toml      # Project dependencies and metadata
```

## Getting Started

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management and project orchestration.

### Installation

```bash
# Sync dependencies and create a virtual environment
uv sync
```

### Running Locally

To start the FastAPI server:

```bash
uv run fastapi dev main.py
```

The application will be accessible at `http://localhost:8000`.

## Scripts and Pipelines

The system is designed as a pipeline that can be controlled via the API or individual scripts in `src/`.

### API Endpoints

- **`POST /query`**: Performs a RAG search.
  - Payload: `{"question": "What is the ESG score of Company X?", "top_k": 5}`
- **`POST /update`**: Orchestrates the entire ingestion pipeline.
  - Options: Skip fetch, skip download, force extract, force parse, reset/override collection.

### Direct Script Usage

You can also run individual parts of the pipeline:

```bash
uv run src/load.py      # Fetch new data
uv run src/extract.py   # Run OCR/extraction
uv run src/parse.py     # Clean and format data
uv run src/populate.py  # Index into ChromaDB
```

## Docker Workflow

Building and pushing Docker images is managed via the `makefile`.

### Build Image

```bash
make build
```
This builds the image tagged with the current version from `pyproject.toml` and `latest`.

### Push to Registry

```bash
make push
```
Pushes the images to `ghcr.io/dataesr/flash-rag`.

### Combined Build & Push

```bash
make build-push
```

## Release Process

The project follows a semantic versioning pattern. Releases are automated via the `makefile`.

1. **Tag a new version**:
   ```bash
   make release VERSION=X.Y.Z
   ```
   This updates `pyproject.toml`, commits the change, and creates a git tag.

2. **Push to main**:
   ```bash
   git push origin main --tags
   ```

Wait for the CI/CD pipeline to pick up the new tag and deploy the image.
