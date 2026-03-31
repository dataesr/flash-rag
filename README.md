# Flash Notes RAG

Natural-language Q&A over a collection of company flash notes (JSON / plain text).

## Architecture

```
flash_rag/
├── data/
│   ├── sample_notes/     ← your JSON or .txt flash notes go here
│   ├── chroma_db/        ← auto-created: vector index for semantic search
│   └── structured.db     ← auto-created: SQLite for tables & KPIs
├── src/
│   ├── ingest.py         ← parse notes → embed → store
│   └── query.py          ← route query → retrieve → LLM answer
└── requirements.txt
```

Two query paths:

- **Semantic** — embeds your question, retrieves the most relevant note chunks,
  asks an LLM to synthesise an answer with citations.
- **Structured** — detects numeric / aggregation questions, generates SQL,
  runs it on SQLite, adds a brief LLM commentary.

## Setup

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# 3. Drop your flash notes into data/sample_notes/
#    Supported formats: .json  .txt  .md
#
#    JSON schema (all fields except "text" are optional):
#    {
#      "id":       "note_001",
#      "date":     "2024-03-15",
#      "title":    "Q1 2024 Revenue Flash",
#      "category": "financials",
#      "text":     "Free-text narrative ...",
#      "tables": [
#        {
#          "name":    "revenue_by_segment",
#          "columns": ["segment", "revenue_eur_m", "yoy_pct"],
#          "rows":    [["SaaS", 68.5, 18.1], ...]
#        }
#      ]
#    }

# 4. Ingest your notes (run once, re-run after adding new notes)
python src/ingest.py --notes_dir data/sample_notes
```

## Usage

```bash
# Single question
python src/query.py "What drove revenue growth in Q1 2024?"

# Aggregation / structured query
python src/query.py "Show average SaaS revenue across all quarters"

# Interactive REPL
python src/query.py
```

## Example queries

| Query                                            | Route      |
| ------------------------------------------------ | ---------- |
| "What concerns were flagged in the Q2 note?"     | semantic   |
| "What is the AI product roadmap for 2025?"       | semantic   |
| "Show me total budget across all AI initiatives" | structured |
| "Compare SaaS revenue Q1 vs Q2 2024"             | structured |
| "Average EBITDA margin across all notes"         | structured |
| "What drove margin improvement?"                 | semantic   |

## Extending

**Add new notes**: drop JSON/TXT files in `data/sample_notes/` and re-run `ingest.py`.
The ingester uses `upsert` so re-running is safe — existing notes are updated, not duplicated.

**Swap the LLM**: change `LLM_MODEL` in `query.py`. Works with any OpenAI-compatible API
(e.g. point `base_url` at a local Ollama instance for fully offline use).

**Improve table handling**: for complex tables, consider storing them with explicit
type casting (REAL / INTEGER) rather than TEXT. Edit `insert_tables()` in `ingest.py`.

**Add an MCP layer**: once this pipeline is working, you can wrap `answer()` in
`query.py` as an MCP tool, exposing it to Claude or other agents.
