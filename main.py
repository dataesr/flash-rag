import logging
from typing import Literal
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.query import query as run_query
from src.update import update as run_update, UpdateRequest

logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=logging.DEBUG)
for logger_name in ["httpx", "httpcore", "asyncio", "watchfiles"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(title="Flash Notes RAG API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = False
    use_cross_encoder: bool = False
    use_hybrid_search: bool = False
    use_mistral: bool = False
    filters: dict[str, str | list[str]] = {}


@app.post("/query")
async def query(payload: QueryRequest):
    logger.info(f"[query] Received query request: {payload}")
    sources, answer, citations = run_query(
        payload.query,
        payload.top_k,
        payload.use_reranker,
        payload.use_cross_encoder,
        payload.use_hybrid_search,
        payload.use_mistral,
        payload.filters,
    )
    return {"query": payload.query, "sources": sources, "answer": answer, "citations": citations}


@app.post("/update")
async def update(payload: UpdateRequest, background_tasks: BackgroundTasks):
    logger.info(f"[update] Received update request: {payload}")
    background_tasks.add_task(run_update, payload)
    return {"status": "accepted"}


app.mount("/", StaticFiles(directory="./static", html=True), name="static")
