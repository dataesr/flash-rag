from typing import Literal
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.query import query as run_query
from src.update import update as run_update, UpdateRequest

app = FastAPI(title="Flash Notes RAG API")


class QueryRequest(BaseModel):
    query: str
    source: Literal["all", "eesr", "ssmesr"] = "all"
    top_k: int = 5
    use_reranker: bool = True
    use_hybrid_search: bool = True


@app.post("/query")
async def query(payload: QueryRequest):
    print(f"[query] Received query request: {payload}")
    answer, sources = run_query(
        payload.query, 
        payload.source, 
        payload.top_k,
        payload.use_reranker,
        payload.use_hybrid_search
    )
    return {"query": query, "answer": answer, "sources": sources}


@app.post("/update")
async def update(payload: UpdateRequest, background_tasks: BackgroundTasks):
    print(f"[update] Received update request: {payload}")
    background_tasks.add_task(run_update, payload)
    return {"status": "accepted"}


app.mount("/", StaticFiles(directory="./static", html=True), name="static")
