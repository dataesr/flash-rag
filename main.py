from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.query import query as run_query
from src.update import run_update, UpdateRequest

app = FastAPI(title="Flash Notes RAG API")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/query")
async def query(payload: QueryRequest):
    query = payload.query
    print(f"[query] Received query: {query}")
    answer, sources = run_query(query, payload.top_k)
    return {"query": query, "answer": answer, "sources": sources}


@app.post("/update")
async def update(payload: UpdateRequest, background_tasks: BackgroundTasks):
    print(f"[update] Received update request: {payload}")
    background_tasks.add_task(run_update, payload)
    return {"status": "accepted"}


app.mount("/", StaticFiles(directory="./static", html=True), name="static")
