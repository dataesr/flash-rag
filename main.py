from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.query import query as run_query

app = FastAPI(title="Flash Notes RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/query")
async def query(payload: QueryRequest):
    question = payload.question
    print(f"[query] Received query: {question}")
    answer, sources = run_query(["--query", question, "--k", str(payload.top_k)])
    return {"question": question, "answer": answer, "sources": sources}


app.mount("/", StaticFiles(directory="./static", html=True), name="static")
