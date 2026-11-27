import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.ask import cited_sources
from src.documents import SUPPORTED_TYPES
from src.generator import GENERATOR_MODEL, LLM_PROVIDER, generate_answer
from src.pipeline_run import build_library, retrieve

DOCS_DIR = "eval_docs"
library = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    files = [
        os.path.join(DOCS_DIR, name)
        for name in sorted(os.listdir(DOCS_DIR))
        if os.path.splitext(name)[1].lower() in SUPPORTED_TYPES
    ]
    store, hybrid, images = build_library(files)
    library["hybrid"] = hybrid
    library["files"] = files
    library["images"] = images
    library["passages"] = len(store.chunks)
    yield


app = FastAPI(title="Multimodal RAG", lifespan=lifespan)


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    ready = library.get("passages", 0) > 0
    body = {
        "status": "ok" if ready else "not ready",
        "llm": LLM_PROVIDER,
        "model": GENERATOR_MODEL,
        "documents": library.get("files", []),
        "passages": library.get("passages", 0),
        "images": len(library.get("images", [])),
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/ask")
def ask(body: Question):
    selected = retrieve(body.question, library["hybrid"], use_reranking=True)
    contexts = [r["chunk"] for r in selected]
    answer = generate_answer(body.question, contexts, source_tags=True)
    return {"answer": answer, "sources": cited_sources(answer, contexts)}