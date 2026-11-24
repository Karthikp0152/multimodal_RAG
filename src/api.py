import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.documents import SUPPORTED_TYPES
from src.pipeline_run import build_library

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": library.get("files", []),
        "passages": library.get("passages", 0),
        "images": len(library.get("images", [])),
    }