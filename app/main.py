from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.middleware import MaxBodySizeMiddleware
from app.routes.chat import router as chat_router
from app.routes.document import router as document_router
from app.routes.search import router as search_router
from app.routes.text import router as text_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db.open_pool()
    try:
        yield
    finally:
        db.close_pool()


app = FastAPI(title="RAG Knowledge Base", version="0.1.0", lifespan=lifespan)
# Body-size cap is registered last so it sits outermost — ASGI wraps in
# reverse order, and we want too-big requests rejected before any other
# middleware runs.
app.add_middleware(MaxBodySizeMiddleware)
app.include_router(text_router)
app.include_router(document_router)
app.include_router(search_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
