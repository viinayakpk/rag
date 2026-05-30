import json
from hashlib import sha256

from fastapi import APIRouter, HTTPException

from app.chunking import chunk_text
from app.db import ConnDep
from app.graph_extract import extract_graphs_from_chunks
from app.graph_resolve import resolve_entities
from app.ingest import (
    embed_with_error_mapping,
    find_existing_by_hash,
    insert_document_with_chunks,
)
from app.models import IngestResponse, IngestTextRequest

router = APIRouter()


@router.post("/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest, conn: ConnDep) -> IngestResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty after stripping whitespace")

    content_hash = sha256(text.encode("utf-8")).hexdigest()

    existing = find_existing_by_hash(conn, content_hash)
    if existing is not None:
        return existing

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="text produced no chunks")

    embeddings = embed_with_error_mapping(chunks)

    # DECISIONS.md §18.4 — extraction + resolution run before the transaction
    # opens. A failure here bails out before any DB rows are written, so a
    # retry won't be short-circuited by the content-hash dedupe path.
    graphs = extract_graphs_from_chunks(chunks)
    alias_map = resolve_entities(conn, graphs)

    return insert_document_with_chunks(
        conn,
        title=req.title,
        author=req.author,
        published_date=req.published_date,
        metadata=json.dumps(req.metadata),
        text=text,
        content_hash=content_hash,
        chunks=chunks,
        embeddings=embeddings,
        graphs=graphs,
        alias_map=alias_map,
    )
