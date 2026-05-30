import json
from hashlib import sha256
from typing import Annotated

import pymupdf
from fastapi import APIRouter, Form, HTTPException

from app.chunking import chunk_text
from app.db import ConnDep
from app.extraction import TextTooLargeError, extract_text
from app.graph_extract import extract_graphs_from_chunks
from app.graph_resolve import resolve_entities
from app.ingest import (
    embed_with_error_mapping,
    find_existing_by_hash,
    insert_document_with_chunks,
)
from app.models import IngestDocumentRequest, IngestResponse

router = APIRouter()


@router.post("/document", response_model=IngestResponse)
def ingest_document(
    body: Annotated[IngestDocumentRequest, Form()],
    conn: ConnDep,
) -> IngestResponse:
    pdf_bytes = body.file.file.read()

    # Magic-byte check (DECISIONS.md §17). The Content-Type header is
    # client-controlled and untrustworthy; the %PDF- prefix is a property
    # of the file itself.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="file is not a PDF (missing %PDF- header)")

    try:
        text = extract_text(pdf_bytes)
    except pymupdf.FileDataError as exc:
        raise HTTPException(status_code=400, detail=f"malformed PDF: {exc}") from exc
    except TextTooLargeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # PyMuPDF can emit \x00 for unmapped glyphs and certain malformed text
    # streams. Postgres TEXT/JSONB reject NULs, which would otherwise surface
    # here as an opaque 500 from psycopg.
    text = text.replace("\x00", "").strip()
    if not text:
        # Image-only / scanned PDFs without an OCR'd text layer hit this.
        # Per §17 this is a 400 (mirrors /text's empty-after-strip), not 422.
        raise HTTPException(
            status_code=400,
            detail="PDF contains no extractable text (possibly an image-only or scanned document; OCR is not supported)",
        )

    # Parse the metadata JSON string (Q1: multipart sends metadata as JSON).
    try:
        metadata = json.loads(body.metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"metadata field is not valid JSON: {exc}",
        ) from exc
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")

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
        title=body.title,
        author=body.author,
        published_date=body.published_date,
        metadata=json.dumps(metadata),
        text=text,
        content_hash=content_hash,
        chunks=chunks,
        embeddings=embeddings,
        graphs=graphs,
        alias_map=alias_map,
    )
