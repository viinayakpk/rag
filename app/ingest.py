"""Shared persistence + embedding glue for /text and /document.

Both ingestion routes share the same SHA-256 dedupe pre-check, the same Voyage
error mapping (per DECISIONS.md §11), and the same atomic document + chunks
insert with a UniqueViolation race fallback (per DECISIONS.md §12). This
module owns those pieces so the routes can stay focused on input validation
and content shape (text vs PDF).

DECISIONS.md §18.4: graph extraction + resolution happens in the route layer
(before the transaction opens) so an extraction failure never leaves a
content_hash row behind that would short-circuit the next retry. Inside the
transaction we persist chunks with returned UUIDs and then write the graph
rows in one atomic step.
"""

from datetime import date
from uuid import UUID

import psycopg
from psycopg import Connection

from app.embeddings import embed_chunks, map_voyage_errors
from app.graph import persist_graph
from app.models import Chunk, EntityType, ExtractedGraph, IngestResponse


def embed_with_error_mapping(chunks: list[Chunk]) -> list[list[float]]:
    """Embed chunks via Voyage, translating SDK errors to HTTPException per §11."""
    with map_voyage_errors():
        return embed_chunks(chunks)


def find_existing_by_hash(conn: Connection, content_hash: str) -> IngestResponse | None:
    """Return the existing document for `content_hash`, or None if no row matches.

    Used both for the cheap pre-check (skip Voyage entirely) and for the
    UniqueViolation race-fallback inside `insert_document_with_chunks`.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, (SELECT count(*) FROM chunks WHERE document_id = documents.id)
            FROM documents WHERE content_hash = %s
            """,
            (content_hash,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    existing_id, n_chunks = row
    return IngestResponse(document_id=existing_id, n_chunks=n_chunks)


def insert_document_with_chunks(
    conn: Connection,
    *,
    title: str,
    author: str | None,
    published_date: date | None,
    metadata: str,
    text: str,
    content_hash: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    graphs: list[ExtractedGraph],
    alias_map: dict[tuple[str, EntityType], str],
) -> IngestResponse:
    """Persist document, chunks, and graph rows in one transaction.

    Chunks INSERT is a per-row loop (not executemany) so the generated chunk
    UUIDs can be captured in order — they're the foreign keys for the graph's
    chunk_entity_mentions and relations rows, all written in the same
    transaction via `persist_graph`.

    On UniqueViolation (a concurrent request inserted the same content_hash
    between our caller's pre-check and this insert), re-query and return the
    existing document_id — idempotent per §12. `metadata` is the JSON-encoded
    string the route will store as `jsonb`; the caller is responsible for
    JSON-encoding before passing it in.
    """
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (title, author, published_date, metadata, raw_text, content_hash)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """,
                (title, author, published_date, metadata, text, content_hash),
            )
            document_id: UUID = cur.fetchone()[0]

            chunk_ids: list[UUID] = []
            for c, e in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, ordinal, text, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (document_id, c.ordinal, c.text, c.token_count, e),
                )
                chunk_ids.append(cur.fetchone()[0])

            persist_graph(
                conn,
                document_id=document_id,
                chunk_ids=chunk_ids,
                graphs=graphs,
                alias_map=alias_map,
            )
    except psycopg.errors.UniqueViolation:
        existing = find_existing_by_hash(conn, content_hash)
        if existing is None:
            # Shouldn't happen — the only UNIQUE on this transaction is
            # documents.content_hash, so a violation implies a hash row exists.
            # Re-raise rather than silently dropping the request.
            raise
        return existing

    return IngestResponse(document_id=document_id, n_chunks=len(chunks))
