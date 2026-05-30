"""Knowledge-graph persistence and query-time entity resolution.

Two responsibilities:
  1. Write path — `persist_graph` upserts canonical entities, inserts
     aliases, mention rows, and relation rows after extraction+resolution.
     Called inside the ingest transaction.
  2. Read path — `resolve_entity_names` translates ?entity=X query-param
     names into entity UUIDs (as strings) for the retrieval filter.

DECISIONS.md §18.8: every relations row stores both source_chunk_id and
source_document_id so future graph-answer endpoints can cite the exact
chunk that grounded each edge via the existing search_result blocks.
"""

from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row

from app.models import EntityType, ExtractedGraph


def upsert_entity(
    conn: Connection,
    canonical_name: str,
    entity_type: EntityType,
    description: str,
) -> UUID:
    """Insert or return the entity row for (canonical_name, entity_type).

    ON CONFLICT keeps whichever description is longer. Length is a coarse
    quality proxy but a robust one: a chunk that mentions an entity in
    passing tends to produce a thin description ("A company"); a chunk that
    centers on the entity tends to produce a richer one. Locking in the
    longer description preserves the resolution prompt's disambiguation
    signal across ingests instead of clobbering rich descriptions with
    bare ones.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (canonical_name, entity_type)
            DO UPDATE SET description = CASE
                WHEN length(EXCLUDED.description) > length(entities.description)
                THEN EXCLUDED.description
                ELSE entities.description
            END
            RETURNING id
            """,
            (canonical_name, entity_type.value, description),
        )
        return cur.fetchone()[0]


def insert_alias(conn: Connection, entity_id: UUID, alias: str) -> None:
    """Insert an alias row; ignore duplicates (idempotent under retries)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_aliases (entity_id, alias)
            VALUES (%s, %s)
            ON CONFLICT (entity_id, alias) DO NOTHING
            """,
            (entity_id, alias),
        )


def insert_mention(conn: Connection, chunk_id: UUID, entity_id: UUID) -> None:
    """Insert a chunk_entity_mentions row; ignore duplicates."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunk_entity_mentions (chunk_id, entity_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (chunk_id, entity_id),
        )


def insert_relation(
    conn: Connection,
    *,
    source_entity_id: UUID,
    predicate: str,
    target_entity_id: UUID,
    source_chunk_id: UUID,
    source_document_id: UUID,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO relations
              (source_entity_id, predicate, target_entity_id,
               source_chunk_id, source_document_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (source_entity_id, predicate, target_entity_id,
             source_chunk_id, source_document_id),
        )


def persist_graph(
    conn: Connection,
    *,
    document_id: UUID,
    chunk_ids: list[UUID],
    graphs: list[ExtractedGraph],
    alias_map: dict[tuple[str, EntityType], str],
) -> None:
    """Persist all entity / alias / mention / relation rows for one document.

    Called inside the ingest transaction. `chunk_ids[i]` corresponds to
    `graphs[i]`. The `alias_map` from `graph_resolve.resolve_entities`
    maps every raw extracted name to its canonical name within its type.
    """
    entity_id_cache: dict[tuple[str, EntityType], UUID] = {}

    def get_or_create_entity(raw_name: str, etype: EntityType, desc: str) -> UUID:
        canonical = alias_map.get((raw_name, etype), raw_name)
        cache_key = (canonical, etype)
        if cache_key not in entity_id_cache:
            entity_id_cache[cache_key] = upsert_entity(conn, canonical, etype, desc)
            insert_alias(conn, entity_id_cache[cache_key], canonical)
        entity_id = entity_id_cache[cache_key]
        if raw_name != canonical:
            insert_alias(conn, entity_id, raw_name)
        return entity_id

    for chunk_id, graph in zip(chunk_ids, graphs):
        chunk_entity_ids: dict[str, UUID] = {}
        for entity in graph.entities:
            eid = get_or_create_entity(entity.name, entity.type, entity.description)
            chunk_entity_ids[entity.name] = eid
            insert_mention(conn, chunk_id, eid)

        for rel in graph.relations:
            src_id = chunk_entity_ids.get(rel.source)
            tgt_id = chunk_entity_ids.get(rel.target)
            if src_id is None or tgt_id is None:
                # The cookbook prompt instructs the model to only emit
                # relations connecting extracted entities, but defensive
                # skip handles any drift.
                continue
            insert_relation(
                conn,
                source_entity_id=src_id,
                predicate=rel.predicate,
                target_entity_id=tgt_id,
                source_chunk_id=chunk_id,
                source_document_id=document_id,
            )


def resolve_entity_names(conn: Connection, names: list[str]) -> list[str]:
    """Resolve a list of raw entity names to entity UUIDs (str form).

    Case-insensitive alias lookup. Names that don't match any alias are
    silently dropped from the result. This silent-drop shape is right for
    /chat (DECISIONS.md §18.9 — graceful degradation falls back to plain
    hybrid retrieval) but wrong for /search (which contracts that an
    unknown name yields no chunks); /search wraps this with
    `parse_entity_filter` in routes/search.py, which appends a sentinel
    UUID on partial resolution so the AND filter is unsatisfiable rather
    than silently relaxed.
    """
    if not names:
        return []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT DISTINCT ea.entity_id::text
            FROM entity_aliases ea
            WHERE LOWER(ea.alias) = ANY(
                SELECT LOWER(n) FROM unnest(%s::text[]) AS n
            )
            """,
            (names,),
        )
        return [row["entity_id"] for row in cur.fetchall()]
