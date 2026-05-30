"""Hybrid retrieval (DECISIONS.md §7) — dense pgvector + lexical tsvector,
fused with Reciprocal Rank Fusion.

One function — `retrieve(conn, query, k, filters) -> list[RetrievedChunk]`.
Endpoints don't write SQL; they call this. The route layer then applies a
reranker (DECISIONS.md §7.2) over the candidate pool returned here.

Shape mirrors pgvector's reference RRF example
(github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/rrf.py)
and Supabase's `hybrid_search` RPC (supabase.com/docs/guides/ai/hybrid-search):
two CTEs (one per lane) joined with FULL OUTER JOIN — a LEFT JOIN would
silently drop lex-only hits (the very queries the lexical lane exists to
catch: rare proper nouns, IDs, technical jargon).
"""

import json
from datetime import date

from psycopg import Connection
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from app.embeddings import embed_query_with_error_mapping
from app.models import RetrievedChunk

# RRF constant from Cormack et al. (2009) — the canonical default. pgvector
# and Supabase both use 60. Smaller k weights top-rank agreement harder;
# larger k flattens the contribution curve.
RRF_K = 60

# Per-lane candidate pool. Each lane returns its top N before fusion;
# matches pgvector's example. The outer LIMIT (passed by callers as `k`)
# trims the fused pool to what the rerank stage will actually score.
RRF_CANDIDATES_PER_LANE = 50


class Filters(BaseModel):
    """Structured filter input to `retrieve`. Built by route layer from query params."""

    author: str | None = None
    published_after: date | None = None
    published_before: date | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    # DECISIONS.md §18.7 / §18.11 — resolved entity UUIDs grouped by the
    # user-passed name they came from. Outer list is one group per
    # user-name; inner list is the entity_ids that name resolved to. The
    # inner list can be > 1 element when the same canonical name was
    # extracted as multiple entity types (`Maria Garcia` as PERSON vs
    # ORGANIZATION vs EVENT — each is a separate `entities` row). The SQL
    # filter counts distinct *groups* per chunk, so a chunk mentioning any
    # one member of each group satisfies that group, with AND across groups.
    # Empty outer list = no entity filter applied.
    entity_id_groups: list[list[str]] = Field(default_factory=list)


# Filters are applied inside both lanes (not just the outer SELECT) so the
# candidate pool per lane is already post-filter — a selective filter
# otherwise leaves each lane's top-50 mostly empty after the outer WHERE.
#
# `SET LOCAL hnsw.iterative_scan = 'strict_order'` from §6.1 stays for the
# vec lane: when the planner picks an HNSW Index Scan, iterative_scan keeps
# walking past the initial batch until enough rows pass the WHERE to fill
# LIMIT. At small-corpus scale the planner chooses Seq Scan and the GUC is
# inert — flagged in §6.1 as a deliberate validation gap.
#
# Entity filter shape (§18.7 / §18.11): UNNEST builds (entity_id, group_id)
# rows from two parallel arrays, JOINs to chunk_entity_mentions, and counts
# distinct groups per chunk. A chunk satisfies the filter when every group
# is represented by at least one mention — equivalent to "AND across user-
# passed names, OR within each name's resolved entity_ids" (which is the
# documented intent for type-split entities like the same canonical name
# classified as PERSON in one chunk and ORGANIZATION in another).
_SQL = f"""
WITH vec AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(query_vec)s::vector) AS rank
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE  (%(author)s::text IS NULL OR d.author = %(author)s::text)
      AND  (%(after)s::date  IS NULL OR d.published_date >= %(after)s::date)
      AND  (%(before)s::date IS NULL OR d.published_date <= %(before)s::date)
      AND  (%(meta)s::jsonb  IS NULL OR d.metadata @> %(meta)s::jsonb)
      AND  (%(n_groups)s = 0 OR EXISTS (
              SELECT 1 FROM chunk_entity_mentions cem
              JOIN UNNEST(
                       %(entity_ids)s::uuid[],
                       %(entity_groups)s::int[]
                   ) AS x(eid, grp) ON cem.entity_id = x.eid
              WHERE cem.chunk_id = c.id
              GROUP BY cem.chunk_id
              HAVING COUNT(DISTINCT x.grp) = %(n_groups)s
           ))
    ORDER BY c.embedding <=> %(query_vec)s::vector
    LIMIT {RRF_CANDIDATES_PER_LANE}
),
lex AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS rank
    FROM chunks c
    JOIN documents d ON d.id = c.document_id,
         plainto_tsquery('simple', %(query_text)s) q
    WHERE c.tsv @@ q
      AND  (%(author)s::text IS NULL OR d.author = %(author)s::text)
      AND  (%(after)s::date  IS NULL OR d.published_date >= %(after)s::date)
      AND  (%(before)s::date IS NULL OR d.published_date <= %(before)s::date)
      AND  (%(meta)s::jsonb  IS NULL OR d.metadata @> %(meta)s::jsonb)
      AND  (%(n_groups)s = 0 OR EXISTS (
              SELECT 1 FROM chunk_entity_mentions cem
              JOIN UNNEST(
                       %(entity_ids)s::uuid[],
                       %(entity_groups)s::int[]
                   ) AS x(eid, grp) ON cem.entity_id = x.eid
              WHERE cem.chunk_id = c.id
              GROUP BY cem.chunk_id
              HAVING COUNT(DISTINCT x.grp) = %(n_groups)s
           ))
    ORDER BY ts_rank_cd(c.tsv, q) DESC
    LIMIT {RRF_CANDIDATES_PER_LANE}
),
fused AS (
    SELECT
        COALESCE(vec.id, lex.id) AS id,
        COALESCE(1.0 / ({RRF_K} + vec.rank), 0)
            + COALESCE(1.0 / ({RRF_K} + lex.rank), 0) AS score
    FROM vec
    FULL OUTER JOIN lex ON vec.id = lex.id
)
SELECT
    c.id              AS chunk_id,
    c.ordinal         AS ordinal,
    c.text            AS text,
    d.id              AS document_id,
    d.title           AS document_title,
    d.author          AS author,
    d.published_date  AS published_date,
    d.metadata        AS metadata,
    fused.score       AS score
FROM fused
JOIN chunks c    ON c.id = fused.id
JOIN documents d ON d.id = c.document_id
ORDER BY fused.score DESC
LIMIT %(k)s
"""


def retrieve(
    conn: Connection,
    query: str,
    k: int,
    filters: Filters,
) -> list[RetrievedChunk]:
    """Embed `query`, run hybrid RRF retrieval, return the top-k candidate chunks.

    `score` on the returned chunks is the RRF fusion score (sum of two
    1/(k+rank) terms), bounded above by 2/(RRF_K+1) ≈ 0.0328 and not
    comparable to cosine. The route layer reranks this candidate pool with
    Voyage rerank-2.5 before returning to the caller / handing to Claude.
    """
    query_vec = embed_query_with_error_mapping(query)
    meta_json = json.dumps(filters.metadata) if filters.metadata else None

    # DECISIONS.md §18.7 / §18.11: chunk-level AND co-mention with OR-within-group
    # for type-split canonicals. Flatten the [(group_id, entity_ids), …] shape
    # into two parallel arrays — UNNEST(arr1, arr2) zips them in SQL.
    flat_entity_ids: list[str] = []
    flat_entity_groups: list[int] = []
    for group_id, group in enumerate(filters.entity_id_groups):
        for eid in group:
            flat_entity_ids.append(eid)
            flat_entity_groups.append(group_id)

    # n_groups=0 short-circuits the EXISTS clause so behaviour with no entity
    # filter is identical to the pre-graph version.
    params = {
        "query_vec": query_vec,
        "query_text": query,
        "author": filters.author,
        "after": filters.published_after,
        "before": filters.published_before,
        "meta": meta_json,
        "entity_ids": flat_entity_ids,
        "entity_groups": flat_entity_groups,
        "n_groups": len(filters.entity_id_groups),
        "k": k,
    }

    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        cur.execute(_SQL, params)
        rows = cur.fetchall()

    return [RetrievedChunk(**row) for row in rows]
