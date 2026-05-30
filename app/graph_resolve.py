"""Incremental entity resolution via Sonnet structured outputs.

Called once per document at ingest, after all chunks have been extracted.
Groups extracted entity mentions by type, fetches the existing canonical
set for that type from the DB (capped to keep the prompt bounded), then
issues one Sonnet call per type to cluster surface forms into canonical
entities.

DECISIONS.md §18.3: per-document-per-type resolution (≤4 Sonnet calls
per ingest) rather than per-chunk-per-type (would be ~120 calls). Cookbook's
incremental "resolve new entities against existing canonical set" pattern.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import anthropic
from fastapi import HTTPException
from psycopg import Connection
from psycopg.rows import dict_row

from app.llm import get_client
from app.models import Cluster, Entity, EntityType, ExtractedGraph, ResolvedClusters
from app.settings import settings

# Verbatim from Anthropic cookbook
# (platform.claude.com/cookbook/capabilities-knowledge-graph-guide).
RESOLVE_PROMPT_TMPL = (
    "Below are {entity_type} entities extracted from several documents. "
    "Some are different surface forms of the same real-world entity.\n\n"
    "<entities>\n{entity_list}\n</entities>\n\n"
    "Cluster them. Each input name must appear in exactly one cluster's "
    "aliases list. Entities that are genuinely distinct get their own "
    "single-element cluster. Use the descriptions to avoid merging entities "
    "that merely share a name. The canonical name should be the most "
    "complete, unambiguous form."
)

# Cap on existing canonicals fed to Sonnet per type to bound prompt growth
# as the corpus grows. At 200 names × ~20 chars each ≈ 4K characters — well
# within Sonnet's context. Future v2: embedding-based shortlisting against
# the new entities (DECISIONS.md §18.11 known limitations).
_MAX_EXISTING_CANONICALS = 200


@contextmanager
def map_graph_resolve_errors() -> Iterator[None]:
    """Translate Anthropic SDK errors — same taxonomy as map_anthropic_errors."""
    try:
        yield
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        raise HTTPException(status_code=500, detail=f"graph resolution auth failure: {exc}") from exc
    except anthropic.BadRequestError as exc:
        raise HTTPException(status_code=400, detail=f"graph resolution rejected input: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"graph resolution rate limited: {exc}") from exc
    except anthropic.APIError as exc:
        raise HTTPException(status_code=503, detail=f"graph resolution upstream failure: {exc}") from exc


def _fetch_existing_canonicals(
    conn: Connection, entity_type: EntityType
) -> dict[str, str]:
    """Return {canonical_name: description} for the most recent canonicals of this type."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT canonical_name, description
            FROM entities
            WHERE entity_type = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (entity_type.value, _MAX_EXISTING_CANONICALS),
        )
        return {row["canonical_name"]: row["description"] for row in cur.fetchall()}


def _resolve_type(
    conn: Connection,
    entity_type: EntityType,
    new_entities: list[Entity],
) -> list[Cluster]:
    """Resolve new entities of one type against the existing canonical set.

    Existing canonicals are fed in as anchors so previously resolved entities
    aren't re-split. With only one entry total, no Sonnet call is needed.
    """
    if not new_entities:
        return []

    existing = _fetch_existing_canonicals(conn, entity_type)

    all_entries: dict[str, str] = dict(existing)
    for e in new_entities:
        all_entries.setdefault(e.name, e.description)

    if len(all_entries) <= 1:
        name, desc = next(iter(all_entries.items()))
        return [Cluster(canonical=name, aliases=[name])]

    entity_list = "\n".join(
        f"- {name}: {desc}" for name, desc in all_entries.items()
    )
    prompt = RESOLVE_PROMPT_TMPL.format(
        entity_type=entity_type.value,
        entity_list=entity_list,
    )

    with map_graph_resolve_errors():
        response = get_client().beta.messages.parse(
            model=settings.resolution_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            output_format=ResolvedClusters,
        )

    clusters = (response.parsed_output or ResolvedClusters(clusters=[])).clusters
    return _anchor_existing_canonicals(clusters, set(existing))


def _anchor_existing_canonicals(
    clusters: list[Cluster], existing_canonicals: set[str]
) -> list[Cluster]:
    """Force any cluster's canonical back to a pre-existing DB canonical.

    The cookbook prompt invites Sonnet to pick "the most complete, unambiguous
    form" as the canonical — which is free to promote a new surface form over
    a name that's already canonical in the DB. If we let that through,
    persist_graph would write a second `entities` row for the same real-world
    entity (the UNIQUE constraint catches only identical canonical_name+type
    pairs). Whenever a cluster's aliases include a name that's already a
    canonical in the DB, force the cluster's canonical to that existing name.
    """
    if not existing_canonicals:
        return clusters
    for cluster in clusters:
        if cluster.canonical in existing_canonicals:
            continue
        overlap = existing_canonicals.intersection(cluster.aliases)
        if overlap:
            cluster.canonical = next(iter(overlap))
    return clusters


def resolve_entities(
    conn: Connection,
    graphs: list[ExtractedGraph],
) -> dict[tuple[str, EntityType], str]:
    """Resolve all entities across a document's chunks to canonical names.

    Returns {(raw_name, entity_type): canonical_name}. Issues at most one
    Sonnet call per EntityType that has at least one entity (≤4 total).
    """
    seen: dict[tuple[str, EntityType], str] = {}
    for graph in graphs:
        for entity in graph.entities:
            seen.setdefault((entity.name, entity.type), entity.description)

    by_type: dict[EntityType, list[Entity]] = {t: [] for t in EntityType}
    for (name, etype), desc in seen.items():
        by_type[etype].append(Entity(name=name, type=etype, description=desc))

    alias_map: dict[tuple[str, EntityType], str] = {}
    for etype, entities in by_type.items():
        for cluster in _resolve_type(conn, etype, entities):
            for alias in cluster.aliases:
                alias_map[(alias, etype)] = cluster.canonical

    return alias_map
