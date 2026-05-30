"""GET /search — hybrid retrieval + rerank (DECISIONS.md §7 / §7.2)."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.db import ConnDep
from app.graph import resolve_entity_names
from app.models import SearchResponse
from app.rerank import rerank
from app.retrieval import RRF_CANDIDATES_PER_LANE, Filters, retrieve

router = APIRouter()


class SearchParams(BaseModel):
    """Bound from query string by FastAPI's Annotated[Model, Query()]."""

    # max_length=4096 short-circuits absurd queries before they reach Voyage
    # (the 25 MB body cap doesn't apply to GET URLs, and a meaningful
    # retrieval query is rarely longer than a paragraph).
    q: str = Field(min_length=1, max_length=4096, description="Search query")
    k: int = Field(default=10, ge=1, le=50)
    author: str | None = None
    published_after: date | None = None
    published_before: date | None = None

    @field_validator("q")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


def parse_meta_filters(request: Request) -> dict[str, str]:
    """Pull `meta.<key>=<value>` query params into a flat dict (single-value-per-key).

    Multiple values for the same key collapse to the last; documented as a
    single-value-per-key API. Numeric/boolean coercion is intentionally not
    attempted — query params are strings, JSONB values may not be.
    """
    return {k[5:]: v for k, v in request.query_params.items() if k.startswith("meta.")}


# A UUID guaranteed not to match any chunk_entity_mentions row. Appended
# to the resolved list when a user-passed entity name fails to resolve, so
# the EXISTS + HAVING COUNT(DISTINCT) = n_entities gate in retrieval.py
# yields empty results — matching the DECISIONS.md §18.7 contract that
# "an unknown name yields no chunks." gen_random_uuid() never produces
# all-zeros, so this is collision-free.
_UNRESOLVABLE_ENTITY_SENTINEL = "00000000-0000-0000-0000-000000000000"


def parse_entity_filter(
    conn: ConnDep,
    entity: Annotated[
        list[str],
        Query(description="Entity names — repeat for multiple (chunk-level AND)."),
    ] = [],
) -> list[list[str]]:
    """Resolve repeated ?entity=<name> query params into entity_id groups.

    Returns one group per user-passed name; each group's inner list holds
    the entity UUIDs that name resolved to. The list is > 1 when the same
    canonical name was extracted as multiple types (e.g. `Maria Garcia` as
    PERSON, ORGANIZATION, and EVENT — DECISIONS.md §18.11). The retrieval
    SQL counts distinct *groups* per chunk, so a chunk satisfies a group
    by mentioning any one of its entity_ids; AND across groups gives the
    user-intended chunk-level co-mention.

    A name with no matching alias makes the AND filter unsatisfiable. The
    function returns a group containing only `_UNRESOLVABLE_ENTITY_SENTINEL`
    so the EXISTS+HAVING gate yields empty results — matching the §18.7
    contract that "an unknown name yields no chunks."

    /chat takes a different path (DECISIONS.md §18.9): unresolved names
    silently drop (no sentinel) and the route falls back to plain hybrid
    retrieval rather than emptying. /chat builds its own groups inline.
    """
    if not entity:
        return []

    groups: list[list[str]] = []
    for name in entity:
        resolved = resolve_entity_names(conn, [name])
        if not resolved:
            resolved = [_UNRESOLVABLE_ENTITY_SENTINEL]
        groups.append(resolved)
    return groups


@router.get("/search", response_model=SearchResponse)
def search(
    conn: ConnDep,
    params: Annotated[SearchParams, Query()],
    metadata: Annotated[dict[str, str], Depends(parse_meta_filters)],
    entity_id_groups: Annotated[list[list[str]], Depends(parse_entity_filter)],
) -> SearchResponse:
    filters = Filters(
        author=params.author,
        published_after=params.published_after,
        published_before=params.published_before,
        metadata=metadata,
        entity_id_groups=entity_id_groups,
    )
    candidates = retrieve(conn, params.q, k=RRF_CANDIDATES_PER_LANE, filters=filters)
    results = rerank(params.q, candidates, top_k=params.k)
    return SearchResponse(results=results)
