"""Voyage rerank-2.5 — second stage of the hybrid retrieval pipeline (DECISIONS.md §7.2).

`rerank(query, candidates, top_k)` takes the RRF candidate pool from
`app.retrieval.retrieve` and returns the top-k chunks reordered by Voyage's
`relevance_score`. Returned chunks are copies of the input candidates with
the score field replaced by the rerank `relevance_score`; the upstream RRF
score is not preserved (DECISIONS.md §8 specifies the rerank score as the
single score returned to clients).

Voyage SDK errors are translated via the same context manager used by the
embedder (`map_voyage_errors`) so operators see one taxonomy across both
upstream calls (DECISIONS.md §11).
"""

from app.embeddings import get_client, map_voyage_errors
from app.models import RetrievedChunk

# DECISIONS.md §7.2 — Voyage's recommended general-purpose reranker, 32K
# context, multilingual, instruction-following. Anthropic's recommended
# embedding partner ships this; matches the "Anthropic stack end-to-end"
# narrative from §2.
RERANK_MODEL = "rerank-2.5"


def rerank(
    query: str, candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Rerank `candidates` against `query`; return the top `top_k` reordered."""
    if not candidates:
        return []

    documents = [c.text for c in candidates]
    with map_voyage_errors():
        result = get_client().rerank(
            query=query,
            documents=documents,
            model=RERANK_MODEL,
            top_k=top_k,
        )

    reranked: list[RetrievedChunk] = []
    for r in result.results:
        chunk = candidates[r.index]
        reranked.append(
            RetrievedChunk.model_validate(
                {**chunk.model_dump(), "score": r.relevance_score}
            )
        )
    return reranked
