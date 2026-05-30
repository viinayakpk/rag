from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

import voyageai
from fastapi import HTTPException
from tokenizers import Encoding

from app.models import Chunk
from app.settings import settings

# Voyage-4 per-request caps (docs.voyageai.com/docs/embeddings).
MAX_INPUTS_PER_REQUEST = 1000
MAX_TOKENS_PER_REQUEST = 320_000


@cache
def get_client() -> voyageai.Client:
    return voyageai.Client(api_key=settings.voyage_api_key or None)


@contextmanager
def map_voyage_errors() -> Iterator[None]:
    """Translate Voyage SDK errors to HTTPException per DECISIONS.md §11.

    AuthenticationError -> 500 (server misconfig — missing/invalid VOYAGE_API_KEY;
        surfaced as 500 rather than a 5xx-transient code so operators don't
        mistake it for an upstream blip and wait it out).
    RateLimitError -> 429.
    InvalidRequestError -> 400 (payload Voyage refused).
    VoyageError catch-all -> 503 (timeout / ServiceUnavailable / generic upstream;
        transient, the client may retry).
    """
    try:
        yield
    except voyageai.error.AuthenticationError as exc:
        raise HTTPException(status_code=500, detail=f"embedding auth failure: {exc}") from exc
    except voyageai.error.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"embedding rate limited: {exc}") from exc
    except voyageai.error.InvalidRequestError as exc:
        raise HTTPException(status_code=400, detail=f"embedding rejected input: {exc}") from exc
    except voyageai.error.VoyageError as exc:
        raise HTTPException(status_code=503, detail=f"embedding upstream failure: {exc}") from exc


def embed_chunks(chunks: list[Chunk], input_type: str = "document") -> list[list[float]]:
    """Embed chunks for one document.

    Sub-batches internally to stay under voyage-4's 1000-input / 320K-token
    per-request caps. Returns embeddings in input order. Voyage errors are
    not caught here; they propagate to the route as 503.
    """
    if not chunks:
        return []

    embeddings: list[list[float]] = []
    batch_texts: list[str] = []
    batch_tokens = 0

    def flush() -> None:
        nonlocal batch_texts, batch_tokens
        if not batch_texts:
            return
        result = get_client().embed(batch_texts, model=settings.embedding_model, input_type=input_type)
        embeddings.extend(result.embeddings)
        batch_texts = []
        batch_tokens = 0

    for chunk in chunks:
        # Flush before adding if this chunk would exceed either cap.
        would_exceed_inputs = len(batch_texts) + 1 > MAX_INPUTS_PER_REQUEST
        would_exceed_tokens = batch_tokens + chunk.token_count > MAX_TOKENS_PER_REQUEST
        if batch_texts and (would_exceed_inputs or would_exceed_tokens):
            flush()
        batch_texts.append(chunk.text)
        batch_tokens += chunk.token_count

    flush()
    return embeddings


def embed_query(text: str, *, input_type: str = "query") -> list[float]:
    """Embed a single query string and return one 1024-dim vector.

    Voyage prepends a different internal prompt for `input_type="query"` vs
    `"document"`; pass query at retrieval time per DECISIONS.md §2. Errors are
    not caught here — wrap with `map_voyage_errors()` (or use
    `embed_query_with_error_mapping`).
    """
    result = get_client().embed([text], model=settings.embedding_model, input_type=input_type)
    return result.embeddings[0]


def embed_query_with_error_mapping(text: str) -> list[float]:
    """Embed a query string, translating Voyage SDK errors to HTTPException."""
    with map_voyage_errors():
        return embed_query(text)


def count_tokens(texts: list[str]) -> int:
    return get_client().count_tokens(texts, model=settings.embedding_model)


def per_text_token_counts(texts: list[str]) -> list[int]:
    if not texts:
        return []
    return [len(e.tokens) for e in tokenize(texts)]


def tokenize(texts: list[str]) -> list[Encoding]:
    """Return Voyage's HuggingFace `Encoding` objects for each text.

    Each encoding exposes `.tokens`, `.ids`, and `.offsets` (a list of
    (char_start, char_end) tuples per token).
    """
    return get_client().tokenize(texts, model=settings.embedding_model)
