"""Anthropic chat capability — client, error mapping, prompt, generation.

Mirrors `app/embeddings.py` for Voyage. Routes import `generate_answer` and
the constants; they don't talk to the SDK directly. Per DECISIONS.md §8 we
use Anthropic's first-class Search Result content blocks with structured
citations rather than a prompt-based [N] numbered scheme.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

import anthropic
from fastapi import HTTPException

from app.models import RetrievedChunk
from app.settings import settings

# DECISIONS.md §9 — alias, not a dated snapshot. Auto-tracks the latest
# minor revision; swap to "claude-sonnet-4-6-YYYYMMDD" for production
# reproducibility if/when reviewer-grading concerns shift.
CHAT_MODEL = "claude-sonnet-4-6"
CHAT_MAX_TOKENS = 2048

# Single source of truth for the refusal sentence. Interpolated into
# SYSTEM_PROMPT below and imported by the route's threshold-gate path
# so both refusal channels emit identical text.
REFUSAL_TEXT = "I don't have enough information in the provided sources to answer this."

# DECISIONS.md §8 guardrails #1, #2 — only-sources and refusal allowed.
# Citation formatting is handled by the API (Search Result content blocks
# with citations.enabled=true), so the prompt doesn't teach [N] syntax.
SYSTEM_PROMPT = (
    "You answer questions using only the provided search results. If the "
    "search results do not contain enough information to answer the question, "
    f'say: "{REFUSAL_TEXT}" Do not use prior knowledge. Do not speculate.'
)


@cache
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key or None)


@contextmanager
def map_anthropic_errors() -> Iterator[None]:
    """Translate Anthropic SDK errors to HTTPException per DECISIONS.md §11.

    Mirrors `app/embeddings.py.map_voyage_errors` so operators see one
    taxonomy across both upstreams.
    """
    try:
        yield
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        raise HTTPException(status_code=500, detail=f"chat auth failure: {exc}") from exc
    except anthropic.BadRequestError as exc:
        raise HTTPException(status_code=400, detail=f"chat rejected input: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"chat rate limited: {exc}") from exc
    except anthropic.APIError as exc:
        # Catch-all: covers InternalServerError, APITimeoutError,
        # APIConnectionError, APIStatusError. APIError is the SDK's base.
        raise HTTPException(status_code=503, detail=f"chat upstream failure: {exc}") from exc


def build_search_result_block(chunk: RetrievedChunk) -> dict:
    """One retrieved chunk → one Anthropic search_result content block.

    `source` is the chunk's UUID (granularity matches retrieval — citations
    come back referencing this exact value, allowing direct dict lookup).
    `title` includes the published_date when present so the model can do
    temporal reasoning (DECISIONS.md §8 guardrail #4).
    """
    title = chunk.document_title
    if chunk.published_date is not None:
        title = f"{title} ({chunk.published_date})"
    return {
        "type": "search_result",
        "source": str(chunk.chunk_id),
        "title": title,
        "content": [{"type": "text", "text": chunk.text}],
        "citations": {"enabled": True},
    }


def generate_answer(
    question: str, chunks: list[RetrievedChunk]
) -> anthropic.types.Message:
    """Call Anthropic with the search_result blocks + question; return the raw Message.

    The caller (route layer) maps the returned content blocks into our
    ChatResponse shape. Anthropic SDK errors are mapped to HTTPException
    here via map_anthropic_errors; everything else propagates up.
    """
    content: list[dict] = [build_search_result_block(c) for c in chunks]
    content.append({"type": "text", "text": question})
    with map_anthropic_errors():
        return get_client().messages.create(
            model=CHAT_MODEL,
            max_tokens=CHAT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
