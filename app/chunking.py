from functools import cache

from semantic_text_splitter import TextSplitter

from app.embeddings import get_client, per_text_token_counts
from app.models import Chunk
from app.settings import settings

# Per DECISIONS.md §3: token-based splitting via `semantic-text-splitter`,
# 600 tokens / 15% overlap (~90 tokens). The splitter is constructed from
# Voyage's HuggingFace tokenizer (the same one used at embed time), so chunk
# sizes match exactly what the embedding API will see and the Rust splitter
# walks the tokenizer natively rather than round-tripping through a Python
# callback for each binary-search probe.
MAX_TOKENS = 600
OVERLAP_TOKENS = 90


@cache
def _get_splitter() -> TextSplitter:
    tokenizer = get_client().tokenizer(settings.embedding_model)
    return TextSplitter.from_huggingface_tokenizer(
        tokenizer, capacity=MAX_TOKENS, overlap=OVERLAP_TOKENS
    )


def chunk_text(text: str) -> list[Chunk]:
    if not text:
        return []

    pieces = _get_splitter().chunks(text)
    counts = per_text_token_counts(pieces)
    return [
        Chunk(text=p, ordinal=i, token_count=c)
        for i, (p, c) in enumerate(zip(pieces, counts))
    ]
