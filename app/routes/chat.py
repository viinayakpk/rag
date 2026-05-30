"""POST /chat — hybrid RAG + rerank + structured Anthropic citations (DECISIONS.md §7 / §7.2 / §8)."""

import anthropic
from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.db import ConnDep
from app.graph import resolve_entity_names
from app.graph_extract import extract_entities_from_question
from app.llm import REFUSAL_TEXT, generate_answer
from app.models import (
    AnswerBlock,
    ChatResponse,
    ChatSource,
    CitationRef,
    RetrievedChunk,
)
from app.rerank import rerank
from app.retrieval import RRF_CANDIDATES_PER_LANE, Filters, retrieve

router = APIRouter()

# DECISIONS.md §8 — top-K retrieval for chat.
CHAT_TOP_K = 8


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4096)
    # Opt-out for the graph pre-filter so the same /chat call can run as the
    # pre-graph baseline. Default True preserves existing behavior.
    use_graph: bool = True

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, conn: ConnDep) -> ChatResponse:
    # DECISIONS.md §18.9: auto-extract entities from the question and
    # resolve to UUID groups for the graph pre-filter. Per-name resolution
    # so a single canonical name extracted as multiple types (§18.11)
    # forms one group whose members are OR-matched in retrieval, rather
    # than collapsing into multiple "AND" requirements. Names that fail
    # to resolve silently drop — graceful degradation, not sentinel
    # padding — so /chat keeps whatever signal extraction surfaced rather
    # than emptying when one name is out-of-corpus.
    entity_id_groups: list[list[str]] = []
    if req.use_graph:
        raw_names = extract_entities_from_question(req.question)
        for name in raw_names:
            resolved = resolve_entity_names(conn, [name])
            if resolved:
                entity_id_groups.append(resolved)

    candidates = retrieve(
        conn,
        req.question,
        k=RRF_CANDIDATES_PER_LANE,
        filters=Filters(entity_id_groups=entity_id_groups),
    )

    # DECISIONS.md §18.9: never let the graph hide otherwise-relevant
    # content. If the entity filter narrowed candidates to zero (entities
    # the question named are in the corpus but never co-mentioned in a
    # single chunk), retry without the filter so the answer pipeline gets
    # a fair shot.
    if not candidates and entity_id_groups:
        candidates = retrieve(
            conn, req.question, k=RRF_CANDIDATES_PER_LANE, filters=Filters()
        )

    chunks = rerank(req.question, candidates, top_k=CHAT_TOP_K)

    # Refusal is owned by the system prompt + Claude's citation behavior
    # (DECISIONS.md §8): we don't pre-filter on a score threshold. The only
    # structural guard is "retrieval returned nothing" — no candidates means
    # there's nothing to send to the LLM. This pattern matches Anthropic's
    # contextual-retrieval blog, the pgvector RRF example, and Supabase's
    # hybrid_search RPC, none of which apply a rerank-score cutoff.
    if not chunks:
        return _refusal_response()

    message = generate_answer(req.question, chunks)
    return _build_response(message, chunks)


def _refusal_response() -> ChatResponse:
    return ChatResponse(
        answer=REFUSAL_TEXT,
        answer_blocks=[AnswerBlock(text=REFUSAL_TEXT, citations=[])],
        sources=[],
        stop_reason=None,
    )


def _build_response(
    message: anthropic.types.Message, retrieved: list[RetrievedChunk]
) -> ChatResponse:
    """Map Anthropic Message → our ChatResponse.

    Preserves the native answer_blocks shape; flattens to `answer` for
    simple display; annotates each retrieved chunk with cited/cited_text.
    """
    chunk_by_id = {str(c.chunk_id): c for c in retrieved}
    answer_blocks: list[AnswerBlock] = []
    citations_by_chunk: dict[str, list[str]] = {}

    for block in message.content:
        if block.type != "text":
            continue
        refs: list[CitationRef] = []
        for cite in block.citations or []:
            if cite.type != "search_result_location":
                continue
            # Defensive: Anthropic's contract says `source` echoes a value we
            # passed in, but if it ever doesn't, drop the citation rather than
            # 500 the request.
            chunk = chunk_by_id.get(cite.source)
            if chunk is None:
                continue
            refs.append(
                CitationRef(
                    chunk_id=chunk.chunk_id,
                    document_title=chunk.document_title,
                    published_date=chunk.published_date,
                    cited_text=cite.cited_text,
                )
            )
            citations_by_chunk.setdefault(cite.source, []).append(cite.cited_text)
        answer_blocks.append(AnswerBlock(text=block.text, citations=refs))

    answer = "".join(b.text for b in answer_blocks)
    sources = [
        ChatSource(
            **chunk.model_dump(),
            cited=str(chunk.chunk_id) in citations_by_chunk,
            cited_text=citations_by_chunk.get(str(chunk.chunk_id), []),
        )
        for chunk in retrieved
    ]
    return ChatResponse(
        answer=answer,
        answer_blocks=answer_blocks,
        sources=sources,
        stop_reason=message.stop_reason,
    )
