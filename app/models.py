from datetime import date
from enum import StrEnum
from typing import Any
from uuid import UUID

from fastapi import UploadFile
from pydantic import BaseModel, Field

from app.limits import MaxText


class Chunk(BaseModel):
    text: str
    ordinal: int
    token_count: int


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1)
    text: MaxText
    author: str | None = None
    published_date: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestDocumentRequest(BaseModel):
    """Form-encoded request body for POST /document.

    Fields ride the multipart envelope as individual form slots (per
    DECISIONS.md, Q1 of Step 4 planning). `metadata` is a JSON-encoded
    string because multipart form fields can't carry nested objects;
    the route parses it via json.loads after Pydantic validation.
    The 3M-char text cap applies to the *extracted* text and is enforced
    page-by-page in `app.extraction.extract_text` (raises `TextTooLargeError`,
    mapped to 422 by the route), not declared on this model.

    `file` is included so FastAPI's Annotated[Model, Form()] handler
    expands form fields and the file slot in one go. Declaring `file`
    as a separate UploadFile parameter alongside this model causes
    FastAPI to treat the model as a single body field instead of
    expanding its fields.
    """

    title: str = Field(min_length=1)
    author: str | None = None
    published_date: date | None = None
    metadata: str = "{}"
    file: UploadFile


class IngestResponse(BaseModel):
    document_id: UUID
    n_chunks: int


class RetrievedChunk(BaseModel):
    """One chunk + its parent-document context, returned by retrieval.

    Shared between GET /search (wrapped in SearchResponse.results) and POST
    /chat (subclassed by ChatSource which adds `cited` / `cited_text` per
    DECISIONS.md §8).

    `score` semantics depend on the pipeline stage. `app.retrieval.retrieve`
    populates it with the RRF fusion score (small numbers, ≤ 0.0328); the
    rerank stage (`app.rerank.rerank`) overwrites it with Voyage's
    `relevance_score`. Routes call rerank before returning, so the score
    seen by API clients is always the rerank score. It's not calibrated
    across queries — Voyage doesn't claim cross-query stability — so don't
    threshold on it.
    """

    chunk_id: UUID
    ordinal: int
    document_id: UUID
    document_title: str
    author: str | None
    published_date: date | None
    metadata: dict[str, Any]
    score: float
    text: str


class SearchResponse(BaseModel):
    results: list[RetrievedChunk]


class CitationRef(BaseModel):
    """One citation attached to an AnswerBlock — slim view of Anthropic's
    search_result_location. `chunk_id` is mapped from the SDK's `source`
    field; `document_title` and `published_date` are looked up from the
    retrieved chunk (not echoed from Anthropic) so the citation view stays
    in lockstep with `ChatSource` — same field names, same content.
    """

    chunk_id: UUID
    document_title: str
    published_date: date | None
    cited_text: str


class AnswerBlock(BaseModel):
    """One text block from Claude's response, optionally with attached
    citations. Mirrors Anthropic's response shape (a list of these is what
    `client.messages.create` returns under .content for text blocks).
    """

    text: str
    citations: list[CitationRef] = Field(default_factory=list)


class ChatSource(RetrievedChunk):
    """Retrieved chunk + chat-specific annotations.

    `cited` is True iff Claude grounded at least one claim on this chunk.
    `cited_text` collects the verbatim quotes — multiple if the chunk was
    cited from more than one answer block.
    """

    cited: bool = False
    cited_text: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """POST /chat response. Carries both the flat `answer` string (for
    simple display) and the native Anthropic block list (`answer_blocks`)
    so clients can render claim-with-quote inline. `sources` is all
    retrieved chunks (cited or not), per DECISIONS.md §8.

    `stop_reason` is passed through from Anthropic so clients can detect
    `max_tokens` truncation or `refusal` rather than treating an
    incomplete answer as a complete one. None on the structural
    empty-retrieval refusal path (no Anthropic call was made).
    """

    answer: str
    answer_blocks: list[AnswerBlock]
    sources: list[ChatSource]
    stop_reason: str | None


# DECISIONS.md §18 — knowledge-graph extraction shapes. The four entity
# types are the brief's three (PERSON/ORGANIZATION/LOCATION) plus EVENT;
# ARTIFACT is deliberately omitted (see §18.6). Resolution runs per type
# (cookbook pattern: same-name entities of different types stay distinct).
class EntityType(StrEnum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"


class Entity(BaseModel):
    name: str
    type: EntityType
    description: str


class Relation(BaseModel):
    source: str
    predicate: str
    target: str


class ExtractedGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class Cluster(BaseModel):
    canonical: str
    aliases: list[str]


class ResolvedClusters(BaseModel):
    clusters: list[Cluster]
