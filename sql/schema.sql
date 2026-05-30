-- Database schema for the RAG knowledge base.
-- Applied automatically on first container init via docker-compose.yml's
-- /docker-entrypoint-initdb.d mount. Reset with `docker compose down -v && docker compose up -d`.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT        NOT NULL,
    author          TEXT,
    published_date  DATE,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    raw_text        TEXT        NOT NULL,
    content_hash    TEXT        NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_documents_published_date ON documents (published_date);
CREATE INDEX idx_documents_metadata       ON documents USING GIN (metadata);

CREATE TABLE chunks (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT          NOT NULL,
    text         TEXT         NOT NULL,
    token_count  INT          NOT NULL,
    embedding    vector(1024) NOT NULL,
    -- DECISIONS.md §7 / §7.1: lexical lane for hybrid RRF. 'simple' config
    -- is language-neutral so mixed-language corpora aren't mis-stemmed.
    -- GENERATED so the column can never drift from `text`.
    tsv          tsvector     GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_chunks_tsv ON chunks USING GIN (tsv);

-- DECISIONS.md §18: Knowledge graph (closes issue #30).
-- Entities are corpus-global (not document-scoped) so a canonical entity
-- persists across document deletions; aliases, mentions, and relations
-- cascade-delete with their owning chunk/document.
CREATE TABLE entities (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name TEXT        NOT NULL,
    entity_type    TEXT        NOT NULL,
    description    TEXT        NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Same name across types is allowed ("Apple" ORG vs LOCATION); within
    -- a type the constraint forces resolution to collapse surface forms.
    UNIQUE (canonical_name, entity_type)
);

CREATE TABLE entity_aliases (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL,
    UNIQUE (entity_id, alias)
);

-- Case-insensitive alias lookup powers query-time ?entity=X resolution.
CREATE INDEX idx_entity_aliases_lower     ON entity_aliases (LOWER(alias));
CREATE INDEX idx_entity_aliases_entity_id ON entity_aliases (entity_id);

CREATE TABLE chunk_entity_mentions (
    chunk_id  UUID NOT NULL REFERENCES chunks(id)   ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE INDEX idx_cem_entity_id ON chunk_entity_mentions (entity_id);
CREATE INDEX idx_cem_chunk_id  ON chunk_entity_mentions (chunk_id);

-- source_chunk_id + source_document_id give every relation row provenance
-- back to the chunk that grounded it — enables future graph-answer endpoints
-- to cite the exact chunk via the existing search_result blocks.
CREATE TABLE relations (
    id                 UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id   UUID        NOT NULL REFERENCES entities(id)  ON DELETE CASCADE,
    predicate          TEXT        NOT NULL,
    target_entity_id   UUID        NOT NULL REFERENCES entities(id)  ON DELETE CASCADE,
    source_chunk_id    UUID        NOT NULL REFERENCES chunks(id)    ON DELETE CASCADE,
    source_document_id UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_relations_source   ON relations (source_entity_id);
CREATE INDEX idx_relations_target   ON relations (target_entity_id);
CREATE INDEX idx_relations_document ON relations (source_document_id);
