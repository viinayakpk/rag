# Design Decisions

This document captures the design decisions made before implementation, the alternatives considered for each, and why we chose what we chose. Where a decision deliberately keeps a future upgrade path open, the migration notes are recorded so the next person (or future me) can follow it without re-deriving the analysis.

This is a working artifact — written for the reviewer and for ourselves.

---

## 1. Stack

**Decision:** Python 3.14 + FastAPI + Pydantic v2 + `uv` for env/deps. Postgres 16 + pgvector via Docker Compose (`pgvector/pgvector:pg16`).

**Alternatives considered:**
- Node/TypeScript + Postgres+pgvector
- Python/FastAPI + dedicated vector DB (Qdrant, Weaviate)
- Python 3.12 instead of 3.14

**Why this choice:**
- PDF text extraction, embedding/LLM SDKs, tokenizers, and RAG tooling are all Python-first ecosystems. Node would mean fighting the ecosystem on a take-home.
- pgvector handles 100k+ chunks comfortably and gives us *one* database for documents, chunks, embeddings, metadata, *and* lexical ranking (via tsvector). A dedicated vector DB would force two stores in sync for no benefit at this scale.
- Python 3.14 because "default to the latest stable" is the rule. All deps in our list (`fastapi`, `pydantic`, `psycopg`, `voyageai`, `anthropic`, `pymupdf`) ship 3.14 wheels.
- Docker Compose because it's reproducible by the reviewer with one command and avoids hosted-service signup friction. Free-tier hosted Postgres (Supabase/Neon) was rejected: pausing free tiers, shared credentials, and using ~5% of a managed BaaS would be exactly the unnecessary abstraction the brief warns against.

**Migration notes:** None — this is a load-bearing decision. Don't change the language or vector store later.

---

## 2. Embedding model

**Decision:** Voyage `voyage-4`, 1024 dimensions, cosine distance.

**Alternatives considered:**
- `voyage-4-large` (same 1024-dim default with Matryoshka 256/512/2048; "best general-purpose & multilingual" tier — rejected as overshooting; "biggest model" needs corpus-specific benchmarks to defend, which a take-home doesn't have. Also has a tighter per-request token cap of 120K vs voyage-4's 320K, so any future migration would also need the embedder to re-tune its sub-batching limits.)
- `voyage-4-lite` (same dim options; "optimized for latency and cost" tier — rejected as undershooting on a quality-graded RAG eval where cost differences are pennies)
- `voyage-4-nano` (open-weight, smallest of the 4-series — interesting but no meaningful advantage over `voyage-4` here)
- `voyage-context-3` (1024-dim, contextual embeddings — each chunk's embedding incorporates surrounding-document context; deferred per migration notes below)
- `voyage-3` / `voyage-3.5` / `voyage-3-large` (previous generation per Voyage's own docs — Voyage explicitly recommends migrating to the 4-series; no reason to ship stale)
- OpenAI `text-embedding-3-small` (1536 dim — cross-vendor option; rejected to keep the "Anthropic stack end-to-end" narrative)
- Local sentence-transformers model (no inference infra; rejected)

**Why this choice:**
- `voyage-4` is Voyage's labelled general-purpose default for retrieval — using the recommended default is a stronger story than picking the biggest tier and defending it.
- Voyage is Anthropic's recommended embedding partner; clean README narrative ("used Anthropic's stack end-to-end").
- 1024 dimensions is a clean column type for pgvector with no storage bloat. Matryoshka variants (256/512/2048) are available without a model swap if storage or speed ever becomes a real constraint.
- Cosine via pgvector's `<=>` operator — the conventional choice for retrieval and the one Voyage's documentation examples use. voyage-4 embeddings are L2-normalized, so cosine, dot product, and Euclidean rank identically anyway.
- 32K-token per-input context length is far above our 600-token chunks — no risk of truncation.

**API usage notes (load-bearing for the embedder module):**
- **Always pass `input_type`.** Voyage prepends a different internal prompt depending on whether the text is being embedded as a *document* (for indexing) or as a *query* (for retrieval). Skipping `input_type` works but produces weaker retrieval. The embedder module exposes two functions for the two call sites: `embed_chunks(chunks, input_type="document") -> list[list[float]]` for ingestion and `embed_query(text, *, input_type="query") -> list[float]` for search/chat retrieval. Both default to the correct `input_type` for their call site so callers don't have to remember. Embeddings produced with vs without `input_type` are still mutually compatible — this is purely a quality lever.
- **Per-request caps for voyage-4:** ≤1,000 inputs AND ≤320,000 tokens per `/v1/embeddings` call. The embedder module owns sub-batching and enforces both ceilings simultaneously. With 600-token chunks, the token cap binds first at ~533 chunks per call; almost no real document hits either limit in one batch.
- **`output_dtype="float"`** (the default) — full-precision 32-bit embeddings. Quantized variants (int8 / binary) save storage but lose recall; not worth the complexity at this scale.

**Migration notes — voyage-4 → voyage-context-3 later (cheap):**
- Same default dimension (1024) → **no schema migration**.
- Same distance metric → no retrieval code change.
- Embedder module changes (~30 lines): different SDK method (`vo.contextualized_embed(...)`), different REST endpoint (`/v1/contextualizedembeddings`), and the API takes document-grouped chunk lists (`List[List[str]]`) rather than a flat list of strings.
- Tighter per-request caps: ≤1,000 inputs / ≤120K tokens / ≤16K chunks total — embedder's sub-batching needs to track all three.
- Ingestion pipeline must batch chunks **by document** (which is the natural flow anyway — one doc per upload).
- Re-embed all existing chunks (cents at this scale).
- Total effort: roughly half a day of focused work.
- **Design implication for now:** keep the embedder module behind a single function signature; don't interleave chunks from different documents in batched embedding calls.

---

## 3. Chunking strategy

**Decision:** Token-based splitting via `semantic-text-splitter` for both `/text` and `/document` endpoints. **600 tokens per chunk, 15% overlap (~90 tokens).** Token counting via Voyage's tokenizer, not characters — the splitter is constructed via `TextSplitter.from_huggingface_tokenizer` from Voyage's HuggingFace tokenizer (`get_client().tokenizer(model)` in `app/embeddings.py`), so the Rust splitter walks the same tokenizer the embedding API uses and chunk sizes match exactly what gets sent at embed time.

**Alternatives considered:**
- Fixed character/token splits (rejected — cuts mid-word/sentence)
- Sentence-grouping with `nltk`/`pysbd` (variable chunk sizes; long sentences blow the budget)
- Layout/structure-aware via `docling` (per-section/per-page metadata for PDFs — strong story but +half-day to the build)
- Semantic chunking (extra embedding calls during ingestion; modest gains over recursive in recent benchmarks)
- Hierarchical / parent-child (small chunks for retrieval, larger parents for LLM context — better quality, more complexity)
- LangChain's `RecursiveCharacterTextSplitter` (configurable token-based via `length_function`, but the LangChain text-splitter packages have a history of restructuring — splitting / renaming / re-homing classes — that's churn we'd inherit)
- Custom recursive implementation (what this codebase originally shipped — see "Library vs. custom" below)

**Why these specific numbers:**
- Library-driven splitting respects natural boundaries when possible while keeping chunk sizes predictable.
- 600 tokens because Voyage embeddings are trained for general text in this range; memos/reports rarely have meaningful units shorter than ~150 tokens (paragraph) or longer than ~700 (a few paragraphs). 600 is the middle, keeping citations precise enough to spot-check.
- 15% overlap because boundary-straddling sentences would otherwise be split across chunks and missed by retrieval; 0% loses content, 25%+ inflates results with near-duplicates.

**Library vs. custom:** the original implementation was ~125 lines of hand-rolled recursive splitting with a token-offset fallback for non-Latin scripts. The case for custom rested on two points: visibility of the algorithm in-tree, and explicit handling of CJK / no-space scripts. Under scrutiny only the first holds up — `semantic-text-splitter` walks Unicode graphemes and packs them to a token capacity, so multilingual handling isn't a unique advantage of the custom code. The library is more battle-tested for less code; the swap is the right call once the "I understand the algorithm" signal has been delivered elsewhere (PR history, this document).

**Migration notes — flat → parent-child (1 focused day if architecture is clean):**
- Schema: add `parent_id` column on `chunks` referencing another row (or a separate `parent_chunks` table).
- Chunker module: emit two-level output (parents + children with linkage).
- Retrieval: search hits children, JOIN to parents, DISTINCT to avoid duplicate parent text in LLM context.
- Citations: cite the *child* (precise quote), pass the *parent* (rich context) to the LLM. Prompt + response shape gain one field.
- Re-ingest all documents (cents at this scale).
- **Design implication for now (already baked into the plan):**
  1. Chunker is its own module with signature `chunk_text(text: str) -> list[Chunk]`. Endpoints don't know how splitting works. Adding a `metadata` parameter for parent-child is a local change to this signature.
  2. Retrieval is its own module with signature `retrieve(conn: Connection, query: str, k: int, filters: Filters) -> list[RetrievedChunk]`. Endpoints don't write SQL.
  3. `Chunk` is a Pydantic model. Adding `parent_id: UUID | None = None` later is one line.

**Migration notes — plain-text → structure-aware PDF via `docling` (~half-day):**
- Replace pymupdf extraction with docling for `POST /document` only.
- Docling returns Markdown with hierarchy preserved → richer per-chunk metadata (section, page).
- Chunker signature stays the same.
- README will mention this as the natural next move.

---

## 4. Metadata schema

**Decision:** Hybrid — typed columns for well-known fields + JSONB for the long tail.

**Required (typed) on upload:** `title`.
**Optional (typed):** `author`, `published_date`.
**Catch-all:** `metadata jsonb` with a GIN index for arbitrary extras (including any categorical tagging the user wants to attach).

**A note on `source_type`:** earlier drafts of this design included a typed `source_type` column with a fixed taxonomy (`'memo' | 'report' | 'article' | 'text'`) inspired by the brief's mention of "memos, reports, articles." The brief uses that phrase only as background context describing what users might upload — it does *not* require a fixed categorization, and the metadata schema is explicitly "what you define." Baking the taxonomy into the schema would be inventing constraints the brief doesn't impose. Categorical tagging instead lives in the JSONB `metadata` column (e.g., `{"type": "memo", "department": "finance"}`) and is filterable via `@>` containment, indexed by GIN.

**Alternatives considered:**
- Pure JSONB (one `metadata` column, anything goes — flexible but type-blind; weak data-modeling story)
- Pure typed columns (rigid; bad fit for a generic knowledge base where customers define what they send)
- Denormalizing filter fields onto the `chunks` table for query speed (premature optimization at this scale)

**Why this choice:**
- Pure JSONB fails on date filtering — there's no validation that `published_date` is actually a date, and range queries silently break.
- Pure typed columns are a migration every time a customer wants a new field.
- Hybrid gives validated SQL filters where it matters and flexibility everywhere else. Standard answer for a flexible-but-principled KB schema.
- Filtering happens via JOIN to documents at query time — single source of truth, no denormalization, Postgres handles it fine at this scale.

**Schema sketch:**

```sql
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    author          TEXT,
    published_date  DATE,
    metadata        JSONB NOT NULL DEFAULT '{}',
    raw_text        TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,     -- SHA-256 for dedupe
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_documents_published_date ON documents (published_date);
CREATE INDEX idx_documents_metadata       ON documents USING GIN (metadata);

CREATE TABLE chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    text         TEXT NOT NULL,
    token_count  INT NOT NULL,
    embedding    vector(1024) NOT NULL,
    UNIQUE (document_id, ordinal)
);
CREATE INDEX idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

The `tsv` column and its GIN index land in Step 7 (§7) — `tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED` plus a GIN index for the lexical lane of hybrid retrieval.

**Filter API on `/search`:**
- Typed query params: `author`, `published_after`, `published_before`
- JSONB containment: `metadata @> '{"type": "memo"}'` (the API surface accepts simple `meta.<key>=<value>` query params and translates them to JSONB containment server-side)
- No DSL, no `$and`/`$or`/`$gt` operators — the brief is about RAG, not query languages.

---

## 5. PDF extraction

**Decision:** `pymupdf` (fitz).

**Alternatives considered:**
- `pypdf` (BSD-style, pure Python, lower text quality)
- `pdfplumber` (table-aware, slower, prose quality similar to pypdf)
- `docling` (structure-aware via ML-based layout parsing — overkill for the plain-text-only path we picked, and §5 isn't where the structure-aware decision belongs anyway; deferred to §15 / [#26](https://github.com/nainajnahO/rag-knowledge-base/issues/26))
- `unstructured` (older structure-aware option; superseded by docling)

**Why this choice:**
- Best plain-text extraction quality of the lightweight options, fastest by a margin.
- We're not using structure (deferred to README's "next steps") so a heavier structure-aware library would be paying for capability we don't use.

**Trade-off — AGPL license:**
PyMuPDF is dual-licensed: AGPL by default, with a commercial license available from Artifex. Strict reading of AGPL would require source disclosure for a commercial network-deployed RAG service. **README will explicitly call this out** along with the path to swap for pypdf — extraction is one module, ~50 lines, same-day swap.

**Migration notes — pymupdf → docling (for structure-aware chunking, ~half-day):**
- Replace extraction module body.
- Chunker signature stays the same.
- Re-ingest PDFs to capture section/page metadata.
- Adds richer chunk-level metadata (section title, page number) → stronger citations.

---

## 6. Vector index

**Decision:** HNSW with default parameters: `m=16, ef_construction=64`. Cosine ops (`vector_cosine_ops`).

**Alternatives considered:**
- No index (sequential scan — fast at take-home scale, weaker story)
- IVFFlat (cluster-based, lower recall, requires retuning `lists` as data grows; legacy in 2026)

**Why this choice:**
- HNSW is the production default in pgvector 0.5+. Best recall/speed tradeoff.
- IVFFlat has no advantage for new projects.
- "No index works at this scale" is true but tells a worse story than "I picked the right index and used the documented defaults."

**One subtle thing worth knowing:** HNSW gives approximate nearest neighbor (recall typically >95%, not 100%). If a niche query depends on a single rare chunk, that miss rate matters. Mitigation if needed: `SET LOCAL hnsw.ef_search = 100` per query — trades a bit of latency for recall.

### 6.1 Iterative scan for filtered queries

`/search` and `/chat` retrieve with metadata filters joined to `documents`. Naive HNSW + post-filter can underfill `LIMIT k` when filters reject most candidates: HNSW returns its initial batch (`hnsw.ef_search`, default 40), the filter rejects some, and you get back fewer than `k`. To handle this, `app/retrieval.py` sets `SET LOCAL hnsw.iterative_scan = 'strict_order'` (pgvector ≥0.8.0) inside the retrieval transaction. The index keeps walking the graph past its initial batch until enough rows pass `WHERE` to fill `k`, while preserving similarity ordering. Bounded by `hnsw.max_scan_tuples` (default 20,000) so very selective filters don't trigger a full scan.

`strict_order` over `relaxed_order` because results are returned to the client in displayed score order — out-of-order results would need an outer sort to fix, gaining nothing at this scale.

**Migration notes:** none. The GUC is per-transaction; bumping `max_scan_tuples` if very selective filters routinely underfill is a one-line SET inside `retrieve()`.

**Validation gap (honest note):** at the smoke-test corpus size (~5 chunks), the planner picks Seq Scan + Hash Join, not HNSW — `EXPLAIN` confirmed. Iterative scan only engages when the planner has chosen an HNSW Index Scan node, which won't happen until the corpus is large enough that the HNSW path beats brute-force on cost. The mechanism is therefore production-default insurance, not validated empirically at the only scale this code has been tested at.

**Why iterative scan and not over-fetch CTE (revisited).** During PR #18 review we re-examined whether to swap `SET LOCAL hnsw.iterative_scan` for an over-fetch CTE pattern (inner CTE pulls top-200 by distance, outer JOIN+filter+LIMIT k). Verified the alternative empirically with `EXPLAIN ANALYZE` on the live DB: at the small-corpus / no-HNSW-engaged case, our current shape is **strictly better** — 0.16ms vs 0.28ms execution, 38 vs 67 buffer hits, and it has no underfill failure mode (the JOIN+filter eliminates rows before the sort, where over-fetch CTE truncates to 200 candidates and only then filters). The case for over-fetch CTE was never the no-HNSW path; it was decoupling from the planner's HNSW decision when HNSW *would* be useful but the planner refuses. pgvector 0.8.0's improved cost model is supposed to fix that planner reluctance directly — going back to over-fetch CTE in 2026 is "doing things the pre-0.8.0 way." Decision: keep `SET LOCAL hnsw.iterative_scan`, accept the validation gap, trust pgvector's design over re-implementing it in SQL. Issue #19 closed with these findings.

**Internal cosine-derived score** (not the API-exposed score after Step 7): the dense-lane vec CTE in `app/retrieval.py` orders by `c.embedding <=> %(query_vec)s::vector`, where `<=>` is cosine distance in `[0, 2]`. Internally, `1 - distance ∈ [-1, 1]` mathematically; voyage-4 embeddings are L2-normalized, and natural-language pairs almost always land ≥ 0. **The API does not surface this score.** What clients see in the `/search` and `/chat` responses is Voyage's rerank `relevance_score` — see §7.2. The cosine score is purely an ordering input to the RRF fusion, not a client-visible quantity.

---

## 7. Hybrid search fusion

**Decision:** Reciprocal Rank Fusion (RRF), constant `k=60`.

**Terminology note:** This section uses "lexical ranking" or "lexical lane" rather than "BM25" for the keyword-search component. Postgres's full-text search ships with `ts_rank` and `ts_rank_cd`, which are term-frequency-based ranking functions — *not* the canonical BM25 formula (which adds IDF and saturation parameters `k1`, `b`). The role is identical (lexical complement to the dense lane) and RRF fusion is rank-based, so the specific score formula matters less than the rank ordering it produces. Real BM25 in Postgres requires the `pg_search` extension (ParadeDB), deemed unnecessary at take-home scale. The brief uses "BM25" as a category label; we read it that way.

**Alternatives considered:**
- Linear weighted combination `α · vector + (1−α) · lex` (requires score normalization and tuned α; `lex` here standing in for whatever lexical score we use)
- Convex combination with min-max or z-score normalization (same problem, more steps)
- CombSUM / CombMNZ (older, superseded by RRF)
- Skipping hybrid entirely (vector-only)

**Why this choice:**
- Parameter-free — no tuning data needed, no α to defend.
- No score normalization needed (lexical scores from `ts_rank` are query-dependent in magnitude, cosine is `[-1, 1]`; mixing them naively is broken).
- Industry standard in 2026 (Elastic, Vespa, Weaviate, Qdrant all default to RRF).
- One SQL query with two CTEs and a JOIN — ~20 lines, no Python-side merging.

In the SQL sketch below, `$1` is the query embedding, produced by the same embedder module used during ingestion but called with `input_type="query"` per §2's API usage notes; `$2` is the raw query string.

**Implementation sketch** (matches what shipped in `app/retrieval.py`):

```sql
WITH vec AS (
  SELECT c.id,
         ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1) AS rank
  FROM chunks c JOIN documents d ON d.id = c.document_id
  WHERE  <metadata filters>
  ORDER BY c.embedding <=> $1 LIMIT 50
),
lex AS (
  SELECT c.id,
         ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS rank
  FROM chunks c JOIN documents d ON d.id = c.document_id,
       plainto_tsquery('simple', $2) q
  WHERE c.tsv @@ q
    AND  <metadata filters>
  ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT 50
),
fused AS (
  SELECT COALESCE(vec.id, lex.id) AS id,
         COALESCE(1.0/(60+vec.rank), 0) + COALESCE(1.0/(60+lex.rank), 0) AS score
  FROM vec
  FULL OUTER JOIN lex ON vec.id = lex.id
)
SELECT c.*, d.*, fused.score
FROM fused
JOIN chunks c    ON c.id = fused.id
JOIN documents d ON d.id = c.document_id
ORDER BY fused.score DESC
LIMIT $3;
```

**Three load-bearing details** that diverge from a naïve sketch:

1. **`FULL OUTER JOIN`, not `LEFT JOIN`.** A `LEFT JOIN` anchored on either lane silently drops rows the other lane found exclusively — defeating fusion for the queries the lexical lane exists to catch (rare proper nouns, IDs, technical jargon). pgvector's `rrf.py` and Supabase's `hybrid_search` both use `FULL OUTER JOIN`.
2. **`ts_rank_cd`, not `ts_rank`.** Cover-density variant — proximity-aware. Free upgrade and what both reference examples use.
3. **Metadata filters live inside both CTEs**, not the outer SELECT. A selective filter applied late would leave each lane's top-50 mostly empty after post-filter rejection; applied early, each lane's candidate pool is already 50 post-filter survivors.

**Primary sources for the recipe:**
- pgvector RRF example — github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/rrf.py
- Supabase hybrid_search RPC — supabase.com/docs/guides/ai/hybrid-search

### 7.1 Lexical lane language config — `'simple'`

The `tsvector` column needs a Postgres text-search config that governs tokenization, stemming, and stop-word filtering. The brief doesn't specify a corpus language, so we picked the language-neutral option.

**Alternatives considered:**
- **`'english'`** — Porter stemmer + English stop words. Best lexical recall on English prose. Actively bad on non-English text: mis-stems Norwegian, French, etc., turning the lexical lane into noise on those docs.
- **A specific non-English config** (e.g., `'norwegian'`) — same shape, different bet on which single language wins. Wrong if the corpus turns out to be in a different language.
- **Per-document language tracking** — store `language` per document, copy onto each chunk, generate `tsv` with the per-row config (`to_tsvector(language, text)`). Properly multilingual lexical ranking, but requires language detection at upload and query time, plus a denormalized `language` column on chunks (Postgres GENERATED columns can only reference same-row data). See migration notes below.

**Why `'simple'`:**
- The lexical lane's main value-add in our hybrid is exact-term matching (proper nouns, codes, acronyms, technical jargon). `'simple'` delivers that for any language.
- The morphology and stop-word lift of a language-specific config (`"policies"` matching `"policy"`) is partially absorbed by voyage-4 in the dense lane — semantically equivalent inflections produce nearby vectors.
- The brief is silent on language. `'english'` would be a bet that *penalizes* non-English content rather than just leaving it un-optimized.
- Cross-language retrieval is unaffected by this choice — voyage-4 (multilingual) carries that case through the dense lane regardless of what we put in the lexical lane.
- **Note** that pgvector's reference example and Supabase's `hybrid_search` both use `'english'` by default. We deviate deliberately because *this* corpus is verifiably mixed-language at retrieval time: the experimental ingest covered English academic papers alongside Swedish thesis drafts, Swedish grade reports, and Swedish real-estate listings. `'english'` would mis-stem the Swedish content; `'simple'` is the right pick for a multilingual KB even though it's not the most common default.

**Migration notes — `'simple'` → per-document language tracking (~half-day):**
- Add `language regconfig NOT NULL` on `documents` (default `'simple'`) — detected at upload via a small library (`langdetect`, `fasttext-langid`) or supplied by the client.
- Add `language regconfig NOT NULL` on `chunks` — denormalized from the parent doc on insert (Postgres GENERATED columns can't reference other tables).
- Drop the existing `tsv` column and `idx_chunks_tsv` GIN index.
- Re-add `tsv` as `GENERATED ALWAYS AS (to_tsvector(language, text)) STORED`.
- Recreate the GIN index.
- At query time: detect (or accept via query param) the query's language, parse with `plainto_tsquery(detected_lang, :q)`.
- Tracked in [issue #3](https://github.com/nainajnahO/rag-knowledge-base/issues/3).

### 7.2 Reranker — Voyage `rerank-2.5`

**Decision:** After RRF returns 50 candidates, rerank with Voyage `rerank-2.5` and keep the top-K (K=10 for `/search`'s default, K=8 for `/chat` per §8). The rerank `relevance_score` replaces the RRF score on the returned `RetrievedChunk`. Lives in `app/rerank.py` as a thin module called by routes — retrieval and reranking are separate pipeline stages, not one fused operation.

**Alternatives considered:**
- **No reranker** (the previous §15 deferred-cut framing — "RRF gets us most of the way there"). Anthropic's *Introducing Contextual Retrieval* post documents rerank as the fourth and final stage of their recommended pipeline; Voyage ships `rerank-2.5` in the embedding stack we already use. The marginal cost is ~30 lines, one extra Voyage call, ~150ms latency, and reuses `map_voyage_errors`. Empirically validated on this corpus via a hybrid-only A/B over the 6-query regression set (PR description has the full per-query table): rerank is the load-bearing stage for the Q4 fix — without rerank, the Ahody chunk reaches the 50-candidate pool via the lex lane but RRF ranks cover letters higher (they overlap on "hiring" and "work sample" in *both* lanes), so the chunk never enters Claude's top-8 window. Rerank-2.5, reading the full query, surfaces it to top-1. **Trade-off:** rerank regresses Q1 slightly — drops the primary "Impact of Weather Factors" paper from top-5 in favor of student writing about it. Net positive on this corpus for the Q4-class failure mode that motivated adding hybrid retrieval in the first place.
- **Cohere `rerank-3.5`** or other cross-encoders. Voyage is Anthropic's recommended embedding/reranking partner — same vendor as `voyage-4`, same SDK, same error class hierarchy reused in `app/rerank.py`. Switching would split the upstream story.
- **LLM-as-reranker** (have Claude score chunks). 60× more expensive per Voyage's published comparison and adds an extra LLM round-trip on the hot path.

**Why this choice:**
- The pipeline shape Anthropic's *Introducing Contextual Retrieval* post documents (anthropic.com/news/contextual-retrieval) is retrieve → rerank → top-K to LLM. We ship the rerank step here. The blog also recommends contextual-embedding rewriting (deferred — see §15's `voyage-context-3` row); we adopt one of the two extensions Anthropic recommends layering on top of vanilla hybrid, not both.
- Reuses `map_voyage_errors` from `app/embeddings.py` — same 401/429/500/503 hierarchy across both Voyage calls, no new error code paths to reason about.
- Voyage doesn't claim `relevance_score` is calibrated across queries (the docs make no such statement, and never publish a recommended threshold). We do **not** apply a score threshold; see §8 for how refusal is handled instead.
- Single SDK call per `/search` or `/chat`, ~150ms added latency, ~$0.005 per call at the documented price.

**Migration notes:** Model alias is in one constant (`RERANK_MODEL` in `app/rerank.py`). Swap to a Cohere reranker by replacing the body of `rerank()` and adding a `map_cohere_errors` context manager mirroring §11; signature stays the same.

**Token-budget interaction (load-bearing for future migrations).** Rerank-2.5 has a 32K-token combined input budget (query + all candidate documents). With §3's 600-token chunks and `RRF_CANDIDATES_PER_LANE = 50` from §7, the worst case is ~30,000 token-document budget plus query — under the cap with ~2K margin. The budget depends on three independent config values that aren't linked in code: chunk size (§3), candidate-pool size (§7), and reranker context (this section's `RERANK_MODEL`). Bumping any one can silently exceed it; Voyage's SDK defaults `truncation=True` so overflow degrades quality silently rather than raising. Whoever migrates §3 to parent-child chunking with larger parent chunks, or §7 to a bigger candidate pool, needs to either (a) verify the budget still holds, (b) sub-batch the rerank call analogous to `app/embeddings.py:embed_chunks`'s per-request cap discipline, or (c) explicitly opt into truncation as a documented quality trade-off. No runtime guard is in place today — same posture as §6.1's "Validation gap" note: make the precondition legible, don't enforce it at this scale.

---

## 8. Citation approach (the load-bearing one)

**Decision:** Anthropic's first-class **Search Result content blocks** with structured citations (`cited_text` returned in API response). Each retrieved chunk becomes a `search_result` block with `citations.enabled = true`; Claude's response is a list of text blocks where cited blocks carry `citations: list[search_result_location]` with verbatim `cited_text`. We pass through Anthropic's native shape rather than remapping into a custom `[N]` numbered-citation scheme.

**Alternatives considered:**
- **Prompt-based numbered citations (`[1]`, `[2]`)** — what we originally locked. Teach the model `[N]` syntax in the system prompt, parse `[N]` markers from the answer text, map back to chunks. Works, but Anthropic's own evals (their public docs) report this approach hallucinates citations significantly more than their structured-citations feature, and `[N]` is parseable-but-not-guaranteed (the model can fabricate a `[5]` when only 4 chunks exist).
- **Structured JSON output with per-claim `supported_by` arrays** — incompatible with Citations per Anthropic's docs (a 400 from the API). Skipped on those grounds; would also re-introduce the prompt-engineering verifiability burden.
- **Span-level citation via prompt** — model returns the exact substring supporting each claim. Brittle; models hallucinate spans. Anthropic's structured citations are precisely this idea, done correctly at the API level.
- **Post-hoc verification pass** — second LLM call to validate each claim. 2× cost, 2× latency, doesn't add over what Anthropic's structured citations already guarantee.

**Why this choice:**
- The brief grades **"is it actually possible to verify what the LLM says?"** and **"does the system hallucinate?"** Anthropic's structured-citations API is the strongest available answer to both:
  - `cited_text` is **guaranteed** to be a verbatim substring of the provided chunk (per Anthropic docs: *"citations are guaranteed to contain valid pointers to the provided documents"*).
  - In Anthropic's published evals, the structured-citations feature *"was found to be significantly more likely to cite the most relevant quotes from documents as compared to purely prompt-based approaches."*
- Returning all retrieved chunks (not just cited ones) lets the reviewer see what the model had access to and judge if it cited the right ones — preserved from the original §8 intent.
- The system prompt becomes shorter and easier to audit because the API handles citation formatting; we only have to enforce three guardrails (sources-only, refusal allowed, no speculation).
- Native Anthropic shapes are passed through to the response (`answer_blocks`) so a client can render claim-with-quote inline if it wants. A flat `answer: str` is also returned for simple display.

**Anti-hallucination guardrails (four — one mechanism revised in Step 7):**

1. **System prompt forbids external knowledge.** *"You answer questions using only the provided search results... Do not use prior knowledge. Do not speculate."* The single most important line.
2. **Refusal allowed and encouraged.** *"If the search results do not contain enough information to answer the question, say: 'I don't have enough information in the provided sources to answer this.'"* In-prompt instruction.
3. **Structural empty-retrieval guard.** Lives in the route layer: if `retrieve()` followed by `rerank()` returns zero candidates, return the refusal response with `sources: []` without calling Anthropic. This fires when metadata filters reject every chunk or the corpus is empty — there's literally nothing to send. **No score threshold.** The previous gate (cosine ≥ 0.5, dropped in Step 7) doesn't fit the new pipeline: Anthropic's *Introducing Contextual Retrieval* post, pgvector's RRF example, and Supabase's `hybrid_search` RPC all take top-K unconditionally; none threshold rerank scores. Voyage doesn't document `relevance_score` as a calibrated `[0,1]` quantity, so picking a fixed cutoff would tune against an uncalibrated, query-dependent signal. Refusal on low-relevance retrieval is owned by Claude's response to guardrails #1 and #2 — the pattern Anthropic, pgvector, and Supabase all use.
4. **Chunk metadata passed automatically.** Each `search_result` block has a `title` field set to `"<doc_title> (<published_date>)"` (date when present). The model sees this natively without any prompt-engineering — no need to inject `Title: ... (date)` headers into chunk text.

**Top-K retrieval for chat:** 8 chunks. The rerank stage (§7.2) trims a 50-candidate RRF pool to the top 8, which carry semantically-relevant content for nuanced questions without dilution.

**Per-chunk packaging.** Each retrieved chunk → one `search_result` content block:

```python
{
    "type": "search_result",
    "source": str(chunk.chunk_id),                 # UUID — granularity matches retrieval
    "title": f"{doc_title} ({published_date})"     # date appended when present
            if published_date else doc_title,
    "content": [{"type": "text", "text": chunk.text}],   # one block per chunk
    "citations": {"enabled": True},
}
```

`source = chunk_id` (not `document_id`) because the verifiability story is "this exact quote came from this exact chunk." Citations come back with `source` carrying the same `chunk_id`, so we can do an exact dict lookup `chunk_by_id[citation.source]` to map back.

One block per chunk for this PR. Splitting chunks into sentence-level blocks would give finer citation granularity (`cited_text` is the joined content of `content[start_block_index:end_block_index]`, so finer blocks = shorter cited quotes). Deferred — `cited_text = full chunk` is already verifiable by `grep`-ing our `chunks.text`. Sentence-splitting is a local change in `app/llm.py`'s `build_search_result_block` if/when we want it.

**Request structure (Anthropic's "Method 2: top-level search results"):** user message contains the 8 `search_result` blocks first, then a `text` block with the question. No tool-use loop (Method 1 is for dynamic tool-driven RAG; we have pre-fetched content).

**Response shape:**

```json
{
  "answer": "Revenue grew 12% in Q3 2025, driven by enterprise contracts.",
  "answer_blocks": [
    {
      "text": "Revenue grew 12% in Q3 2025, driven by enterprise contracts.",
      "citations": [
        {
          "chunk_id": "uuid",
          "document_title": "Q3 Revenue Memo",
          "published_date": "2025-10-15",
          "cited_text": "Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts."
        }
      ]
    }
  ],
  "sources": [
    {
      "chunk_id": "uuid",
      "ordinal": 0,
      "document_id": "uuid",
      "document_title": "Q3 Revenue Memo",
      "author": "Finance Team",
      "published_date": "2025-10-15",
      "metadata": {"type": "memo", "department": "finance"},
      "score": 0.87,
      "text": "Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts. The finance team expects continued growth into Q4...",
      "cited": true,
      "cited_text": ["Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts."]
    }
  ],
  "stop_reason": "end_turn"
}
```

`answer` is the concatenation of all `answer_blocks[i].text` for clients that just want a string. `answer_blocks` is Anthropic's native shape (preserving the per-block citation linkage). `sources` is all 8 retrieved chunks (not just cited ones, per the original §8 intent — let the reviewer see what the model had access to). `cited: bool` distinguishes which chunks the model actually grounded claims on; `cited_text: list[str]` carries the verbatim quotes (multiple if the model cited the same chunk in multiple `answer_blocks`). `stop_reason` is passed through from Anthropic — `"end_turn"` is the healthy case; clients should treat `"max_tokens"` as a truncation signal and `"refusal"` as a model-level decline.

**Refusal response shape** (zero chunks retrieved after rerank — guardrail #3):

```json
{
  "answer": "I don't have enough information in the provided sources to answer this.",
  "answer_blocks": [
    {
      "text": "I don't have enough information in the provided sources to answer this.",
      "citations": []
    }
  ],
  "sources": [],
  "stop_reason": null
}
```

`sources: []` because we never called Anthropic — the LLM didn't "have access to" anything. `stop_reason: null` for the same reason — there was no model call to report a stop_reason for. If the reviewer wants to see what retrieval found, they can hit `/search?q=<question>` directly.

**Migration notes — one block per chunk → sentence-splitting (~hour, local):**
- In `app/llm.py.build_search_result_block`, split `chunk.text` into sentences (`pysbd` or simple `re.split`), emit one `{"type": "text", "text": s}` per sentence in the `content` array.
- No schema migration, no other code changes. `cited_text` becomes shorter and per-claim-precise.
- Tracked as a follow-up issue if quality demands surface.

**Migration notes — Anthropic structured citations → multi-vendor portability (~half-day):**
- This locks `/chat` into Anthropic's API shape. If we ever swap LLM providers, we'd need to reimplement citations. §9 already locked Sonnet 4.6 and treats LLM swap as "one config line" — so this isn't really new lock-in.
- If/when we want to abstract: introduce a `Citation` adapter layer that wraps either Anthropic's structured citations or a prompt-based fallback for other providers. Adds one indirection layer.

---

## 9. Chat LLM

**Decision:** Claude Sonnet 4.6.

**Alternatives considered:**
- Claude Haiku 4.5 (~4x cheaper, sufficient for grounded extractive RAG, slightly more risk on multi-source synthesis)
- Claude Opus 4.7 (overkill for grounded synthesis; hard to defend on cost)

**Why this choice:**
- Sonnet is the production default for RAG — best balance of synthesis quality, refusal behavior, and citation discipline.
- Cost difference vs Haiku is irrelevant at demo scale (~2¢ vs ~0.7¢ per call).
- Haiku is the sharper "I thought about cost" answer, but only if defended with benchmarks. Without benchmark data, Sonnet is the unimpeachable choice.

**Migration notes:** Model name lives in one config constant / env var. Swapping is one line.

**Constants:** `app/llm.py` exposes `CHAT_MODEL = "claude-sonnet-4-6"` (alias, not a dated snapshot — auto-tracks the latest minor revision; swap to `"claude-sonnet-4-6-YYYYMMDD"` for production reproducibility) and `CHAT_MAX_TOKENS = 2048`.

---

## 10. /chat shape — single-turn

**Decision:** Single-turn / stateless. Request: `{question}`. Response: `{answer, answer_blocks, sources, stop_reason}` (full shape in §8).

**Alternatives considered:**
- Multi-turn with client-managed message history (adds query-rewriting complexity)
- Multi-turn with server-managed sessions (adds session storage, state management; overkill)

**Why this choice:**
- The brief literally says *"POST /chat takes a question."* Single-turn matches the spec.
- Multi-turn opens a real RAG design problem — query rewriting for follow-up retrieval (the latest user turn alone has no semantic content to retrieve on; the full conversation pollutes embeddings; the right move is an LLM call to rewrite the latest turn into a standalone query before embedding). Doing this badly is worse than not doing it.

**Multi-turn upgrade path** — tracked in [issue #25](https://github.com/nainajnahO/rag-knowledge-base/issues/25). Sketch: client passes the conversation as `messages`, the server uses the LLM to rewrite the latest turn into a standalone search query (informed by history), then retrieves on the rewritten query, then answers with the original conversation in the LLM's context. The non-trivial part isn't the API shape — it's the query rewriter. Defers cleanly into a follow-up PR.

---

## 11. API & error shapes

**Decision:** Plain success bodies, FastAPI default error shape (`{"detail": "..."}`).

**Alternatives considered:**
- Wrapped envelope (`{data: ..., error: ...}` or similar)

**Why plain:**
- It's what FastAPI does by default — wrapping is extra code that adds noise to curl examples.
- The brief explicitly says they're not evaluating perfect error handling, and adding a custom error framework would be the unnecessary abstraction the brief warns against.

**Status codes used explicitly:**

- **422** — Pydantic validation (free). Includes the 3M-character extracted-text cap on both `/text` and `/document` (see §17).
- **413** — multipart file body exceeds the 25 MB cap on `POST /document` (see §17). Rejected at the door before extraction.
- **400** — bad file/input (e.g., empty text, malformed PDF, missing `%PDF-` magic bytes) AND upstream rejection of the input (Voyage `InvalidRequestError`).
- **401** — missing/bad API key. The dependency is **not wired** today (Step 8 skipped — see §14, §15, [issue #29](https://github.com/nainajnahO/rag-knowledge-base/issues/29)). The status code remains in this taxonomy as the shape that would ship if relevance changed.
- **429** — upstream rate limited (Voyage `RateLimitError`).
- **500** — server misconfig (e.g., missing/invalid `VOYAGE_API_KEY` surfacing as Voyage `AuthenticationError`). Distinguished from 503 so operators don't mistake a config bug for a transient upstream issue.
- **503** — upstream transient failure (generic `VoyageError`; Anthropic upstream errors in Step 6 will follow the same hierarchy).

**Vendor error mapping locations.** Voyage SDK errors → `map_voyage_errors()` in `app/embeddings.py`; used by `embed_with_error_mapping` (chunks, in `app/ingest.py`) and `embed_query_with_error_mapping` (queries, in `/search` and `/chat`'s retrieval call). Anthropic SDK errors → `map_anthropic_errors()` in `app/llm.py`; used by `generate_answer` for `/chat`. Both context managers map to the same status-code hierarchy so operators see one taxonomy regardless of which upstream failed:
- `AuthenticationError` / `PermissionDeniedError` → 500 (server misconfig — operators don't mistake a missing/wrong API key for a transient blip and wait it out).
- `BadRequestError` / `InvalidRequestError` → 400 (upstream rejected our payload).
- `RateLimitError` → 429.
- Catch-all `APIError` (Anthropic) / `VoyageError` (Voyage) → 503 (timeout, ServiceUnavailable, generic upstream).

No retries, no circuit breakers, no custom exception hierarchy.

---

## 12. Upload dedupe

**Decision:** SHA-256 hash of raw content stored on `documents.content_hash` (UNIQUE). On collision, return the existing `document_id` instead of creating a duplicate.

**Alternatives considered:**
- No dedupe (simpler, but reviewer-uploads-twice produces noisy duplicate search results)

**Why hash dedupe:**
- ~10 lines of code.
- Real "thought about this" signal for the reviewer.
- Saves embedding cost on accidental re-uploads.
- Doesn't preclude versioning later — the hash is alongside, not as, the primary key.

---

## 13. Tests

**Decision:** Three pytest smoke tests (test 1 covers four cases — short, long-with-overlap, empty, and a no-separator edge case for CJK and giant contiguous blocks) covering the load-bearing seams. As shipped: ~260 lines across `tests/conftest.py` and three test files.

1. **Chunker sanity** — short text yields one chunk; long text yields contiguous-ordinal chunks under `MAX_TOKENS` with observable overlap (the start of `chunks[i+1]` appears inside `chunks[i]`); empty text yields none; no-separator text (CJK without inter-word spaces, or a giant contiguous Latin block) still respects the token cap — the case the old custom chunker needed a `_token_slice` fallback for.
2. **Upload → search smoke** — POST /text, then GET /search; the just-uploaded document is at the top of the results.
3. **Citation-source consistency** — POST /chat, assert every citation in `answer_blocks[*].citations` resolves to a `chunk_id` present in `sources`, and each `cited_text` is a verbatim substring of the corresponding source chunk's `text`. (Updated from an earlier `[N]`-style sketch when §8 moved to Anthropic's structured citations.) Empty-citations responses are gated on `body["answer"] == REFUSAL_TEXT` — anything else is a real regression worth catching.

**Why this and not more:**
- The brief explicitly says they're not evaluating production maturity (CI/CD, full test coverage).
- But code quality is a grading criterion, and zero tests is a weaker signal than a few well-chosen ones.
- Three tests demonstrate the discipline without pretending to be a full suite.

**Test-design choices worth recording:**
- **Real upstreams, not mocks.** Tests hit real Postgres (per docker-compose) and real Voyage / Anthropic. Mock/prod divergence would mask exactly the kinds of integration regressions a smoke suite exists to catch. Tests skip cleanly via `@requires_voyage` / `@requires_anthropic` when an upstream key is unset.
- **Per-test isolation via snapshot/diff.** `tests/conftest.py`'s `db_cleanup` fixture snapshots `documents.id` pre-test and deletes any new rows post-test (chunks cascade via FK). Reuses the pool already opened by `TestClient`'s lifespan.
- **uuid-salted upload payloads.** Each `POST /text` text body is prefixed with `[run {uuid4()}]` so the SHA-256 is unique per run. Without the salt, a crashed prior run that left rows in place would short-circuit subsequent runs through `find_existing_by_hash` (§12) and silently bypass the chunk→embed→persist path the test claims to exercise.

---

## 14. API key auth (stretch goal)

**Decision:** Skipped — see §15. The brief lists auth as a relevance-gated stretch goal ("if you find them relevant to the task — optional"), and the threat model is empty for a local-only single-tenant demo. The sketch below is retained as the shape that would ship if relevance changed.

**Sketch (not implemented):**
- Single bearer token compared against env var `API_KEY` using `secrets.compare_digest`.
- Header: `Authorization: Bearer <key>`. Implemented via FastAPI's built-in `HTTPBearer(auto_error=False)` so the dependency owns the response shape (401 + `WWW-Authenticate: Bearer`) rather than letting FastAPI emit a 403.
- Wired at router scope on the four data-touching routers; `/health` exempt for liveness probes.
- ~half-day to ship including the empty-key bypass guard (`compare_digest("", "")` is `True`).

---

## 15. Deliberate cuts (and why)

Each cut is filed as a GitHub Issue and surfaced in the README's "What I left out and why" section. The point is to make the cuts *legible* — not hidden by omission.

| Cut | Why deferred | Effort to add later | Issue |
|---|---|---|---|
| **Multi-turn /chat with query rewriting** | Real complexity is in the query rewriter, not the API. Brief asks for single-turn. | ~1 day | [#25](https://github.com/nainajnahO/rag-knowledge-base/issues/25) |
| **Structure-aware PDF chunking via docling** | Chose plain-text token-based chunking as the baseline; structure-aware is the next quality lever, not the baseline. | ~half-day | [#26](https://github.com/nainajnahO/rag-knowledge-base/issues/26) |
| **Parent-child / hierarchical chunking** | Real retrieval-quality win, but complicates retrieval and chat. Architecture is designed to make it cheap. | ~1 day | [#27](https://github.com/nainajnahO/rag-knowledge-base/issues/27) |
| **voyage-context-3 contextual embeddings** | Same dimension as voyage-4 → cheap migration. Better story to ship the standard voyage-4 model first; document-grouped batching adds complexity worth deferring. | ~half-day | [#28](https://github.com/nainajnahO/rag-knowledge-base/issues/28) |
| **Streaming responses on /chat** | Brief lists it as stretch. Demo is via curl; streaming complicates citation parsing for the reviewer. | ~few hours | _not filed — defer beyond §15 issue set_ |
| **API key auth** | Brief lists it as relevance-gated stretch ("if you find them relevant"). Threat model is empty for a local-only single-tenant demo: no network exposure, no multi-user model, and upstream costs are already gated by `VOYAGE_API_KEY` / `ANTHROPIC_API_KEY` (server-side, operator-controlled). Adding a key would clutter every README curl example without protecting anything in scope. See §14 sketch for the shape if relevance changed. | ~half-day | [#29](https://github.com/nainajnahO/rag-knowledge-base/issues/29) |
| **Knowledge graph (entities + relations)** | ~~Genuinely interesting, big rabbit hole.~~ **Implemented** in [#30](https://github.com/nainajnahO/rag-knowledge-base/issues/30) — see §18. | _shipped_ | [#30](https://github.com/nainajnahO/rag-knowledge-base/issues/30) |
| **Per-document language tracking for multilingual lexical ranking** | Hybrid uses `'simple'` tsv config — covers any language adequately but skips per-language morphology and stop-word lift. Real per-language lexical morphology needs per-doc language detection. See §7.1. | ~half-day | [#3](https://github.com/nainajnahO/rag-knowledge-base/issues/3) |
| **Production logging/metrics/CI** | Brief explicitly excludes these. | n/a | _not filed — out of scope_ |

---

## 16. Build sequence

PRs are sized to be individually reviewable. Each merges to `main` with a merge commit (not squash) so individual commits survive.

> **Numbering.** The steps below are **build-sequence ordinals**, not GitHub PR numbers. Quality follow-ups (e.g., chunker fallbacks, dependency-injection refactors) get their own GitHub PRs interleaved with feature PRs, so GitHub `#N` and "Step N" in this list are not the same N. Use `gh pr list --state merged` to see the actual PR-by-PR landing order.

1. **Step 1** — Scaffold: `.gitignore`, `pyproject.toml` (uv, Python 3.14), `docker-compose.yml`, `app/main.py` with `/health`, `.env.example`, README skeleton.
2. **Step 2** — Schema: `sql/schema.sql` with `documents` and `chunks` tables, pgvector extension, HNSW vector index, and GIN index on JSONB metadata. Auto-applied on first container init via `/docker-entrypoint-initdb.d/`. Lexical-ranking column (`tsv` + GIN) deferred to Step 7.
3. **Step 3** — `POST /text` end-to-end: chunker module, Voyage embedder module, persistence.
4. **Step 4** — `POST /document`: pymupdf extraction + content-hash dedupe.
5. **Step 5** — `GET /search`: vector similarity with metadata filters via JOIN. **Done.** k=10/max=50, no score threshold, AND-across-keys metadata filters, HNSW iterative scan with `strict_order` (§6.1), shared `RetrievedChunk` model with `/chat`. See PR for the full set of locked design choices.
6. **Step 6** — `POST /chat`: retrieval + Sonnet 4.6 + structured citations + four guardrails. **Done.** Path B chosen during planning research after surfacing Anthropic's first-class Search Result content blocks (§8 rewrite). Response carries `answer` (display), `answer_blocks` (native Anthropic shape), and `sources` (all retrieved chunks with `cited` flag and verbatim `cited_text`). See PR for the full set of locked design choices.
7. **Step 7** — Hybrid search + rerank: tsvector column + RRF query, plus Voyage `rerank-2.5` over the 50-candidate fused pool. **Done.** Replaced dense `/search` and `/chat` retrieval with hybrid+rerank. Dropped the cosine-threshold gate from `/chat` per §8 guardrail #3 — refusal is now structural (empty-retrieval) plus model-driven (system prompt). Schema, retrieval SQL, and `app/rerank.py` shape verified against pgvector and Supabase reference examples; rerank choice and the absent-threshold pattern verified against Anthropic's *Introducing Contextual Retrieval*.
8. **Step 8** — API key auth dependency. **Skipped** — see §15. Threat model is empty for a local-only single-tenant demo; the brief gates this stretch goal on relevance and there's nothing in scope for a key to defend.
9. **Step 9** — Smoke tests (pytest). **Done.** Three smoke tests covering chunker invariants, upload→search retrieval, and citation/source consistency. Real upstreams; uuid-salted payloads to keep the chunk→embed→persist path always exercised; per-test snapshot/diff cleanup. See §13.
10. **Step 10** — README polish, sample documents demonstrating metadata filtering, curl/Postman collection, opening "deliberate cuts" issues. **In flight.** §15 issues filed ([#25](https://github.com/nainajnahO/rag-knowledge-base/issues/25)–[#30](https://github.com/nainajnahO/rag-knowledge-base/issues/30) plus pre-existing [#3](https://github.com/nainajnahO/rag-knowledge-base/issues/3)). README polished in this same step. Sample documents + Postman collection still pending.
11. **Step 11** — Knowledge graph: per-chunk Haiku entity extraction + per-document-per-type Sonnet resolution + entity-aware retrieval. **Done.** Closes [#30](https://github.com/nainajnahO/rag-knowledge-base/issues/30); see §18 for the load-bearing decisions.

---

## 17. Upload size limits

**Decision:** Two-layer cap on uploaded content:

- **25 MB file-size cap** on the multipart body of `POST /document`. Returns **413 Payload Too Large**, rejected at the door before the body bytes are read into memory.
- **3,000,000 character cap on extracted text** on both `POST /text` and `POST /document`. Returns **422 Unprocessable Entity** via Pydantic validation on `/text`, and via a page-by-page running-total check inside `app/extraction.py` (raises `TextTooLargeError`, mapped to 422 by the route) on `/document`.

**Alternatives considered:**

- **No caps at all.** Simpler but no defense against either RAM blow-up (huge file read into memory before extraction) or runaway embedding cost (huge text → thousands of chunks → many Voyage calls).
- **File-size cap only.** Catches huge uploads cheaply at the door, but a text-heavy 25 MB PDF can extract to ~20 MB of plain text → ~10,000 chunks at 600 tokens/chunk → real Voyage cost + minutes of latency per upload.
- **Text-length cap only.** Catches runaway content cost, but a 500 MB PDF still gets fully read into RAM *before* extraction tells us to reject it. The text-length cap fires too late to defend memory.
- **Tighter file cap (~5 MB) sized to make the text cap unnecessary.** Math: 3M chars of text-heavy PDF ≈ 5 MB file. Aggressive — would reject many legitimate academic papers (image-heavy academic PDFs are routinely 5–10 MB).

**Why two layers:**

- **File-size cap (25 MB)** protects RAM at the door. PyMuPDF reads the whole file into memory before extraction; without a door check, a single malicious upload could exhaust the worker. 25 MB covers any normal document (memos ≤ 100 KB, reports 1–2 MB, image-heavy academic papers 5–10 MB). The per-request memory ceiling is **input + extracted-text buffer**: 25 MB for the file bytes plus up to ~2× `MAX_TEXT_CHARS` for the in-flight `parts` list and joined string (≈ 24 MB worst case in Python), bounded because `extract_text` aborts page-by-page once the running total crosses `MAX_TEXT_CHARS` (otherwise compressed text streams could expand far beyond the 25 MB input). Multiply by uvicorn's effective concurrency (Starlette's threadpool size, default 40) to get the worker-level ceiling. (DB pool size doesn't gate this; body parsing happens before the conn dep blocks on a pool slot.)
- **Text-length cap (3M chars)** protects against runaway chunking + embedding cost *after* extraction. At ~4 chars/token, 3M chars ≈ 750K tokens ≈ ~1,400 chunks at 600 tokens/chunk with 15% overlap. Without this cap, a text-heavy 25 MB PDF could produce ~10,000 chunks per upload — real money in Voyage embeddings + minutes of latency.
- **Symmetry between `/text` and `/document`.** Both endpoints feed the same chunking + embedding + persistence pipeline, so they share the same content limit. A user can't bypass `/text`'s 3M-char cap by repackaging the same content as a PDF.

**Why these specific numbers:**

- **25 MB file cap.** Typical memo: 100 KB. Typical report: 1–2 MB. Image-heavy academic paper: 5–10 MB. Slide decks with embedded raster images: 10–20 MB. 25 MB covers all of these comfortably while still bounding worst-case memory at a manageable ceiling. Round number, easy to remember, easy to widen later if a real document hits the cap.
- **3M character cap.** ~750K tokens of text → ~1,400 chunks. The largest legitimate single-upload artifact a reviewer would plausibly send is a textbook chapter at ~50K characters; 3M gives 60× headroom while still bounding the worst case at ~$0.15 in Voyage cost per upload (vs. ~$1+ unbounded).

**Implementation:**

- **File-size cap.** Enforced via FastAPI middleware that reads the `Content-Length` header before the multipart body is parsed. Reject early with 413 — never reads the body bytes. (For requests without `Content-Length`, fall back to a streaming-byte counter that aborts at 25 MB.) Middleware is **global** (applies to every endpoint), not path-scoped to `/document`. Free for the other endpoints — `/text` is already bounded by the 3M-char cap (≤ ~12 MB UTF-8 worst case) and `/chat` request bodies are tiny — and avoids a per-path branch in the middleware.
- **Text-length cap.** `/text` enforces it via the reusable type alias `MaxText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]` in `app/limits.py` (`IngestTextRequest.text: MaxText` replaces today's inline `Field(max_length=3_000_000)`). `/document` enforces the same constant page-by-page inside `app/extraction.py`: `extract_text` walks pages, tracks a running character total (including the 2-char `"\n\n"` joins), and raises `TextTooLargeError` the moment it crosses `MAX_TEXT_CHARS`. The route catches that and returns 422 — same status code, same constant, no shared validator. Two enforcement sites for two genuinely different shapes (Pydantic-validated request body vs. a streaming extraction loop), one shared constant. Empty extraction is caught **after** the text-length cap (it can't fire if extraction was capped first) and raised as **400** (mirroring `/text`'s post-strip check at `app/routes/text.py:20-22`), so §11's 400 entry remains authoritative for empty-text errors.
- **Error shape.** Both errors use FastAPI's default `{"detail": "..."}` per §11 — no wrapping, no custom hierarchy.
- **Where the constants live.** `MAX_UPLOAD_BYTES = 25 * 1024 * 1024`, `MAX_TEXT_CHARS = 3_000_000`, and the `MaxText` type alias all live as module-level definitions in `app/limits.py`. Middleware imports `MAX_UPLOAD_BYTES`; `IngestTextRequest` imports `MaxText`; `app/extraction.py` imports `MAX_TEXT_CHARS` directly for its page-by-page running-total check. Not surfaced through `Settings` / env vars — these are tied to embedding economics and worker memory, not per-deployment knobs.

**Migration notes:** None — these are operational constants. If a real customer uploads a document that hits either cap, the cap moves (one edit in `app/limits.py`); the *structure* (file cap + text cap, two layers) is the load-bearing decision.

---

## 18. Knowledge graph (closes #30)

**Decision:** Add a knowledge graph layer on top of the existing hybrid retrieval. Entities (PERSON, ORGANIZATION, LOCATION, EVENT) and relations (typed edges with edge provenance back to a chunk) are extracted per-chunk via Haiku and resolved per-document-per-type via Sonnet, both at ingest time. `GET /search` accepts repeated `?entity=X&entity=Y` params with chunk-level AND co-mention semantics; `POST /chat` auto-extracts entities from the question and uses the same filter, with a fallback to plain retrieval if the filter narrows to zero candidates. Relations are stored but not yet queryable — the v1 surface is entity co-mention, which is what the brief's example asks for.

The implementation closes [issue #30](https://github.com/nainajnahO/rag-knowledge-base/issues/30) ("Defer: knowledge graph (entities + relations)"). Brief stretch goal wording: *"extract entities (people, organizations, places) from the documents and store relationships (e.g., Neo4j, Memgraph, or a relations table). Show how the graph can improve retrieval (e.g., 'all documents that mention X together with Y')."*

### 18.1 Why native Anthropic Structured Outputs (and which model where)

Extraction and resolution both use `client.beta.messages.parse(...)` (Anthropic's Structured Outputs) with Pydantic schemas — the exact path Anthropic's *Knowledge Graph Construction with Claude* cookbook documents (March 2026 release, `platform.claude.com/cookbook/capabilities-knowledge-graph-guide`). No hand-rolled JSON parsing, no tool-use round-trip; the SDK validates parsed output against the same Pydantic models the rest of the codebase uses.

**Model split** matches the cookbook:

- **Extraction** uses `claude-haiku-4-5` — high-volume schema-constrained work where speed and cost dominate (one call per chunk, ~30 chunks per typical multi-page doc).
- **Resolution** uses `claude-sonnet-4-6` — arbitrating conflicting evidence across surface forms (is "Acme" the company, the cartoon brand, or the holding company?). Sonnet's better at the "weigh descriptions, decide what merges" call than Haiku is.

**SDK note on `output_format` deprecation.** As of the structured-outputs beta (`structured-outputs-2025-12-15`), the SDK emits a `DeprecationWarning` for the `output_format=Pydantic` keyword in favor of `output_config={"format": ...}`. We keep `output_format` because the deprecated parameter is what populates `ParsedBetaMessage.parsed_output` end-to-end (the SDK's `parse()` method auto-runs the API request *and* parses the response into the Pydantic type); the new `output_config.format` API requires manual JSON-schema generation via private SDK helpers (`transform_schema`) and skips the parse step. The cookbook still uses the deprecated kwarg verbatim. Worth migrating once the SDK exposes a public path that preserves auto-parse.

### 18.2 Why per-chunk extraction (the cookbook's outlier choice)

The cookbook's example extracts per-document. Industry consensus disagrees — every production graph-RAG framework I surveyed extracts per chunk:

| Framework | Granularity | How it handles cross-chunk relations |
|---|---|---|
| Microsoft GraphRAG | Per text-unit (chunk) | Merges entities with same title+type across chunks; configurable chunk overlap |
| LightRAG | Per chunk (default 4k tokens) | Chunk overlap (128 tokens) + "gleaning" retry pass |
| LlamaIndex `PropertyGraphIndex` | Per chunk node | Each chunk becomes a graph node; `MENTIONS` edge from chunk → entity |
| Neo4j `LLMGraphTransformer` | Per chunk (combinable via `NUMBER_OF_CHUNKS_TO_COMBINE`) | `HAS_ENTITY` edge from chunk → entity |
| Anthropic cookbook | Per **document** | Designed for short Wikipedia summaries (~1.5k tokens — ~2× our chunk size) |

The cookbook gets away with per-document because its example is short atomic Wikipedia summaries. Real customer content (multi-page PDFs, long memos) would either overflow a single Haiku call or suffer long-context attention drift on the back half of the document. Per-chunk is the industry's answer; chunk overlap (we already ship 15%) plus the resolution pass we already chose handle the "relations split across chunks" mitigation.

The trade is more LLM calls per ingest. Mitigated by **prompt caching** on the extraction system block: the fixed prompt + schema is reused across every per-chunk call in a document, so chunk text is the only varying input. At ~30 chunks per doc, this turns N expensive calls into one expensive + (N-1) cheap.

### 18.3 Why per-document-per-type resolution (not per-chunk-per-type)

If the cookbook's pure incremental pattern were applied at chunk granularity, resolution would issue one Sonnet call per chunk per type — at 30 chunks × 4 types that's 120 Sonnet calls per ingest. Per-document-per-type collapses to ≤4 Sonnet calls per ingest (one for each `EntityType` that has at least one entity in this document).

Implementation: collect every `(name, type, description)` across the document's chunks, dedupe by `(name, type)`, pull the existing canonical set per type from the DB (capped at `_MAX_EXISTING_CANONICALS = 200` to bound prompt growth), and feed that combined list to one `messages.parse()` call. The cookbook's existing-canonicals-as-anchors trick keeps previously resolved entities from being re-split between ingests.

### 18.4 Why sync at ingest, not async/Batch API

Anthropic's Batch API (50% off, ≤24 h SLA) is the obvious cost lever for a use case like this. We don't apply it because we deliberately picked **synchronous extraction at ingest** — the user uploads, gets a `document_id` back, and the graph for that document is queryable on the next request. Async would require either a `graph_status` column + polling endpoint or a background-worker fan-out; both add ops surface beyond what the brief asks for ("we're not evaluating production maturity").

The cost: ingest now blocks for ~1-2 seconds (per-chunk Haiku calls run in a thread pool capped at 10 concurrent — see `app/graph_extract.py` — plus ≤4 Sonnet calls for resolution, all serial after the parallel extraction stage). For a demo + work-sample corpus this is acceptable; documented as a known trade-off and surfaced in the README so reviewers don't think it's accidental.

### 18.5 Why Postgres relations tables, not Neo4j / Memgraph / Apache AGE

The brief offers three storage choices: *"Neo4j, Memgraph, or a relations table"* — the third option is explicitly endorsed. We pick it because:

1. **One store**, per §1's stack-minimization principle. Neo4j would mean a second container, a second connection pool, second authentication surface, and either denormalized entity IDs across two stores or a sync job between them.
2. **One-hop traversal is enough.** The brief's example query — *"all documents that mention X together with Y"* — is a one-hop set intersection, not a Cypher pattern match. SQL handles it cleanly via `EXISTS + GROUP BY + HAVING COUNT(DISTINCT)`. Cypher pays for itself when you're walking 3+ hops or pattern-matching cycles; we're not.
3. **Clean upgrade path if needed later.** The Postgres schema (`entities`, `entity_aliases`, `chunk_entity_mentions`, `relations`) maps 1:1 to a Neo4j property graph — there's nothing to throw away if a real customer surfaces a deep-traversal use case.

Apache AGE (Postgres extension that adds Cypher) was the runner-up. Rejected for v1 because it requires modifying the Docker image and Cypher buys nothing for the brief's example query.

### 18.6 Entity types: PERSON, ORGANIZATION, LOCATION, EVENT (no ARTIFACT)

The brief names PERSON, ORGANIZATION, LOCATION explicitly. We add EVENT — memo and report content is heavily time-anchored, so *"Q3 2025 earnings call"* and *"Acme acquisition of Antwerp Hub"* become first-class queryable nodes rather than just substrings inside chunks.

We deliberately omit the cookbook's fifth type, ARTIFACT (products, software, regulations, file names, vehicles). It would have helped — but the cost-benefit doesn't pencil out for this corpus:

1. **Self-references.** A memo titled *"Q3 2025 Strategy Update.pdf"* would extract itself as an ARTIFACT. Now every chunk in the memo "mentions" the memo itself, and `?entity=Q3 2025 Strategy Update` returns the memo's own chunks back to itself — circular and useless. One bogus self-entity per ingested document.
2. **Weak within-type coherence breaks resolution.** The Sonnet resolver clusters surface forms within a type. PERSON works because two "Tim Cook" mentions almost certainly refer to the same human. ARTIFACT bundles unrelated kinds of things (iPhone 17 + GDPR + Apollo 11 + report.pdf) — the resolver either over-merges entities that share words or under-merges entities under different surface forms ("the report" vs. "Q3 Memo").
3. **Demo muddying.** The brief's *"X together with Y"* assumes meaningful named entities. ARTIFACT noise would seed the entity list with semantically thin items — exactly the wrong impression for evaluation criterion #1 ("sensible choices … no obvious holes").

Mitigations exist (a stop-list, post-filter on document-title overlap, a constrained ARTIFACT-extraction prompt) but each is hand-rolled remediation work *because* we took ARTIFACT. Cleaner not to.

### 18.7 Chunk-level AND co-mention as the `/search` default

`/search?entity=X&entity=Y` returns only chunks where every listed entity is mentioned in **the same chunk**, not just the same document. SQL gate inside both the vec and lex CTEs:

```sql
AND (%(n_entities)s = 0 OR EXISTS (
        SELECT 1 FROM chunk_entity_mentions cem
        WHERE cem.chunk_id = c.id
          AND cem.entity_id = ANY(%(entity_ids)s::uuid[])
        GROUP BY cem.chunk_id
        HAVING COUNT(DISTINCT cem.entity_id) = %(n_entities)s
   ))
```

Why chunk-level over document-level (the brief's literal wording):

- **Stronger signal for citations.** A chunk that itself names both X and Y is the strongest evidence the LLM can cite from. Document-level co-mention can return a chunk that names neither, leaving citation grounding on parts of the document Claude never sees.
- **Document-level is a trivial rollup** if we ever expose it: `SELECT DISTINCT document_id FROM …`. Going the other direction (deriving chunk-level from a doc-level store) would require re-running extraction.
- **The `%(n_entities)s = 0` short-circuit** preserves identical no-filter behaviour. Pre-graph queries are byte-identical post-graph when no `?entity=` is supplied.

### 18.8 Relations stored, not queryable in v1

The brief asks to *"store relationships,"* and we do — every relation row carries `source_entity_id`, `predicate`, `target_entity_id`, plus `source_chunk_id` and `source_document_id` for **edge provenance** (each extracted relation can be cited back to the chunk that grounded it).

We stop short of exposing `?relation=Acme:supplied-by:Antwerp` on `/search` for v1. That surface needs its own design: a query parser, a prompt to extract a relation from the chat question (not just entities), a different SQL shape, and tests for the relation-grounding round-trip. It's a follow-up PR — the v1 demo is co-mention, which is what the brief's example asks for.

The provenance columns are a ~1-hour bonus that costs nothing now and unblocks two future things: (a) a graph-traversal answer endpoint that can cite edges via the existing search_result blocks, and (b) a debugging surface to trace any incorrect relation back to the chunk text that produced it.

### 18.9 `/chat` auto-extracts entities and falls back if the filter empties

The bridge from a free-text question to entity-aware retrieval is `extract_entities_from_question(question) → list[str]` — a small `messages.parse()` call against the same `ExtractedGraph` schema, with a wide `try/except` that returns `[]` on any failure. The handler resolves names to UUIDs, threads them into `Filters`, and **retries retrieval without the filter if the filter narrowed candidates to zero**:

```python
candidates = retrieve(conn, q, k, Filters(entity_ids=resolved))
if not candidates and resolved:
    candidates = retrieve(conn, q, k, Filters())
```

**Coverage caveat.** The pre-filter activates only when the question's surface forms case-insensitively match an alias `persist_graph` wrote — and aliases are limited to what some chunk's Haiku call literally extracted. A question like *"What did Acme do?"* against a corpus where every document says *"Acme Corporation"* produces no alias hit, `entity_ids = []`, and unfiltered retrieval — the graph contribution is silent in that case. The pre-filter is most useful when users phrase questions with the same surface forms the documents use; it degrades to plain hybrid retrieval otherwise. The fallback below covers the *different* failure mode (alias hit, but no chunk co-mentions all named entities).

Without that fallback, the graph could *hide* otherwise-relevant content: a question naming entities that *are* in the corpus but never co-mentioned in a single chunk would dead-end at zero candidates and refuse incorrectly. The fallback ensures the graph only adds precision when the corpus has the relevant entities co-mentioned; it never blocks an answer.

The `/chat` answer pipeline downstream of `retrieve()` is **unchanged** — same rerank, same `search_result` content blocks, same `citations: enabled`. Citations remain at chunk granularity. The graph adds a pre-filter; it doesn't replace any part of the citation machinery (per the user's standing preference for native Anthropic shapes).

### 18.10 Schema details

Four new tables, all in `sql/schema.sql` (no migration system; reset workflow is `docker compose down -v && docker compose up -d`):

- **`entities`** — `(canonical_name, entity_type)` is `UNIQUE`. Entities are corpus-global (not document-scoped) so a canonical entity persists even after every document mentioning it is deleted. `description` is updated on conflict so it stays fresh as more docs mention the same canonical.
- **`entity_aliases`** — `(entity_id, alias)` `UNIQUE`. `idx_entity_aliases_lower` on `LOWER(alias)` powers case-insensitive `?entity=` resolution at query time.
- **`chunk_entity_mentions`** — composite PK `(chunk_id, entity_id)`. Both directions indexed (`idx_cem_chunk_id`, `idx_cem_entity_id`) so the EXISTS+HAVING gate runs cheaply.
- **`relations`** — every row carries both `source_chunk_id` and `source_document_id` (edge provenance per §18.8). Cascading FKs on chunks, documents, and entities mean a `DELETE FROM documents` cleans the graph transitively without leaving dangling rows.

### 18.11 Known limitations (and their mitigations / migration paths)

- **Resolution prompt cap (`_MAX_EXISTING_CANONICALS = 200`).** As the corpus grows the canonical set per type can exceed 200. Past that we feed the most recent 200 by `created_at`; older canonicals beyond the cap can be re-split if the resolver thinks the new entity is distinct. v2 path: embedding-shortlist the existing canonicals against the new entity before sending to Sonnet.
- **Same-name across types.** "Apple" the ORGANIZATION and "Apple" the LOCATION are two separate `entities` rows (the `UNIQUE` constraint is on the pair). `resolve_entity_names` returns *all* matching entity IDs across types — `parse_entity_filter` then groups them so the retrieval SQL counts distinct *user-passed names* per chunk, with OR-within each name's resolved entity_ids. Net result: `?entity=Apple` returns chunks mentioning Apple in any capacity, even when the same canonical was extracted as different types in different chunks. If type-scoped filtering is needed later, add `?entity_type=ORGANIZATION` — drop-in column filter on `entity_aliases JOIN entities`.
- **Cross-test entity pollution.** The `db_cleanup` fixture deletes documents but not entities (they're corpus-global). Smoke tests use uuid-salted entity names (`Zorbnak-<8>`, `Belgrade-<8>`) so test runs don't collide; assertions check *absence* of the wrong document rather than position-1 of the right one.
- **Resolution mis-merges across genuinely distinct entities.** The cookbook's prompt feeds Sonnet the entity description as a disambiguation signal, but ambiguous descriptions ("Apollo 11" the spacecraft vs. the documentary) can still merge incorrectly. A worsening graph quality, not an integrity issue — duplicate canonicals have no DB-level consequence.
- **Concurrent-ingest race on canonical naming.** `_fetch_existing_canonicals` reads the canonical set before `insert_document_with_chunks` opens its transaction, so two simultaneous ingests can each call Sonnet against the same pre-state. If both extract "Acme" and Sonnet picks "Acme Corporation" for one ingest and "Acme Corp" for the other, both rows commit (the `UNIQUE` constraint is on the literal canonical_name, not on the underlying real-world entity). For the synchronous-curl demo this doesn't fire; under concurrent traffic it would seed the duplicates the resolver is otherwise designed to prevent. Mitigation paths: (a) advisory lock around resolve+persist, (b) move resolution inside the transaction at the cost of serialised ingest, or (c) the embedding-shortlist v2 path above (which would catch the duplicate at persist time even without locking).

### 18.12 Effort vs. the original 2-3 day estimate

Filed in §15 at 2-3 days. Shipped in 12 small commits on `knowledge-graph-entity-extraction`. What got cut against the cookbook's full pipeline:

- No stage-3 entity profile summarization (cookbook's hub-node summaries) — brief doesn't ask for it
- No `?relation=A:p:B` query surface — see §18.8
- No GET /entities or GET /entities/{id}/relations inspection endpoints — relations are present in the DB and inspectable via psql
- No ARTIFACT type — see §18.6
- No embedding-based shortlisting for resolution — see §18.11
- No async / Batch API — see §18.4

Each of those is a clear future PR; the v1 ships the brief's stretch goal cleanly without dragging them in.

### 18.13 Demonstrating the improvement: eval harness shape

The brief's third graph ask is "show how the graph can improve retrieval." The `evals/` harness is the proof artifact: a runtime opt-out flag (`use_graph` on `ChatRequest`) plus a small corpus + 7 golden questions that run through `/chat` in both modes and produce a markdown comparison table.

**Why a runtime flag, not separate endpoints.** `/chat` auto-extracts entities and applies the filter unconditionally per §18.9, so a baseline comparison on the same handler needs an opt-out. `?use_graph=false` keeps the comparison on a single endpoint with one observable contract — no shadow `/chat-baseline` route to drift, no dead code path. `/search` already supports baseline comparison out of the box (omit `?entity=`), so the flag is `/chat`-only.

**Why pass/fail per question, not LLM-as-judge.** At 7 questions the cost-vs-signal of an LLM judge is unfavourable: an extra 7+ Sonnet calls per run for noisier numbers. The pass criterion is two deterministic gates — every expected substring in the answer (case-insensitive) and every expected document title in the response sources — so the same run produces the same table. Strict on retrieval (the chunk must have made it into context), lenient on phrasing (the LLM can paraphrase).

**Why two metrics, not one.** Pass rate measures *correctness* — did the pipeline answer the question. Avg distinct sources measures *precision* — how many distinct documents the answer was grounded on. The graph's effect is on precision (it narrows the candidate pool); at small corpus scales hybrid retrieval + rerank deliver correct answers in both modes, so a correctness-only metric reads 7/7 vs 7/7 and looks like no improvement. The source-count metric exposes what the graph actually does — typical run shows 3.0 avg sources graph-on vs 5.0 graph-off, with co-mention questions hitting 1 vs 5 (a clean 5× narrowing where the graph filter activates).

**Why /chat the surface, not /search.** The brief frames the deliverable as "chat with knowledge base"; `/chat` is the user-visible surface where the graph earns or doesn't earn its keep. Driving the eval through `/chat` exercises the full path — auto-extraction, entity resolution, filter, retrieval, rerank, citation — not just the SQL.

**Why a 15-doc handcrafted corpus (5 target + 10 distractor), not the work-sample PDF.** The corpus is sized deliberately above `CHAT_TOP_K = 8` so baseline retrieval has to choose 8 of 15 rather than fit-everything. The 5 target docs carry the answers; the 10 distractors share entity surface forms (`Quintara`, `Garcia`, `Berlin`, `Tokyo`, `Tanaka`, `TechSummit`) on different topics so the entity AND filter has real lexical/semantic competition to narrow against. Co-mention questions still resolve to a single correct document (verified by hand against question expectations). A real-world corpus would require hand-curating expected document IDs per question on content the eval author didn't choose — doesn't scale at work-sample effort.

**Why corpus > top_k matters.** An earlier draft of the eval used 5 documents (1 chunk each, so corpus = 5 chunks ≤ top_k = 8). Baseline retrieval was *forced* to return all 5 because there was nothing to compete them against, so the graph filter narrowing from 5 → 1 was real but uninformative — it was "everything vs the right one," not "noisy candidate pool vs the right one." Sizing the corpus above top_k lets baseline actually choose, which is what the comparison should be measuring. The earlier 5-doc snapshot also masked two real bugs (entity-type-split breaking AND filter, and document-prompt under-extracting on short questions) — both fixed in this PR; both surfaced by expanding the corpus.

**Known limits this eval doesn't show.** It doesn't measure rung-2 capability (graph traversal across documents — see [§18.7](#187-chunk-level-and-co-mention-as-the-search-default), explicitly out of scope), doesn't stress-test resolver mis-merges (the corpus is constructed to avoid them), and doesn't handle non-determinism beyond noting "single-run snapshot" — re-running can shift the table by a row depending on extraction outcomes. These are documented as the trade-offs of a 2-3 day work-sample eval, not a benchmark suite.

**Scale limits the bug fixes don't reach.** The Bug A (SQL group-by-name) and Bug B (question-tuned prompt) fixes are corpus-size-agnostic — they correct root causes in the SQL semantics and the prompt, not symptoms at the 15-doc scale. They scale cleanly: the SQL handles N type-splits per name as readily as 3, and the question prompt operates on the question text not the corpus. What they *don't* touch are the separate corpus-scale limits already filed in [§18.11](#1811-known-limitations-and-their-mitigations--migration-paths):

- **`_MAX_EXISTING_CANONICALS = 200`** still caps the resolver's view of the canonical set per type. Past 200 canonicals, older ones beyond the cap can be re-split by Sonnet thinking the new entity is distinct. Embedding-based shortlisting is the v2 path.
- **Resolution mis-merges across genuinely distinct entities** ("Apollo 11" the spacecraft vs the documentary) are a separate quality issue from Bug A's same-canonical-different-type split — the cookbook's description-disambiguation helps but isn't perfect, and the bug fixes here don't change that.
- **Filter selectivity at very large corpora.** Graph narrowing is biggest when the AND filter cuts hardest (rare co-mentions — the brief's "X together with Y" shape). At very large scale, a single-entity filter on a common name returns most of the corpus, and the relative narrowing vs baseline shrinks. This is a characteristic of chunk-level co-mention as a primitive, not a bug — and the eval at 15 docs doesn't exercise it.
