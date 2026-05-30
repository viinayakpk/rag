"""Test 1 of §13 — chunker sanity.

Asserts the structural invariants of `chunk_text`: ordinals are contiguous,
no chunk exceeds MAX_TOKENS, and short text produces a single chunk. Token
counting routes through Voyage's local HF tokenizer (not an API call), but
client construction reads VOYAGE_API_KEY — skip when unset.
"""

from app.chunking import MAX_TOKENS, chunk_text

from .conftest import requires_voyage


@requires_voyage
def test_short_text_produces_single_chunk() -> None:
    chunks = chunk_text("A short paragraph that fits well under the chunk budget.")
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].token_count <= MAX_TOKENS


@requires_voyage
def test_long_text_chunks_have_contiguous_ordinals_size_cap_and_overlap() -> None:
    # Varied paragraphs (each carrying its own index) so an overlap
    # assertion can't be satisfied trivially by identical content.
    paragraphs = [
        f"Paragraph {i} discusses topic-{i} in detail. "
        "Token-based splitting respects natural boundaries when possible "
        "while keeping chunk sizes predictable. The library walks Unicode "
        "graphemes and packs them to a token capacity. "
        for i in range(50)
    ]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text)

    assert len(chunks) > 1, "long text should produce multiple chunks"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.token_count <= MAX_TOKENS, (
            f"chunk {c.ordinal} has {c.token_count} tokens, exceeds cap {MAX_TOKENS}"
        )
        assert c.text.strip(), "chunks should not be empty after stripping"

    # `semantic-text-splitter` carries an OVERLAP_TOKENS-sized tail of
    # chunks[i] into chunks[i+1], so the start of chunks[i+1] must appear
    # inside chunks[i]. Check the first 40 chars (less than one paragraph) —
    # small enough to be a real overlap signal but large enough to avoid
    # trivial punctuation-only matches.
    for i in range(len(chunks) - 1):
        head = chunks[i + 1].text[:40]
        assert head in chunks[i].text, (
            f"chunk {i + 1} doesn't share an opening with the end of chunk {i}: "
            f"no overlap detected"
        )


@requires_voyage
def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("") == []


@requires_voyage
def test_no_separator_text_still_respects_token_cap() -> None:
    # CJK without inter-word spaces is the case the old custom chunker needed
    # a `_token_slice` fallback for: Latin separators (`\n\n`, `. `, ` `)
    # never apply, so a recursive splitter would hand a single oversize piece
    # to the embedder. `semantic-text-splitter` walks Unicode graphemes
    # natively, so the cap should hold without any fallback. Same expectation
    # for any contiguous block (e.g. one giant token-free Latin string).
    cjk = "日本語のテキストをチャンクに分割するテストです。" * 200
    cjk_chunks = chunk_text(cjk)
    assert len(cjk_chunks) > 1, "long CJK text should produce multiple chunks"
    for c in cjk_chunks:
        assert c.token_count <= MAX_TOKENS, (
            f"CJK chunk {c.ordinal} has {c.token_count} tokens, exceeds cap "
            f"{MAX_TOKENS}"
        )

    contiguous = "a" * 10000
    contig_chunks = chunk_text(contiguous)
    assert len(contig_chunks) > 1, "long contiguous block should split"
    for c in contig_chunks:
        assert c.token_count <= MAX_TOKENS, (
            f"contiguous chunk {c.ordinal} has {c.token_count} tokens, "
            f"exceeds cap {MAX_TOKENS}"
        )
