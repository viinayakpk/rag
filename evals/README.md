# Eval harness — graph vs baseline

A small hand-crafted corpus + 7 golden questions to measure the
quantitative effect of the knowledge-graph layer on `/chat`. Each
question is run twice — once with `use_graph=true`, once with
`use_graph=false` — and the runner prints a pass/fail markdown table
that pastes verbatim into the main README's "Graph vs baseline" section.

## Run

```bash
docker compose up -d                  # postgres
uv run python -m evals.run_eval       # prints the markdown table
```

Requires `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` in env (same as the
test suite — see `tests/conftest.py` for the skip markers).

## Output shape

Each row reports two numbers per mode: pass/fail (correctness) and the
count of distinct documents that appeared in the response sources
(precision). Aggregate rows summarize both at the bottom.

```
| # | Question | Graph on | Graph off |
| 1 | ... | ✓ (1 src) | ✓ (5 src) |
| ... | ... | ✓ (N src) | ✓ (M src) |
| **Pass rate**         | | **N/7** | **M/7** |
| **Avg distinct sources** | | **N.N** | **M.M** |
```

Two distinct signals:

- **Pass rate** — answer-correctness gate: did the right document make
  it into context, did the answer contain the expected substring? This
  measures whether the pipeline can answer the question at all. At
  small corpus scales hybrid retrieval + rerank land the right chunks
  in both modes, so this number is often equal across modes.
- **Avg distinct sources** — precision: how many distinct documents
  was the answer grounded on? This captures the graph's actual
  narrowing effect on the candidate pool, which the pass criterion
  cannot see by construction. Lower is sharper.

The brief's "show how the graph can improve retrieval" lands here:
a smaller average source set with equal correctness is the rung-1
graph filter doing what it's designed for — narrowing precision on
chunks that mention the asked-about entities together.

## Pass criteria

Each question in `golden.py` declares two strict checks:

- `expected_substrings` — case-insensitive substrings the answer text
  must contain. Strict on retrieval (the chunk must have made it into
  context) but lenient on phrasing (the LLM can paraphrase).
- `expected_document_titles` — titles that must appear in the
  response's `sources` array. This is a retrieval-correctness check,
  independent of the answer.

A question passes only when **both** checks pass. There's no
LLM-as-judge — the criteria are deterministic so the table reflects
retrieval quality, not judging variance.

## Notes

- LLM-driven extraction is non-deterministic, so the table is a
  single-run snapshot. Rerun to sample again.
- The eval ingests under `EVAL/`-prefixed titles and cleans up after
  itself on exit (also at start, defensively). Other ingested documents
  are not touched. Cleanup runs in a `finally` block so a crash mid-run
  still empties the harness's data.
- Entity rows are corpus-global per the KG design (DECISIONS §18.10)
  and are not deleted by cleanup. This is intentional and stable
  across reruns — the resolver finds and reuses existing canonicals
  via incremental resolution.
- Both modes go through the same retrieval/rerank/citation pipeline;
  the only difference is whether the entity pre-filter is applied.
