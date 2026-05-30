"""Run the graph-vs-baseline eval — ingest the golden corpus, run each
question through /chat with use_graph=True and use_graph=False, print a
markdown comparison table to stdout.

Usage:
    uv run python -m evals.run_eval

Requires Postgres up (`docker compose up -d`) and VOYAGE_API_KEY +
ANTHROPIC_API_KEY in env (same prerequisites as the test suite).

Uses fastapi.testclient.TestClient — same pattern as tests/conftest.py —
so no live uvicorn is required. Cleans up its own EVAL/-prefixed
documents at start, ingests the corpus, runs all questions in both
modes, prints the table, and leaves the EVAL/ data in place so you can
manually curl /chat afterwards. Re-running the eval cleans up the prior
run before re-ingesting.
"""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from app import db
from app.main import app
from evals.golden import CORPUS, QUESTIONS, Question

EVAL_TITLE_PREFIX = "EVAL/"


def cleanup_prior_eval_data() -> None:
    """Delete documents from a previous eval run (title prefix-based).

    Cascades to chunks, chunk_entity_mentions, and relations via FKs.
    Entity rows and aliases persist (corpus-global by design — DECISIONS
    §18.10) and are reused across reruns via incremental resolution.
    """
    assert db.pool is not None, "TestClient lifespan should have opened the pool"
    with db.pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM documents WHERE title LIKE %s",
            (f"{EVAL_TITLE_PREFIX}%",),
        )


def ingest_corpus(client: TestClient) -> None:
    for doc in CORPUS:
        r = client.post("/text", json={"title": doc["title"], "text": doc["text"]})
        r.raise_for_status()


def score(response_json: dict, question: Question) -> bool:
    """Pass when every expected substring appears in `answer`
    (case-insensitive) and every expected title appears in `sources`.
    """
    answer = response_json.get("answer", "").lower()
    source_titles = {s["document_title"] for s in response_json.get("sources", [])}

    for substring in question["expected_substrings"]:
        if substring.lower() not in answer:
            return False
    for title in question["expected_document_titles"]:
        if title not in source_titles:
            return False
    return True


def run_question(
    client: TestClient, question: Question, *, use_graph: bool
) -> tuple[bool, int]:
    """Returns (passed, distinct_source_count). Source count is the size of
    the candidate pool the answer was grounded on — it captures the graph's
    actual narrowing effect, which the pass criterion doesn't see at small
    corpus scales (hybrid + rerank still land the right chunk).
    """
    r = client.post(
        "/chat",
        json={"question": question["question"], "use_graph": use_graph},
    )
    if r.status_code != 200:
        return (False, 0)
    body = r.json()
    passed = score(body, question)
    n_sources = len({s["document_title"] for s in body.get("sources", [])})
    return (passed, n_sources)


def main() -> int:
    with TestClient(app) as client:
        cleanup_prior_eval_data()
        ingest_corpus(client)

        try:
            results: list[tuple[Question, tuple[bool, int], tuple[bool, int]]] = []
            for q in QUESTIONS:
                graph_on = run_question(client, q, use_graph=True)
                graph_off = run_question(client, q, use_graph=False)
                results.append((q, graph_on, graph_off))
        finally:
            # Clean up after too — leftover EVAL/ docs pollute the test suite
            # (test_entity_filter_unknown_name_returns_empty asserts an empty
            # search response and expects a clean DB). Idempotent reruns.
            cleanup_prior_eval_data()

    # Two metrics per mode: pass/fail (correctness) and distinct source
    # count (precision — the graph's actual narrowing effect).
    print("| # | Question | Graph on | Graph off |")
    print("|---|---|---|---|")
    for i, (q, on, off) in enumerate(results, 1):
        on_cell = f"{'✓' if on[0] else '✗'} ({on[1]} src)"
        off_cell = f"{'✓' if off[0] else '✗'} ({off[1]} src)"
        print(f"| {i} | {q['question']} | {on_cell} | {off_cell} |")

    on_pass = sum(1 for _, on, _ in results if on[0])
    off_pass = sum(1 for _, _, off in results if off[0])
    on_avg = sum(on[1] for _, on, _ in results) / len(results)
    off_avg = sum(off[1] for _, _, off in results) / len(results)
    n = len(results)
    print(f"| **Pass rate** | | **{on_pass}/{n}** | **{off_pass}/{n}** |")
    print(f"| **Avg distinct sources** | | **{on_avg:.1f}** | **{off_avg:.1f}** |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
