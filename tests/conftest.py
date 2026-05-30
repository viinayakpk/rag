"""Test fixtures for the smoke tests (DECISIONS.md §13).

Per §13 the goal is "three tests demonstrate the discipline without pretending
to be a full suite." These fixtures hit the real dev Postgres (per
docker-compose) and the real Voyage / Anthropic upstreams when keys are set;
tests that need an upstream are skipped if the matching key is empty.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.settings import settings

requires_voyage = pytest.mark.skipif(
    not settings.voyage_api_key,
    reason="VOYAGE_API_KEY not set",
)
requires_anthropic = pytest.mark.skipif(
    not settings.anthropic_api_key,
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """FastAPI TestClient with lifespan enabled — opens/closes the DB pool."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_cleanup(client: TestClient) -> Iterator[None]:
    """Snapshot document IDs pre-test, delete any new ones after.

    Chunks cascade via the FK on documents. Uses the pool already opened by
    TestClient's lifespan.
    """
    assert db.pool is not None, "TestClient lifespan should have opened the pool"

    with db.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM documents")
        before = {row[0] for row in cur.fetchall()}

    yield

    with db.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM documents")
        after = {row[0] for row in cur.fetchall()}
        new_ids = list(after - before)
        if new_ids:
            cur.execute("DELETE FROM documents WHERE id = ANY(%s)", (new_ids,))
