from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.settings import settings

pool: ConnectionPool | None = None


def open_pool() -> None:
    global pool
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=10,
        configure=register_vector,
        open=True,
    )


def close_pool() -> None:
    global pool
    if pool is not None:
        pool.close()
        pool = None


def get_conn() -> Iterator[Connection]:
    if pool is None:
        raise RuntimeError("connection pool not initialized; check FastAPI lifespan")
    with pool.connection() as conn:
        yield conn


# Reusable Annotated-style FastAPI dependency for route signatures.
# Routes import as `from app.db import ConnDep` and write `conn: ConnDep`.
ConnDep = Annotated[Connection, Depends(get_conn)]
