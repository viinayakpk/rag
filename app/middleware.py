"""Global body-size middleware — enforces DECISIONS.md §17's 25 MB cap.

Applies to every endpoint, not path-scoped to /document. /text is bounded by
the 3M-char text cap, /chat request bodies are tiny — the global cap costs
nothing on those routes and avoids per-path branching here.
"""

import json

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.limits import MAX_UPLOAD_BYTES


class MaxBodySizeMiddleware:
    """Reject requests whose body exceeds MAX_UPLOAD_BYTES with 413.

    Cheap path: trust Content-Length when it's present and parses. Fallback
    path: when the header is missing or unparseable (chunked transfer
    encoding), wrap `receive` and abort once cumulative bytes exceed the cap.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = _parse_content_length(scope)
        if content_length is not None and content_length > self.max_bytes:
            await _send_413(send, content_length, self.max_bytes)
            return

        if content_length is None:
            # Header missing or unparseable — wrap both `receive` and `send`
            # so we can abort the request once cumulative bytes exceed the
            # cap and silently drop any response the downstream app tries to
            # emit afterward (uvicorn would otherwise raise on the duplicate
            # http.response.start).
            state = _AbortState()
            wrapped_receive = _make_counting_receive(receive, send, self.max_bytes, state)
            wrapped_send = _make_guarded_send(send, state)
            await self.app(scope, wrapped_receive, wrapped_send)
            return

        await self.app(scope, receive, send)


class _AbortState:
    """Shared mutable flag between the wrapped receive and wrapped send."""

    def __init__(self) -> None:
        self.aborted = False


def _parse_content_length(scope: Scope) -> int | None:
    raw = Headers(scope=scope).get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _make_counting_receive(
    receive: Receive, send: Send, max_bytes: int, state: _AbortState
) -> Receive:
    total = 0

    async def counting_receive() -> Message:
        nonlocal total
        if state.aborted:
            # Once aborted, signal disconnect so the app stops processing
            # rather than reading further chunks that violate ASGI ordering
            # after the synthetic more_body=False below.
            return {"type": "http.disconnect"}
        message = await receive()
        if message["type"] != "http.request":
            return message
        total += len(message.get("body", b""))
        if total > max_bytes:
            state.aborted = True
            await _send_413(send, total, max_bytes)
            # End the request stream cleanly from the app's perspective. The
            # guarded send drops whatever response the app tries to emit.
            return {"type": "http.request", "body": b"", "more_body": False}
        return message

    return counting_receive


def _make_guarded_send(send: Send, state: _AbortState) -> Send:
    async def guarded_send(message: Message) -> None:
        if state.aborted:
            return
        await send(message)

    return guarded_send


async def _send_413(send: Send, got_bytes: int, max_bytes: int) -> None:
    got_mb = got_bytes / 1024 / 1024
    max_mb = max_bytes // 1024 // 1024
    detail = f"upload exceeds the {max_mb} MB limit (got {got_mb:.1f} MB)"
    body = json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})
