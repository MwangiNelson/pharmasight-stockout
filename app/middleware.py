"""
Custom middleware stack.

Includes:
  - **RequestIDMiddleware** — adds a unique ``X-Request-ID`` header to every
    request/response for tracing across logs.
  - **RequestLoggingMiddleware** — logs method, path, status, and duration of
    every request.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("pharma_sight.middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Inject a unique request ID into each request's state and echo it back
    in the ``X-Request-ID`` response header.  If the caller already supplies
    the header, it is reused for correlation.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and duration (ms)."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        rid = getattr(request.state, "request_id", "-")
        logger.info(
            "%s %s → %d (%.1f ms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            rid,
        )
        return response
