"""HTTP middlewares: request logging, API key auth."""
from __future__ import annotations

import time
import uuid

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger("http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "request",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "status": status_code,
                    "latency_ms": elapsed_ms,
                    "request_id": request_id,
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Optional shared-secret auth: enabled only when settings.api_key is set."""

    PUBLIC_PATHS = ("/health", "/docs", "/redoc", "/openapi.json", "/api/v1/internal/status")

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if not settings.api_key:
            return await call_next(request)
        if any(request.url.path.startswith(p) for p in self.PUBLIC_PATHS):
            return await call_next(request)
        header = request.headers.get("x-api-key")
        if header != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid X-API-Key.",
            )
        return await call_next(request)
