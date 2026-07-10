"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.core.config import get_settings
from app.core.exceptions import AIServiceError
from app.core.logging import setup_logging, get_logger
from app.middleware import ApiKeyMiddleware, RequestLoggingMiddleware
from app.routers import chat, expense, health, nlu, ocr
from app.schemas.common import ErrorBody, ErrorResponse
from app.schemas.nlu import NLURequest, NLUResponse
from app.services.nlu_service import get_nlu_service


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("startup")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        from app.services.nlu_service import get_nlu_service
        from app.services.ocr_service import get_ocr_service
        from app.adapters.expense_ocr_nlu import pre_import_real_backends

        pre_import_real_backends()

        if settings.use_real_ocr and settings.preload_ocr:
            logger.info("Pre-loading OCR models...")
            try:
                ocr_loaded = get_ocr_service().try_load()
                logger.info("OCR preload finished", extra={"ocr_loaded": ocr_loaded})
            except Exception as e:
                logger.warning("OCR preload failed (will retry on first bill): %s", e)

        # Optional eager NLU load — skip when LAZY_LOAD_MODELS=true (default) for fast startup
        if settings.lazy_load_models:
            logger.info(
                "NLU lazy load enabled — NLU loads on first infer or POST /internal/reload-models",
                extra={"event": "lazy_load_skipped"},
            )
        else:
            logger.info("Starting NLU model loading on startup...")
            try:
                nlu_loaded = get_nlu_service().try_load()
                ocr_loaded = get_ocr_service().try_load() if not (settings.use_real_ocr and settings.preload_ocr) else True
                logger.info(
                    "Model loading completed",
                    extra={
                        "event": "models_loaded",
                        "extra_fields": {
                            "nlu_loaded": nlu_loaded,
                            "ocr_loaded": ocr_loaded,
                            "use_real_nlu": settings.use_real_nlu,
                            "use_real_ocr": settings.use_real_ocr,
                        },
                    },
                )
            except Exception as e:
                logger.error(f"Error during model loading: {e}")

        try:
            yield
        finally:
            logger.info("AI service shutdown")

    app = FastAPI(
        lifespan=lifespan,
        title=settings.app_name,
        version=__version__,
        description=(
            "Microservice nhận dạng chi tiêu (NLU văn bản tiếng Việt + OCR hóa đơn).\n\n"
            "Hai backend: `real` (load pipeline gốc từ `expense-ocr-nlu`) và `mock` "
            "(regex + keyword) — service tự fallback nếu không tải được model thật."
        ),
        contact={"name": "MoneyStory", "url": "https://example.com"},
        license_info={"name": "Internal"},
        openapi_tags=[
            {"name": "health", "description": "Liveness + readiness."},
            {"name": "nlu", "description": "Text NLU: intent + amount + category."},
            {"name": "ocr", "description": "OCR hóa đơn (ảnh) hoặc text đã extract."},
            {"name": "expense", "description": "Endpoint cấp cao: text/ảnh → giao dịch."},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ApiKeyMiddleware)

    @app.exception_handler(AIServiceError)
    async def ai_service_error_handler(_: Request, exc: AIServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=ErrorBody(code=exc.code, message=exc.message, details=exc.details)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorBody(
                    code="invalid_input",
                    message="Request payload validation failed.",
                    details={"errors": exc.errors()},
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def fallback_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorBody(code="internal_error", message=str(exc))
            ).model_dump(),
        )

    app.include_router(health.router)
    api_prefix = "/api/v1"
    app.include_router(nlu.router, prefix=api_prefix)
    app.include_router(ocr.router, prefix=api_prefix)
    app.include_router(expense.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)
    from app.routers import bill_retrain, internal
    app.include_router(bill_retrain.router, prefix=api_prefix)
    app.include_router(internal.router, prefix=api_prefix)

    # ── Backward-compatible /infer route (old clients) ──────────
    @app.post("/infer", response_model=NLUResponse, tags=["compat"],
              summary="Legacy /infer endpoint – delegates to /api/v1/nlu/infer")
    def legacy_infer(payload: NLURequest) -> NLUResponse:
        return get_nlu_service().infer(payload)

    return app


app = create_app()
