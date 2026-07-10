"""Health & readiness endpoints."""
from __future__ import annotations

import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.adapters import expense_ocr_nlu as adapter
from app.core.config import get_settings
from app.schemas.common import HealthResponse


router = APIRouter(tags=["health"])

_STARTUP_TIME = datetime.datetime.now(tz=datetime.timezone.utc)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    settings = get_settings()
    nlu_version = None
    if settings.use_real_nlu:
        try:
            from pathlib import Path
            from app.services.nlu_registry import load_registry
            reg = load_registry(Path(settings.expense_ocr_nlu_dir).resolve())
            nlu_version = reg.get("version")
        except Exception:
            nlu_version = "v1.1-global"
    return HealthResponse(
        version=__version__,
        environment=settings.environment,
        nlu_real=settings.use_real_nlu,
        ocr_real=settings.use_real_ocr,
        nlu_loaded=adapter.is_nlu_loaded(),
        ocr_loaded=adapter.is_ocr_loaded(),
        nlu_version=nlu_version,
    )


class StatusResponse(BaseModel):
    model_version: str
    service_version: str
    environment: str
    nlu_backend: str
    ocr_backend: str
    nlu_loaded: bool
    ocr_loaded: bool
    uptime_seconds: int
    started_at: str


@router.get(
    "/api/v1/internal/status",
    response_model=StatusResponse,
    summary="N6: Internal service status — model versions + load state",
    tags=["internal"],
)
def internal_status() -> StatusResponse:
    from app.services.nlu_service import get_nlu_service
    from app.services.ocr_service import get_ocr_service

    # Trigger model load if not already initialized
    get_nlu_service().try_load()
    get_ocr_service().try_load()

    settings = get_settings()
    nlu_loaded = adapter.is_nlu_loaded()
    ocr_loaded = adapter.is_ocr_loaded()
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    uptime = int((now - _STARTUP_TIME).total_seconds())
    return StatusResponse(
        model_version=__version__,
        service_version=__version__,
        environment=settings.environment,
        nlu_backend="real" if nlu_loaded else "mock",
        ocr_backend="real" if ocr_loaded else "mock",
        nlu_loaded=nlu_loaded,
        ocr_loaded=ocr_loaded,
        uptime_seconds=uptime,
        started_at=_STARTUP_TIME.isoformat(),
    )
