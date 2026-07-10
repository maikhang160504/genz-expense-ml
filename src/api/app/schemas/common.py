"""Shared response envelope + error model."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(..., examples=["invalid_input"])
    message: str = Field(..., examples=["Field 'text' is required."])
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "expense-ai-service"
    version: str
    environment: str
    nlu_real: bool
    ocr_real: bool
    nlu_loaded: bool
    ocr_loaded: bool
    nlu_version: str | None = None
