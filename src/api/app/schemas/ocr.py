"""Pydantic schemas for OCR endpoint."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OCRTextRequest(BaseModel):
    """Used when the caller already has the raw text extracted by another OCR."""

    text: str = Field(..., min_length=1)
    profile: dict | None = None


class OCRLine(BaseModel):
    text: str
    bbox: list[float] | None = None
    confidence: float | None = None


class OCRSuggestion(BaseModel):
    amount: float | None = None
    category: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    currency: str = "VND"


class OCRResponse(BaseModel):
    """Output of the receipt OCR pipeline (PaddleOCR + VietOCR + rule field extractor)."""

    model_config = ConfigDict(extra="allow")

    success: bool = True
    text: str = Field(..., description="Raw text joined from OCR lines.")
    lines: list[OCRLine] = []
    suggestion: OCRSuggestion
    requires_confirmation: bool = True
    backend: str = Field(default="mock")
    latency_ms: int | None = None
