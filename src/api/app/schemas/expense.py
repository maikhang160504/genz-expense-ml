"""High-level "expense from text/image" schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .nlu import NLUResponse
from .ocr import OCRResponse


class ExpenseFromTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    user_id: str | None = None
    profile: dict | None = None
    run_llm: bool = False


class ExpenseExtracted(BaseModel):
    model_config = ConfigDict(extra="allow")

    amount: float | None = None
    category: str | None = None
    note: str | None = None
    record_type: Literal["Expense", "Income"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ExpenseFromTextResponse(BaseModel):
    success: bool = True
    flow: Literal["text"] = "text"
    extracted: ExpenseExtracted
    requires_category_selection: bool = False
    nlu: NLUResponse
    latency_ms: int | None = None


class ExpenseFromImageResponse(BaseModel):
    success: bool = True
    flow: Literal["bill"] = "bill"
    extracted: ExpenseExtracted
    requires_confirmation: bool = True
    ocr: OCRResponse
    nlu: NLUResponse | None = None
    latency_ms: int | None = None
