"""OCR endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from app.schemas.ocr import OCRLine, OCRResponse, OCRSuggestion, OCRTextRequest
from app.services.ocr_service import get_ocr_service


router = APIRouter(prefix="/ocr", tags=["ocr"])


MAX_BYTES = 8 * 1024 * 1024  # 8 MB


@router.post(
    "/image",
    response_model=OCRResponse,
    summary="OCR ảnh hóa đơn → text + amount + category",
)
async def ocr_image(
    file: UploadFile = File(..., description="Receipt image (jpg/png)"),
    locale: str | None = Form(default="vi"),
) -> OCRResponse:
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        from app.core.exceptions import InvalidInputError

        raise InvalidInputError("Image is larger than 8 MB.")
    return get_ocr_service().infer_image(raw, file.filename)


@router.post(
    "/text",
    response_model=OCRResponse,
    summary="Parse ảnh đã được OCR bên ngoài (đưa thẳng text)",
)
def ocr_from_text(payload: OCRTextRequest) -> OCRResponse:
    return get_ocr_service().infer_text(payload.text)


class OCRReviewResponse(BaseModel):
    """Lightweight review response — text + raw lines for user inspection."""

    text: str
    lines: list[OCRLine] = []
    suggestion: OCRSuggestion
    review_hint: str = "Hãy kiểm tra và chỉnh sửa nếu cần trước khi lưu."
    backend: str = "mock"
    latency_ms: int | None = None
    extra_fields: dict[str, Any] = {}


@router.post(
    "/review",
    response_model=OCRReviewResponse,
    summary="OCR hóa đơn → text + lines để user kiểm tra trước khi lưu",
)
async def ocr_review(
    file: UploadFile = File(..., description="Receipt image (jpg/png)"),
    locale: str | None = Form(default="vi"),
) -> OCRReviewResponse:
    """N5: Returns raw OCR output for user review — does NOT save any transaction."""
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        from app.core.exceptions import InvalidInputError

        raise InvalidInputError("Image is larger than 8 MB.")
    result = get_ocr_service().infer_image(raw, file.filename)
    return OCRReviewResponse(
        text=result.text,
        lines=result.lines,
        suggestion=result.suggestion,
        backend=result.backend,
        latency_ms=result.latency_ms,
        extra_fields={k: v for k, v in result.model_extra.items()} if result.model_extra else {},
    )
