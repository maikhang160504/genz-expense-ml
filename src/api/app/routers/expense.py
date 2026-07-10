"""Combined high-level endpoints: text → expense, bill → expense."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidInputError
from app.schemas.expense import (
    ExpenseFromImageResponse,
    ExpenseFromTextRequest,
    ExpenseFromTextResponse,
)
from app.services.expense_service import get_expense_service


router = APIRouter(prefix="/expense", tags=["expense"])


MAX_BYTES = 8 * 1024 * 1024


@router.post(
    "/from-text",
    response_model=ExpenseFromTextResponse,
    summary="Trích xuất giao dịch từ câu tự nhiên (ăn phở 45k → amount=45000 category=Food)",
)
def from_text(payload: ExpenseFromTextRequest) -> ExpenseFromTextResponse:
    return get_expense_service().from_text(payload)


@router.post(
    "/from-bill",
    response_model=ExpenseFromImageResponse,
    summary="OCR ảnh hóa đơn rồi build giao dịch (luôn requires_confirmation=true)",
)
async def from_bill(
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),
) -> ExpenseFromImageResponse:
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise InvalidInputError("Image is larger than 8 MB.")
    return get_expense_service().from_image(raw, file.filename)
