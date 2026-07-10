"""Chat endpoint — rule-based Mimo responses with optional LLM fallback."""
from __future__ import annotations

import re
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    user_id: str | None = None
    emotion: str | None = None


class ChatResponse(BaseModel):
    response: str
    intent_action: dict[str, Any] | None = None
    backend: str = "mock"
    latency_ms: int | None = None


_GREET = re.compile(r"(xin chào|hello|chào|hi\b)", re.I)
_BALANCE = re.compile(r"(số dư|balance|còn bao nhiêu|tài khoản)", re.I)
_EXPENSE = re.compile(r"(chi tiêu|chi bao nhiêu|tốn|tiêu|mua)", re.I)
_SAVE = re.compile(r"(tiết kiệm|save|để dành)", re.I)
_HELP = re.compile(r"(giúp|help|hướng dẫn|làm sao|thế nào)", re.I)
_THANKS = re.compile(r"(cảm ơn|thank|ok|được rồi|ổn rồi)", re.I)


def _mock_reply(text: str) -> tuple[str, dict | None]:
    if _GREET.search(text):
        return "Xin chào! Mình là Mimo 😊 Mình có thể giúp bạn theo dõi chi tiêu, xem số dư, hay đặt mục tiêu tiết kiệm. Bạn cần gì nào?", None
    if _BALANCE.search(text):
        return "Để xem số dư chính xác, bạn hãy vào màn hình Tổng quan nhé! Mình không có quyền truy cập trực tiếp vào tài khoản ngân hàng của bạn 😄", None
    if _EXPENSE.search(text):
        return "Bạn có thể xem chi tiết chi tiêu trong tab Báo cáo 📊 Hoặc nhắn cho mình kiểu 'ăn phở 45k' để mình ghi lại luôn nha!", {"type": "navigate", "target": "report"}
    if _SAVE.search(text):
        return "Tiết kiệm là chìa khóa tài chính! 🔑 Bạn đã đặt mục tiêu tiết kiệm chưa? Vào tab Mục tiêu để tạo mới nhé. Mình sẽ nhắc bạn đóng góp đều đặn!", {"type": "navigate", "target": "goals"}
    if _THANKS.search(text):
        return "Không có gì! Cần gì cứ hỏi Mimo nha 😊", None
    if _HELP.search(text):
        return (
            "Mình có thể giúp bạn:\n"
            "• Ghi chép chi tiêu (ví dụ: 'cà phê 35k')\n"
            "• Xem báo cáo chi tiêu theo tháng\n"
            "• Theo dõi mục tiêu tiết kiệm\n"
            "• Đặt giới hạn ngân sách theo danh mục\n\n"
            "Bạn muốn bắt đầu với điều gì? 😄",
            None,
        )
    return (
        "Mình nghe bạn rồi! 😊 Bạn có thể nói cụ thể hơn không? "
        "Ví dụ: 'ăn trưa 50k', 'xem chi tiêu tháng này', hay 'đặt mục tiêu tiết kiệm'.",
        None,
    )


@router.post("", response_model=ChatResponse, summary="Chat với Mimo AI assistant")
def chat(payload: ChatRequest) -> ChatResponse:
    t0 = time.monotonic()
    last_user = next(
        (m.content for m in reversed(payload.messages) if m.role == "user"), ""
    )
    response, intent_action = _mock_reply(last_user)
    latency = int((time.monotonic() - t0) * 1000)
    return ChatResponse(
        response=response,
        intent_action=intent_action,
        backend="mock",
        latency_ms=latency,
    )
