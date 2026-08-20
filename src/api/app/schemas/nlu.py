"""Pydantic schemas for the NLU endpoint."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Intent = Literal["Record", "Action", "Chitchat", "Unknown"]
RecordType = Literal["Expense", "Income"]


class NLUProfile(BaseModel):
    """User context attached to a request (budget, preference, ...)."""

    model_config = ConfigDict(extra="allow")

    user_id: str | None = None
    budget_total: float | None = Field(default=None, ge=0)
    budget_remain: float | None = Field(default=None, ge=0)
    preferred_vibe: str | None = None


class UserCorrectionItem(BaseModel):
    text: str
    category_code: str | None = None
    intent: str | None = None
    record_type: str | None = None


class NLURequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["ăn phở 45k"])
    profile: NLUProfile | None = None
    run_llm: bool = False
    nlg_persona: str | None = Field(
        default=None,
        description="Giọng NLG (hai_huoc, dan_doi, ...) — không phải tên file PNG",
        examples=["hai_huoc"],
    )
    emotion: str | None = Field(
        default=None,
        deprecated=True,
        description="Alias cũ của nlg_persona",
        examples=["hai_huoc"],
    )
    user_id: str | None = None
    user_corrections: list[UserCorrectionItem] | None = None
    caller_context: str | None = Field(
        default="chat",
        description=(
            "Ngữ cảnh gọi: 'chat' (mặc định — chạy Stage 1 phân loại intent bình thường) "
            "hoặc 'addstory' (bỏ qua Stage 1, luôn force intent = Record — "
            "dùng khi người dùng đang ở màn hình AddStory của app)."
        ),
        examples=["chat", "addstory"],
    )

    def resolved_nlg_persona(self) -> str:
        return (self.nlg_persona or self.emotion or "hai_huoc").strip()


class MultiRecordItem(BaseModel):
    text: str
    item: str | None = None
    note: str | None = None
    amount: float | None = None
    category: str | None = None
    record_type: RecordType | None = None


class IntentRecordResponse(BaseModel):
    id: str | int | None = None
    text: str
    category: str | None = None
    type: RecordType | None = None
    is_money: bool = True
    confidence: float | None = None
    engine: str | None = None


class TestPromptRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    override_prompt: str = Field(..., description="System prompt to test")
    persona: str | None = None


class ActionDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    verb: str | None = None
    target: str | None = None
    target_type: str | None = None
    value: float | None = None
    unit: str | None = None


class NLUResponse(BaseModel):
    """Normalized NLU output consumed by the backend."""

    model_config = ConfigDict(extra="allow")

    intent: Intent
    intent_confidence: float | None = None
    text: str
    item: str | None = None
    category: str | None = None
    amount: float | None = None
    record_type: RecordType | None = None
    is_expense: bool | None = None
    income_type: str | None = None
    action_type: str | None = None
    action_details: ActionDetails | None = None
    multi_records: list[MultiRecordItem] = []
    multi_record_task: bool = False
    sentiment: str | None = None
    llm_json: dict[str, Any] | None = None
    gemini_json: dict[str, Any] | None = None  # deprecated alias for BE compat
    llama_json: dict[str, Any] | None = None
    nlg_persona: str | None = Field(default=None, description="Persona NLG đã dùng (hai_huoc, ...)")
    mimo_emotion: str | None = Field(default=None, description="Tên file PNG mascot (PascalCase)")
    llm_emotion: str | None = Field(default=None, description="Alias of mimo_emotion")
    mascot_mood: str | None = Field(default=None, description="Alias of mimo_emotion for Flutter")
    nlg_response: str | None = None
    backend: str = Field(default="mock", description="`real` if full pipeline used, else `mock`.")
    rule_used: str | None = Field(default=None, description="Tên rule đã sử dụng trong llm_v2")
    latency_ms: int | None = None
    suggested_actions: list[str] | None = Field(default=None, description="Danh sách các tính năng/câu hỏi gợi ý khi intent là Chitchat")
