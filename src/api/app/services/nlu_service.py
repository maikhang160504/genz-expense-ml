"""High-level NLU service. Picks real pipeline or mock based on settings + availability."""
from __future__ import annotations

import re
import time
from typing import Any

from app.adapters import expense_ocr_nlu as adapter
from app.adapters.mock_pipeline import run_nlu_mock
from app.core.config import get_settings
from app.core.exceptions import NLUProcessingError
from app.core.logging import get_logger
from app.schemas.nlu import NLURequest, NLUResponse


logger = get_logger(__name__)


def _sanitize_nlg_text(text: str) -> str:
    """Strip CJK chars, code snippets, and repeated words from LLM NLG response."""
    if not text:
        return text
    # 1. Remove CJK / Chinese characters (U+3000-U+9FFF range)
    text = re.sub(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+", "", text)
    # 2. Remove code patterns: .addAction(...), {...}, // comments
    text = re.sub(r"\.\w+\([^)]*\)", "", text)
    text = re.sub(r"\{[^}]{0,200}\}", "", text)
    text = re.sub(r"//.*", "", text)
    # 3. Collapse immediate repeated words: "mascot mascot" → "mascot"
    text = re.sub(r"\b(\w{3,})\s+\1\b", r"\1", text)
    # 4. Clean up extra whitespace and trailing punctuation
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.rstrip("，。！？,.")
    return text or "Mimo đã ghi nhận rồi nha!"

_MIMO_ASSETS = {
    "Alert", "Angry", "Approved", "Celebrate", "Chill", "Cooking", "Cool",
    "Determined", "Error", "Excited", "Giggle", "Happy", "Hello", "Loading",
    "Love", "Proud", "Relax", "Sad", "Sleepy", "Sassy", "Shopping", "Travel",
    "Sorry", "Success", "Taunting", "Thankful", "Thinking", "Working", "Worried",
}

_NLG_PERSONA_KEYS = frozenset({
    "hai_huoc", "dan_doi", "dong_cam", "cham_choc", "nghiem_tuc", "vui",
})

def _intent_mimo_fallback(intent: str | None, record_type: str | None = None) -> str:
    if intent == "Record":
        if record_type == "Income":
            return "Celebrate"
        return "Success"
    if intent == "Action":
        return "Approved"
    return "Hello"


def _coerce_mimo_asset(raw: str | None) -> str | None:
    """Giữ đúng tên LLM trong MIMO_ASSET_NAMES; None nếu không hợp lệ."""
    if not raw or not str(raw).strip():
        return None
    trimmed = str(raw).strip()
    if trimmed in _NLG_PERSONA_KEYS:
        return None
    if trimmed in _MIMO_ASSETS:
        return trimmed
    for asset in _MIMO_ASSETS:
        if asset.lower() == trimmed.lower():
            return asset
    return None


def _extract_nlg_text(raw: dict[str, Any]) -> str | None:
    for block_key in ("gemini_json", "llama_json"):
        block = raw.get(block_key)
        if not isinstance(block, dict):
            continue
        for field in ("response", "story"):
            val = block.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _extract_mimo_emotion_from_raw(raw: dict[str, Any], intent: str | None) -> str:
    """Ưu tiên mimo_emotion trong gemini_json/llama_json (output LLM thật)."""
    for block_key in ("gemini_json", "llama_json"):
        block = raw.get(block_key)
        if not isinstance(block, dict):
            continue
        for field in ("mimo_emotion", "emotion"):
            coerced = _coerce_mimo_asset(block.get(field))
            if coerced:
                return coerced

    for top_field in ("mimo_emotion", "llm_emotion", "mascot_mood"):
        coerced = _coerce_mimo_asset(raw.get(top_field))
        if coerced:
            return coerced

    record_type = raw.get("record_type")
    return _intent_mimo_fallback(intent, record_type)


class NLUService:
    def __init__(self) -> None:
        self._tried_load = False
        self._real_available = False

    def try_load(self) -> bool:
        if self._tried_load:
            return self._real_available
        self._tried_load = True
        if not get_settings().use_real_nlu:
            logger.info("USE_REAL_NLU=false -> mock pipeline only.")
            return False
        self._real_available = adapter.load_real_nlu_safe()
        if not self._real_available:
            logger.warning("Real NLU not loadable: %s", adapter.get_nlu_error())
        return self._real_available

    def reload(self) -> bool:
        self._tried_load = False
        self._real_available = adapter.reload_nlu()
        return self._real_available

    def infer(self, request: NLURequest) -> NLUResponse:
        start = time.perf_counter()
        try:
            payload = self._infer_inner(request)
        except Exception as exc:
            raise NLUProcessingError(f"NLU inference failed: {exc}") from exc
        payload["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return NLUResponse.model_validate(payload)

    def _infer_inner(self, request: NLURequest) -> dict[str, Any]:
        text = request.text
        persona = request.resolved_nlg_persona()
        if self.try_load():
            try:
                profile = request.profile.model_dump(exclude_none=True) if request.profile else {}
                corrections = [c.model_dump(exclude_none=True) for c in request.user_corrections] if request.user_corrections else None
                raw = adapter.run_real_nlu(
                    text,
                    profile=profile,
                    run_llm=request.run_llm,
                    nlg_persona=persona,
                    user_id=request.user_id,
                    user_corrections=corrections,
                )
                return self._normalize_real(raw, text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Real NLU failed at runtime, falling back to mock: %s", exc)
        return self._normalize_mock(text)

    def _normalize_real(self, raw: dict[str, Any], text: str) -> dict[str, Any]:
        intent = raw.get("intent") or "Unknown"
        record_type = raw.get("record_type")
        amount = (raw.get("amount_spent") or raw.get("amount")) if intent == "Record" else raw.get("action_param")

        multi_records_raw = raw.get("multi_records") or []
        multi_records = [
            {
                "text": m.get("text", ""),
                "amount": m.get("amount"),
                "category": m.get("category"),
                "record_type": m.get("record_type"),
            }
            for m in multi_records_raw
        ]

        mimo_emotion = _extract_mimo_emotion_from_raw(raw, intent)
        gemini_json = raw.get("gemini_json")
        if isinstance(gemini_json, dict):
            gemini_json = {**gemini_json, "mimo_emotion": mimo_emotion, "emotion": mimo_emotion}

        nlg_text = _sanitize_nlg_text(raw.get("nlg_response") or _extract_nlg_text(raw))
        return {
            "intent": intent,
            "intent_confidence": raw.get("intent_confidence"),
            "text": text,
            "item": raw.get("item"),
            "category": raw.get("category"),
            "amount": amount,
            "record_type": record_type,
            "is_expense": record_type == "Expense" if intent == "Record" else None,
            "income_type": raw.get("income_type"),
            "action_type": raw.get("action_type"),
            "action_details": raw.get("action_details"),
            "time_range": raw.get("time_range"),
            "multi_records": multi_records,
            "multi_record_task": bool(raw.get("multi_record_task")),
            "sentiment": raw.get("sentiment"),
            "nlg_persona": raw.get("nlg_persona"),
            "nlg_prompt": raw.get("nlg_prompt"),
            "gemini_json": gemini_json,
            "llama_json": raw.get("llama_json"),
            "nlg_response": nlg_text,
            "mimo_emotion": mimo_emotion,
            "llm_emotion": mimo_emotion,
            "mascot_mood": mimo_emotion,
            "backend": raw.get("backend") or "real",
            "debug_info": raw.get("debug_info"),
        }

    def _normalize_mock(self, text: str) -> dict[str, Any]:
        result = run_nlu_mock(text)
        multi = []
        if result.multi_amounts:
            multi = [
                {"text": text, "amount": amt, "category": result.category, "record_type": result.record_type}
                for amt in result.multi_amounts
            ]
        mimo_emotion = _intent_mimo_fallback(result.intent, result.record_type)

        fallback_emotion = mimo_emotion
        category = result.category or "Others"
        amount = result.amount
        
        viet_category_map = {
            "Food": "Ăn uống",
            "Transport": "Di chuyển",
            "Housing": "Nhà ở",
            "Shopping": "Mua sắm",
            "Entertainment": "Giải trí",
            "Health": "Sức khỏe",
            "Education": "Giáo dục",
            "Others": "Tiêu dùng khác",
            "Other": "Tiêu dùng khác",
            "Essentials": "Thiết yếu",
            "Beauty": "Làm đẹp",
            "Social": "Xã hội",
            "Salary": "Lương",
            "Bonus": "Thưởng",
            "Business": "Kinh doanh"
        }
        viet_cat = viet_category_map.get(category, category)
        
        if result.intent == "Record":
            if amount is not None and amount > 0:
                amt_str = f"{int(amount):,}".replace(",", ".")
                if result.record_type == "Income":
                    fallback_text = f"Tuyệt vời! Mimo đã ghi nhận khoản thu nhập {amt_str}đ vào danh mục {viet_cat}. Tích tiểu thành đại, cố gắng phát huy nhé! 🎉"
                else:
                    fallback_text = f"Mimo đã ghi nhận khoản chi {amt_str}đ cho {viet_cat} vào ví của bạn. Hãy cân đối chi tiêu hợp lý nhé!"
            else:
                if result.record_type == "Income":
                    fallback_text = f"Mimo đã chuẩn bị sẵn phiếu ghi nhận thu nhập cho danh mục {viet_cat}. Bạn hãy nhập số tiền và bấm lưu nhé!"
                else:
                    fallback_text = f"Mimo đã chuẩn bị sẵn phiếu ghi nhận chi tiêu cho danh mục {viet_cat}. Bạn hãy nhập số tiền và bấm lưu nhé!"
        elif result.intent == "Action":
            action_type = result.action_type or "Thao tác"
            action_labels = {
                "LIMIT": "Đặt hạn mức",
                "DELETE": "Xóa giao dịch gần nhất",
                "GOAL": "Tạo mục tiêu tiết kiệm",
                "TONE": "Đổi giọng nói Mimo",
                "SEARCH": "Tìm kiếm giao dịch",
                "SETTING": "Mở cài đặt ứng dụng"
            }
            action_label = action_type
            for key, val in action_labels.items():
                if key in action_type.upper():
                    action_label = val
                    break
            fallback_text = f"Mimo đã chuẩn bị sẵn thao tác: {action_label}. Bạn có muốn thực hiện không?"
        else:
            fallback_text = "Chào bạn! Tôi là Mimo. Hôm nay bạn thế nào? Cần tôi hỗ trợ gì về quản lý chi tiêu không?"

        gemini_json = {
            "response": fallback_text,
            "story": fallback_text,
            "mimo_emotion": fallback_emotion,
            "emotion": fallback_emotion
        }

        return {
            "intent": result.intent,
            "intent_confidence": result.intent_confidence,
            "text": text,
            "item": None,
            "category": result.category,
            "amount": result.amount,
            "record_type": result.record_type,
            "is_expense": result.record_type == "Expense" if result.record_type else None,
            "income_type": None,
            "action_type": result.action_type,
            "action_details": None,
            "time_range": result.time_range,
            "multi_records": multi,
            "multi_record_task": len(multi) >= 1,
            "sentiment": None,
            "nlg_prompt": None,
            "gemini_json": gemini_json,
            "llama_json": None,
            "nlg_response": fallback_text,
            "mimo_emotion": mimo_emotion,
            "llm_emotion": mimo_emotion,
            "mascot_mood": mimo_emotion,
            "backend": "mock",
        }


_singleton: NLUService | None = None


def get_nlu_service() -> NLUService:
    global _singleton
    if _singleton is None:
        _singleton = NLUService()
    return _singleton
