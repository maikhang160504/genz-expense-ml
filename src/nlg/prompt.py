import json
from typing import Any

_LIST_EMOTION_INSTRUCTION = (
    "\nYÊU CẦU ĐẦU RA JSON: Trả về một đối tượng JSON có chính xác 2 trường:\n"
    "1. \"response\": Câu thoại nhận xét chi tiêu hoặc câu thoại chitchat tương ứng. "
    "**QUAN TRỌNG: Giới hạn tối đa 30 từ. Ngắn gọn, súc tích, đúng vai.**\n"
    "2. \"emotion\": Chọn chính xác 1 trong danh sách emotions sau ( PascalCase ): "
    "Alert, Angry, Approved, Celebrate, Chill, Cooking, Cool, Determined, Error, Excited, Giggle, Happy, Hello, Loading, Love, Proud, Relax, Sad, Sleepy, Sassy, Shopping, Travel, Sorry, Success, Taunting, Thankful, Thinking, Working, Worried."
)


def _format_context_meta(ctx: dict[str, Any]) -> str:
    """Chuyển context_metadata thành đoạn text có cấu trúc cho LLM dễ đọc."""
    parts = []
    if ctx.get("time_of_day"):
        parts.append(f"- Thời điểm: {ctx['time_of_day']}")
    if ctx.get("weather") and ctx["weather"] != "không_rõ":
        parts.append(f"- Thời tiết: {ctx['weather']}")
    if ctx.get("day_of_month"):
        parts.append(f"- Ngày trong tháng: {ctx['day_of_month']}")
    if ctx.get("days_to_payday") is not None:
        d = ctx["days_to_payday"]
        if d == 0:
            parts.append("- Hôm nay là ngày nhận lương 💸")
        else:
            parts.append(f"- Còn {d} ngày nữa tới kỳ lương")
    if ctx.get("wallet_health") and ctx["wallet_health"] != "không_rõ":
        health_label = {"an_toan": "✅ An toàn", "can_than": "⚠️ Cẩn thận", "bao_dong": "🚨 Báo động"}.get(
            ctx["wallet_health"], ctx["wallet_health"]
        )
        parts.append(f"- Sức khoẻ ví: {health_label}")
    if ctx.get("historical_fact"):
        parts.append(f"- Sự kiện tài chính: {ctx['historical_fact']}")
    msg_data = ctx.get("message_data") or {}
    if msg_data.get("remaining") is not None:
        parts.append(f"- Ngân sách còn lại: {msg_data['remaining']:,}đ")
    return "\n".join(parts) if parts else json.dumps(ctx, ensure_ascii=False)


def _build_relationship_instruction(prompts: dict[str, Any], relationship_tag: str | None, emotion: str) -> str:
    """Trả về chỉ thị override nếu có relationship_tag."""
    if not relationship_tag:
        return ""
    overrides = prompts.get("relationship_override", {})
    tag = relationship_tag.upper()
    if tag == "CHA_ME":
        cfg = overrides.get("CHA_ME", {})
        return cfg.get("rule", "")
    if tag == "NGUOI_YEU":
        cfg = overrides.get("NGUOI_YEU", {})
        positive_emotions = {"vui", "hai_huoc", "dong_cam"}
        if emotion in positive_emotions:
            return cfg.get("rule_happy", "")
        return cfg.get("rule_sad", "")
    return ""


def build_nlg_prompt(
    prompts: dict[str, Any],
    emotion: str,
    nlu_result: dict[str, Any],
    context_metadata: dict[str, Any],
) -> dict[str, Any]:
    common = prompts.get("common", {})
    emotion_cfg = prompts.get("emotions", {}).get(emotion, {})
    diversity_rule = common.get("context_diversity_rule", "")
    relationship_tag = nlu_result.get("relationship_tag")
    rel_instruction = _build_relationship_instruction(prompts, relationship_tag, emotion)
    ctx_text = _format_context_meta(context_metadata)

    system_parts = [emotion_cfg.get("system"), common.get("style"), common.get("response_rules")]
    if diversity_rule:
        system_parts.append(diversity_rule)
    if rel_instruction:
        system_parts.insert(1, f"[QUAN HỆ OVERRIDE] {rel_instruction}")

    system_prompt = " ".join(s for s in system_parts if s)

    intent = nlu_result.get("intent")

    if intent == "Chitchat":
        rules = common.get("chitchat_response_rules") or common.get("response_rules")
        system_prompt = " ".join(
            s for s in [emotion_cfg.get("system"), common.get("style"), diversity_rule, rules] if s
        )
        base_user = common.get("chitchat_user") or emotion_cfg.get("user")
        user_msg = nlu_result.get("text") or ""
        user_prompt = (
            f"{base_user} {rules} "
            f"Câu người dùng: \"{user_msg}\". "
            "Đọc câu để trả lời đúng chủ đề; chọn status (vui/buon/trung_lap) từ ngữ cảnh câu — không dùng nhãn sentiment NLU.\n"
            f"CONTEXT_META:\n{ctx_text}"
            f"{_LIST_EMOTION_INSTRUCTION}"
        )

    elif intent == "Action":
        rules = common.get("action_response_rules") or common.get("response_rules")
        system_prompt = " ".join(
            s for s in [emotion_cfg.get("system"), common.get("style"), rules] if s
        )
        base_user = common.get("action_user") or emotion_cfg.get("user")
        action_type = nlu_result.get("action_type")
        user_prompt = (
            f"{base_user} {rules} "
            f"Loại thao tác: {action_type}. "
            "Không đặt tên món ăn giả; không thêm giao dịch không có trong input.\n"
            f"CONTEXT_META:\n{ctx_text}"
            f"{_LIST_EMOTION_INSTRUCTION}"
        )

    else:
        base_user = emotion_cfg.get("user")
        item = nlu_result.get("item") or ""
        amount = nlu_result.get("amount")
        context_type = context_metadata.get("type") or "NONE"
        amount_str = f"{amount:,}đ" if amount is not None else "không rõ"
        user_prompt = (
            f"{base_user} {common.get('response_rules', '')} "
            f"Món hoặc hạng mục: {item}. Số tiền: {amount_str}. "
            f"Kiểu cảnh báo: {context_type}. "
            f"{rel_instruction + ' ' if rel_instruction else ''}"
            f"CONTEXT_META (phối hợp ≥2 yếu tố):\n{ctx_text}"
            f"{_LIST_EMOTION_INSTRUCTION}"
        )

    return {
        "system": system_prompt.strip(),
        "user": user_prompt.strip(),
        "input": {
            "nlu_result": nlu_result,
            "context_metadata": context_metadata,
            "emotion": emotion,
        },
    }
