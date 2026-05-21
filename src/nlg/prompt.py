import json
from typing import Any


def build_nlg_prompt(
    prompts: dict[str, Any],
    emotion: str,
    nlu_result: dict[str, Any],
    context_metadata: dict[str, Any],
) -> dict[str, Any]:
    common = prompts.get("common", {})
    emotion_cfg = prompts.get("emotions", {}).get(emotion, {})

    system_prompt = " ".join(
        s
        for s in [
            emotion_cfg.get("system"),
            common.get("style"),
            common.get("response_rules"),
        ]
        if s
    )

    intent = nlu_result.get("intent")
    if intent == "Chitchat":
        rules = common.get("chitchat_response_rules") or common.get("response_rules")
        system_prompt = " ".join(
            s for s in [emotion_cfg.get("system"), common.get("style"), rules] if s
        )
        base_user = common.get("chitchat_user") or emotion_cfg.get("user")
        user_prompt = " ".join(s for s in [base_user, rules] if s)
        ctx_json = json.dumps(context_metadata, ensure_ascii=False)
        user_msg = nlu_result.get("text") or ""
        user_prompt = (
            f"{user_prompt} Câu người dùng: \"{user_msg}\". "
            "Đọc câu để trả lời đúng chủ đề; chọn status (vui/buon/trung_lap) từ ngữ cảnh câu — "
            "không dùng nhãn sentiment NLU. "
            f"Ngữ cảnh app (JSON): {ctx_json}."
        )
    elif intent == "Action":
        rules = common.get("action_response_rules") or common.get("response_rules")
        system_prompt = " ".join(
            s for s in [emotion_cfg.get("system"), common.get("style"), rules] if s
        )
        base_user = common.get("action_user") or emotion_cfg.get("user")
        user_prompt = " ".join(s for s in [base_user, rules] if s)
        action_type = nlu_result.get("action_type")
        ctx_json = json.dumps(context_metadata, ensure_ascii=False)
        user_prompt = (
            f"{user_prompt} Loại thao tác: {action_type}. "
            f"Ngữ cảnh hệ thống (JSON): {ctx_json}. "
            "Không đặt tên món ăn giả; không thêm giao dịch không có trong input."
        )
    else:
        base_user = emotion_cfg.get("user")
        user_prompt = " ".join(s for s in [base_user, common.get("response_rules")] if s)
        item = nlu_result.get("item") or ""
        amount = nlu_result.get("amount")
        action_type = nlu_result.get("action_type")
        context_type = context_metadata.get("type") or "NONE"
        ctx_json = json.dumps(context_metadata, ensure_ascii=False)
        user_prompt = (
            f"{user_prompt} Món hoặc hạng mục: {item}. Số tiền: {amount}. "
            f"Hành động (nếu có): {action_type}. Kiểu cảnh báo: {context_type}. "
            f"Ngữ cảnh đầy đủ (JSON): {ctx_json}. "
            "Không đổi tên món; không tự thêm giao dịch."
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
