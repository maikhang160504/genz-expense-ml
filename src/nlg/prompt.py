import json
import random
from typing import Any

from src.nlg.mimo_assets import MIMO_ASSET_NAMES

_MIMO_ASSET_LIST = ", ".join(sorted(MIMO_ASSET_NAMES))

def _is_income_record(nlu_result: dict[str, Any]) -> bool:
    if nlu_result.get("record_type") == "Income":
        return True
    return nlu_result.get("is_expense") is False


def _record_type_instruction(nlu_result: dict[str, Any]) -> str:
    if _is_income_record(nlu_result):
        income_type = nlu_result.get("income_type") or ""
        extra = f" Loại thu: {income_type}." if income_type else ""
        return (
            "LOẠI GIAO DỊCH: Thu nhập (Income)."
            f"{extra} "
            "Phản hồi phải nói tiền VÀO ví / thu / nhận — "
            "TUYỆT ĐỐI KHÔNG nói chi tiêu, mua, tiêu xài, ngân sách cạn, ét ô ét vì mất tiền."
        )
    return (
        "LOẠI GIAO DỊCH: Chi tiêu (Expense). "
        "Phản hồi phải nói tiền RA / chi / mua — "
        "TUYỆT ĐỐI KHÔNG nói thu nhập, lương về, tiền vào ví. "
        "Có thể nhắc ngân sách nếu CONTEXT_META có cảnh báo."
    )


_MIMO_EMOTION_INSTRUCTION = (
    "\nYÊU CẦU ĐẦU RA JSON: Trả về đúng 2 trường:\n"
    "1. \"response\": Câu thoại (tối đa 30 từ).\n"
    f"2. \"mimo_emotion\": Dựa vào ngữ nghĩa của đầu vào và reponse, hãy chọn ĐÚNG 1 tên trong danh sách (PascalCase): {_MIMO_ASSET_LIST}. "
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
    if ctx.get("record_type"):
        label = "Thu nhập" if ctx["record_type"] == "Income" else "Chi tiêu"
        parts.append(f"- Loại giao dịch: {label} ({ctx['record_type']})")
    if ctx.get("historical_fact"):
        parts.append(f"- Sự kiện tài chính: {ctx['historical_fact']}")
    if ctx.get("spent_last_month") is not None:
        parts.append(f"- Chi tiêu tháng trước: {ctx['spent_last_month']:,}đ")
    msg_data = ctx.get("message_data") or {}
    if msg_data.get("remaining") is not None:
        parts.append(f"- Ngân sách còn lại: {msg_data['remaining']:,}đ")
    if parts:
        return "\n".join(parts)
    item = ctx.get("item")
    amount = ctx.get("record_amount")
    if item or amount is not None:
        amount_str = f"{amount:,}đ" if amount is not None else ""
        return f"- Giao dịch: {item or 'không rõ'} {amount_str}".strip()
    return "(không có ngữ cảnh bổ đoán — bám sát câu người dùng)"


def _build_relationship_instruction(prompts: dict[str, Any], relationship_tag: str | None, nlg_persona: str) -> str:
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
        if nlg_persona in positive_emotions:
            return cfg.get("rule_happy", "")
        return cfg.get("rule_sad", "")
    return ""


def build_nlg_prompt(
    prompts: dict[str, Any],
    nlg_persona: str,
    nlu_result: dict[str, Any],
    context_metadata: dict[str, Any],
    chat_history: list | None = None,
    chat_summary: str | None = None,
) -> dict[str, Any]:
    """nlg_persona: key prompts.emotions (hai_huoc, dan_doi, ...) — không phải tên file PNG."""
    common = prompts.get("common", {})
    emotion_cfg = prompts.get("emotions", {}).get(nlg_persona, {})
    diversity_rule = common.get("context_diversity_rule", "")
    relationship_tag = nlu_result.get("relationship_tag")
    rel_instruction = _build_relationship_instruction(prompts, relationship_tag, nlg_persona)
    ctx_text = _format_context_meta(context_metadata)

    slang_list = emotion_cfg.get("slang_pool", [])
    slang_instruction = ""
    if slang_list:
        sampled_slang = random.sample(slang_list, min(3, len(slang_list)))
        slang_instruction = (
            f"[QUAN TRỌNG - PHONG CÁCH PHẢN HỒI]: Bạn PHẢI dùng phong cách '{nlg_persona}'. "
            f"Đặc biệt, HÃY CHỌN DÙNG NGẪU NHIÊN 1-2 TỪ LÓNG SAU ĐÂY: {', '.join(sampled_slang)}. "
            "LƯU Ý QUAN TRỌNG: TUYỆT ĐỐI KHÔNG lặp lại các từ lóng đã dùng ở các câu trước (ví dụ: không được dùng lại 'ét ô ét' liên tục). "
            "Hãy thay đổi từ vựng ngẫu nhiên để thể hiện cá tính thật rõ nét, KHÔNG trả lời chung chung an toàn."
        )

    system_parts = [emotion_cfg.get("system"), common.get("style"), common.get("response_rules")]
    if diversity_rule:
        system_parts.append(diversity_rule)
    if rel_instruction:
        system_parts.insert(1, f"[QUAN HỆ OVERRIDE] {rel_instruction}")
    if slang_instruction:
        system_parts.append(slang_instruction)

    system_prompt = " ".join(s for s in system_parts if s)

    intent = nlu_result.get("intent")
    meta_block = f"CONTEXT_META:\n{ctx_text}\n" if context_metadata else ""

    if intent == "Chitchat":
        rules = common.get("chitchat_response_rules") or common.get("response_rules")
        system_parts_chitchat = [emotion_cfg.get("system"), common.get("style"), diversity_rule, rules]
        if slang_instruction:
            system_parts_chitchat.append(slang_instruction)
        system_prompt = " ".join(s for s in system_parts_chitchat if s)
        base_user = common.get("chitchat_user") or emotion_cfg.get("user")
        user_msg = nlu_result.get("text") or ""
        user_prompt = (
            f"{base_user} {rules} "
            f"Câu người dùng: \"{user_msg}\". "
            "Đọc câu để trả lời đúng chủ đề; chọn mimo_emotion PascalCase phù hợp ngữ cảnh.\n"
            f"{meta_block}"
            f"{_MIMO_EMOTION_INSTRUCTION}"
        )

    elif intent == "Action":
        rules = common.get("action_response_rules") or common.get("response_rules")
        system_parts_action = [emotion_cfg.get("system"), common.get("style"), rules]
        if slang_instruction:
            system_parts_action.append(slang_instruction)
        system_prompt = " ".join(s for s in system_parts_action if s)
        base_user = common.get("action_user") or emotion_cfg.get("user")
        action_type = nlu_result.get("action_type")
        action_facts = context_metadata.get("action_facts")
        facts_block = ""
        if action_facts:
            facts_block = (
                f"\nACTION_FACTS (chỉ dùng số liệu dưới đây, không bịa thêm): "
                f"{json.dumps(action_facts, ensure_ascii=False)}\n"
            )
        user_prompt = (
            f"{base_user} {rules} "
            f"Loại thao tác: {action_type}. "
            "Không đặt tên món ăn giả; không thêm giao dịch không có trong input.\n"
            "mimo_emotion phải khớp ngữ cảnh: Success (hoàn tất/báo cáo), Worried hoặc Alert "
            "(cảnh báo chi tiêu), Celebrate (tin tốt), Approved (xác nhận). "
            f"{facts_block}"
            f"{meta_block}"
            f"{_MIMO_EMOTION_INSTRUCTION}"
        )

    else:
        base_user = emotion_cfg.get("user")
        item = nlu_result.get("item") or ""
        amount = nlu_result.get("amount")
        context_type = context_metadata.get("type") or "NONE"
        amount_str = f"{amount:,}đ" if amount is not None else "không rõ"
        record_rules = (
            common.get("record_income_rules")
            if _is_income_record(nlu_result)
            else common.get("record_expense_rules")
        ) or ""
        meta_label = f"CONTEXT_META (phối hợp ≥1 yếu tố):\n{ctx_text}\n" if context_metadata else ""
        user_prompt = (
            f"{base_user} {common.get('response_rules', '')} "
            f"{record_rules} "
            f"{_record_type_instruction(nlu_result)} "
            f"Món hoặc hạng mục: {item}. Số tiền: {amount_str}. "
            f"Kiểu cảnh báo (chỉ Expense): {context_type}. "
            f"{rel_instruction + ' ' if rel_instruction else ''}"
            f"{meta_label}"
            f"{_MIMO_EMOTION_INSTRUCTION}"
        )

    # Format chat history & summary & previous action context
    history_block = ""
    prev_action_context = ""
    
    if chat_history and isinstance(chat_history, list):
        history_msgs = chat_history[:-1] if chat_history[-1].get("role") == "user" else chat_history
        
        # Check for previous action context (search results or reports)
        for msg in reversed(history_msgs):
            intent_act = msg.get("intent_action") or {}
            nlu_data = intent_act.get("nlu") or {}
            action_res = nlu_data.get("action_result") or {}
            if action_res.get("kind") == "search" and action_res.get("items"):
                items = action_res["items"]
                formatted_items = []
                for idx, it in enumerate(items, 1):
                    formatted_items.append(f"[{idx}] {it.get('note')} {it.get('amount'):,}đ ({it.get('categoryCode')})")
                prev_action_context = f"KẾT QUẢ TÌM KIẾM GẦN NHẤT:\n" + "\n".join(formatted_items) + "\n\n"
                break
            elif action_res.get("kind") == "report":
                prev_action_context = f"BÁO CÁO CHI TIÊU GẦN NHẤT: {action_res.get('period_label')} - Tổng chi: {action_res.get('total_expense'):,}đ\n\n"
                break

        history_block += "LỊCH SỬ ĐỐI THOẠI GẦN ĐÂY:\n"
        for msg in history_msgs:
            role_label = "Người dùng" if msg.get("role") == "user" else "Mascot MiMo"
            content = msg.get("content", "")
            history_block += f"- {role_label}: {content}\n"
        history_block += "\n"

    if chat_summary:
        history_block = f"BỐI CẢNH TRÒ CHUYỆN TRƯỚC ĐÓ:\n- {chat_summary}\n\n" + history_block

    if prev_action_context:
        history_block = prev_action_context + history_block

    if history_block:
        user_prompt = history_block + user_prompt

    return {
        "system": system_prompt.strip(),
        "user": user_prompt.strip(),
        "input": {
            "nlu_result": nlu_result,
            "context_metadata": context_metadata,
            "nlg_persona": nlg_persona,
        },
    }
