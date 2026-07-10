"""
Demo executor: theo ``action_type`` in ra thao tác sẽ gọi DB/API (không kết nối thật).
Production: thay phần ``print`` bằng repository / HTTP client.

Đồng bộ với action.md — đã xóa UPDATE_RECORD, DELETE_RECORD, Edit.
Đã bổ sung categoryCode cho REPORT_GENERAL, cải thiện SEARCH_RECORD.
"""
from __future__ import annotations

from typing import Any


def _line(msg: str) -> str:
    return f"[EXECUTOR-DEMO] {msg}"


def describe_action_execution(nlu_result: dict[str, Any]) -> list[str]:
    """
    Trả về danh sách dòng mô tả hành động (chỉ in/log khi demo).

    ``nlu_result`` nên chứa ít nhất: intent, action_type, action_param, action_details (tuỳ có).
    """
    if nlu_result.get("intent") != "Action":
        return []

    action_type = nlu_result.get("action_type")
    if str(action_type) == "SYSTEM_SETTING":
        action_type = "Setting"
    if not action_type:
        return [_line("Không có action_type — bỏ qua.")]

    param = nlu_result.get("action_param")
    details = nlu_result.get("action_details") or {}
    text = nlu_result.get("text", "")
    time_range = nlu_result.get("time_range")

    # ── REPORT_GENERAL: Báo cáo chi tiêu theo thời gian + danh mục ──
    if action_type in ("Report", "REPORT_GENERAL"):
        category = details.get("target") or details.get("category_code")
        parts = ["Báo cáo / tổng hợp chi tiêu"]
        if time_range:
            parts.append(f"theo kỳ: {time_range}")
        if category:
            parts.append(f"danh mục: {category}")
        parts.append("(API GET /reports/summary)")
        msg = " — ".join(parts)

    # ── SEARCH_RECORD: Tìm kiếm giao dịch multi-filter ──
    elif action_type in ("Search", "SEARCH_RECORD"):
        query = details.get("query") or ""
        category = details.get("target") or details.get("category_code")
        min_amount = param
        filters = []
        if query:
            filters.append(f"query={query!r}")
        if category:
            filters.append(f"categoryCode={category}")
        if min_amount:
            filters.append(f"minAmount={min_amount}")
        if time_range:
            filters.append(f"time_range={time_range}")
        filter_str = ", ".join(filters) if filters else "không có filter cụ thể"
        msg = f"Tìm kiếm danh sách giao dịch [{filter_str}] (API GET /records/search)"

    # ── REPORT_COMPARE: So sánh hai kỳ ──
    elif action_type == "REPORT_COMPARE":
        msg = f"So sánh chi tiêu hai kỳ (API GET /reports/compare), câu gốc: {text[:80]!r}"

    # ── SET_LIMIT: Đặt/thay đổi hạn mức chi tiêu ──
    elif action_type == "SET_LIMIT":
        verb = details.get("verb", "SET")
        category = details.get("target") or details.get("category_code")
        verb_desc = {"SET": "Đặt mới", "ADD": "Cộng thêm", "SUB": "Giảm bớt"}.get(
            str(verb).upper(), str(verb)
        )
        msg = f"{verb_desc} hạn mức"
        if category:
            msg += f" danh mục [{category}]"
        if param:
            msg += f", số tiền: {param}"
        msg += " (API PUT /budgets/limit)"

    # ── SET_GOAL / ADD_GOAL: Mục tiêu tiết kiệm ──
    elif action_type in ("SET_GOAL", "ADD_GOAL"):
        goal_name = details.get("goal_name") or details.get("target")
        if action_type == "ADD_GOAL":
            msg = f"Thêm mục tiêu tiết kiệm mới"
        else:
            msg = f"Đặt mục tiêu tiết kiệm"
        if goal_name:
            msg += f" [{goal_name}]"
        if param:
            msg += f", giá trị: {param}"
        msg += f" (API {'POST' if action_type == 'ADD_GOAL' else 'PUT'} /goals)"

    # ── SET_TONE: Đổi giọng nói mascot ──
    elif action_type == "SET_TONE":
        style = details.get("verbal_style") or details.get("style")
        msg = f"Đổi giọng NLG (API PUT /user_prefs/tone)"
        if style:
            msg += f", phong cách: {style}"

    # ── SET_USERNAME: Đổi tên người dùng ──
    elif action_type == "SET_USERNAME":
        name = details.get("name") or details.get("value") or param
        msg = f"Đổi tên hiển thị (API PUT /user_prefs/name)"
        if name:
            msg += f": {name}"

    # ── SET_ALERT: Bật/tắt cảnh báo hạn mức ──
    elif action_type == "SET_ALERT":
        enabled = details.get("enabled")
        category = details.get("target") or details.get("category_code")
        action = "Bật" if str(enabled).lower() in ("true", "1", "on") else "Tắt" if enabled is not None else "Cập nhật"
        msg = f"{action} cảnh báo vượt hạn mức"
        if category:
            msg += f" cho [{category}]"
        msg += " (API PUT /alerts)"

    # ── SET_INCOME: Cập nhật thu nhập cố định ──
    elif action_type == "SET_INCOME":
        msg = f"Cập nhật thu nhập cố định (API PUT /income_profile)"
        if param:
            msg += f", số: {param}"

    # ── SUGGEST_BUDGET: Gợi ý ngân sách ──
    elif action_type == "SUGGEST_BUDGET":
        target_month = details.get("time") or time_range
        msg = f"Gợi ý ngân sách chi tiêu (API GET /budgets/suggest)"
        if target_month:
            msg += f", tháng: {target_month}"

    # ── SYSTEM_SETTING / Setting: Mở cài đặt / đổi giao diện ──
    elif action_type in ("Setting", "SYSTEM_SETTING"):
        theme = details.get("theme")
        msg = "Cài đặt hệ thống / theme / ngôn ngữ (API PUT /system_prefs)"
        if theme:
            msg += f", theme: {theme}"

    # ── Fallback cho action_type chưa map ──
    else:
        msg = (
            f"Hành động [{action_type}] chưa map chi tiết — "
            f"gọi API generic /actions/{{type}} với body suy ra từ câu: {text[:100]!r}"
        )

    lines = [_line(msg)]
    if details:
        lines.append(_line(f"Chi tiết parse (demo): {details}"))
    if param is not None:
        lines.append(_line(f"Số tiền / tham số trích từ câu: {param}"))
    return lines


def print_action_execution(nlu_result: dict[str, Any]) -> None:
    for ln in describe_action_execution(nlu_result):
        print(ln)
