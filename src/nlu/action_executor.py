"""
Demo executor: theo ``action_type`` in ra thao tác sẽ gọi DB/API (không kết nối thật).
Production: thay phần ``print`` bằng repository / HTTP client.
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

    handlers: dict[str, str] = {
        "Report": f"Báo cáo / tổng hợp chi tiêu theo yêu cầu trong câu: {text[:80]!r}…",
        "REPORT_GENERAL": "Truy vấn tổng chi + filter theo kỳ (API GET /reports/summary).",
        "REPORT_COMPARE": "So sánh hai kỳ (API GET /reports/compare).",
        "Edit": "Mở form / PATCH bản ghi đang chọn (API PATCH /records/{id}).",
        "UPDATE_RECORD": "Cập nhật trường amount/category của một giao dịch (API PATCH /records/{id}).",
        "Search": "Lọc danh sách giao dịch theo điều kiện tìm (API GET /records?query=…).",
        "SEARCH_RECORD": "Tìm kiếm full-text + filter số tiền (API GET /records/search).",
        "Setting": "Cập nhật cấu hình người dùng hoặc hạn mức (API PUT /settings).",
        "SET_LIMIT": f"Đặt hạn mức danh mục (API PUT /budgets/limit), tham số số tiền: {param}.",
        "SET_GOAL": f"Đặt mục tiêu tiết kiệm (API PUT /goals), giá trị: {param}.",
        "ADD_GOAL": "Thêm mục tiêu mới (API POST /goals).",
        "SET_ALERT": "Bật/tắt cảnh báo ngưỡng chi (API PUT /alerts).",
        "SET_TONE": f"Đổi giọng NLG (API PUT /user_prefs/tone), chi tiết: {details.get('style')}.",
        "SET_USERNAME": f"Đổi tên hiển thị (API PUT /user_prefs/name): {details.get('name')}.",
        "SET_INCOME": f"Cập nhật thu nhập cố định (API PUT /income_profile), số: {param}.",
        "SYSTEM_SETTING": "Cài đặt hệ thống / theme / ngôn ngữ (API PUT /system_prefs).",
        "EXPORT_DATA": "Xuất CSV/Excel (API POST /export).",
        "DELETE_RECORD": "Xóa giao dịch gần nhất hoặc theo id (API DELETE /records/last).",
    }

    msg = handlers.get(str(action_type))
    if msg is None:
        msg = f"Hành động chưa map chi tiết — gọi API generic /actions/{{type}} với body suy ra từ câu: {text[:100]!r}"

    lines = [_line(msg)]
    if details:
        lines.append(_line(f"Chi tiết parse (demo): {details}"))
    if param is not None:
        lines.append(_line(f"Số tiền / tham số trích từ câu: {param}"))
    return lines


def print_action_execution(nlu_result: dict[str, Any]) -> None:
    for ln in describe_action_execution(nlu_result):
        print(ln)
