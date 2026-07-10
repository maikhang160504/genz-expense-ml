"""
DEPRECATED — runtime inference không dùng keyword guard nữa.

Giữ file để tránh import lỗi; mọi hàm trả về False/None.
Intent/action_type/slot do model quyết định (intent_model, action_type_model, action_slots_model).
"""
from __future__ import annotations


def is_action_query(text: str) -> bool:
    return False


def is_limit_or_goal_action(text: str) -> bool:
    return False


def suggest_budget_action_type(text: str) -> str | None:
    return None


def report_general_action_type(text: str) -> str | None:
    return None


def is_system_or_delete_action(text: str) -> bool:
    return False
