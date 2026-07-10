"""Schema cột slot cho intent_action.csv — khớp action.md."""

from __future__ import annotations

SLOT_COLUMNS: list[str] = [
    "verb",
    "category_code",
    "value",
    "goal_name",
    "enabled",
    "theme",
    "verbal_style",
    "time_range",
    "query",
    "note",
]

BASE_COLUMNS: list[str] = ["text", "intent", "action_type"]

ALL_COLUMNS: list[str] = BASE_COLUMNS + SLOT_COLUMNS

# action_type → cột slot cần huấn luyện / dự đoán
SLOTS_BY_ACTION: dict[str, list[str]] = {
    "SET_LIMIT": ["verb", "category_code", "value"],
    "SET_ALERT": ["category_code", "enabled"],
    "SET_GOAL": ["goal_name", "value"],
    "ADD_GOAL": ["verb", "goal_name", "value"],
    "SYSTEM_SETTING": ["theme"],
    "SET_TONE": ["verbal_style"],
    "SET_USERNAME": ["value"],
    "SEARCH_RECORD": ["query", "category_code", "value"],
    "REPORT_GENERAL": ["time_range", "category_code"],
    "SUGGEST_BUDGET": ["time_range"],
}

VERB_LABELS = frozenset({"SET", "ADD", "SUB"})
