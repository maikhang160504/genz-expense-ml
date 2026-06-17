"""Cụm thu nhập rõ (lương về, lg tháng…) — guard record_type khi model nhầm Expense."""
from __future__ import annotations

import os
import re

import unicodedata

def _no_accent(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )

INCOME_KEYWORDS = {
    "luong", "salary", "bonus", "thuong", "hoan tien", "refund",
    "me cho", "ba cho", "bo cho", "cha cho", "duoc tang", "duoc cho",
    "lai tiet kiem", "co tuc", "ban co phieu", "ting ting"
}

def is_clear_income_phrase(text: str) -> bool:
    norm = _no_accent(text.lower())
    return any(kw in norm for kw in INCOME_KEYWORDS)
