"""Cụm thu nhập rõ (lương về, lg tháng…) — guard record_type khi model nhầm Expense."""
from __future__ import annotations

import os
import re

_INCOME = re.compile(
    r"(?:^|\b)(?:"
    r"lương\s+tháng\s+về|luong\s+thang\s+ve|"
    r"lương\s+về|luong\s+ve|"
    r"^lg\s+.*\s+(?:về|ve)\b|"
    r"^nhận\s+lương|^nhan\s+luong|"
    r"^lương\s+part|^luong\s+part"
    r")",
    re.I,
)
_EXPENSE_START = re.compile(r"^(mua|chi|order|thanh\s+toán)\b", re.I)


def is_clear_income_phrase(text: str) -> bool:
    if os.environ.get("USE_INCOME_PHRASE_GUARD", "1") != "1":
        return False
    t = text.strip()
    if _EXPENSE_START.search(t):
        return False
    return bool(_INCOME.search(t))
