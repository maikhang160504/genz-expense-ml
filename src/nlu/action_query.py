"""
Phát hiện câu hỏi báo cáo / tổng chi (intent Action) — chỉ dùng khi encoder chưa đủ tin.

Sau khi intent encoder retrain đạt smoke 20/20, có thể tắt bằng USE_ACTION_QUERY_GUARD=0.
"""
from __future__ import annotations

import os
import re
import unicodedata

def _norm(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


_MONEY_PATTERN = re.compile(
    r"\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|tr|triệu|trieu|củ|cu)\b",
    re.I,
)


_SUGGEST_KW = (
    "goi y chi tieu", "gợi ý chi tiêu", "goi y tiet kiem", "gợi ý tiết kiệm",
    "goi y ngan sach", "gợi ý ngân sách", "goi y han muc", "gợi ý hạn mức",
    "de xuat goi y", "đề xuất gợi ý", "xin goi y chi", "suggest budget",
)
_REPORT_KW = (
    "bao cao", "báo cáo", "thong ke", "thống kê", "tong chi", "tổng chi",
    "tieu bao nhieu", "tiêu bao nhieu", "da tieu bao nhieu", "đã tiêu bao nhiêu",
    "chi het bao nhieu", "chi hết bao nhiêu", "xem tong chi", "xem tổng chi",
)


def is_action_query(text: str) -> bool:
    t = _norm(text).replace("đ", "d")
    if any(k in t for k in _SUGGEST_KW):
        return True
    if any(k in t for k in _REPORT_KW):
        return True
    if re.search(r"\b(so sanh|so sánh)\b", t) and any(w in t for w in ["chi tieu", "chi tiêu", "tuan", "tuần", "thang", "tháng"]):
        return True
    return False


def is_limit_or_goal_action(text: str) -> bool:
    t = _norm(text).replace("đ", "d")
    
    # 1. Target nouns for budget/goal
    has_target = any(w in t for w in ["han muc", "gioi han", "ngan sach", "muc tieu", "tiet kiem"])
    has_verb = any(w in t for w in ["dat", "cai", "thiet lap", "chot", "dat lai", "them", "bot", "giam", "tang", "cong", "tru", "bo sung", "bu"])
    if has_target and has_verb:
        return True
        
    # 2. Action patterns: verb + amount + into/from + category
    has_mod_verb = any(t.startswith(w) or f" {w} " in t for w in ["them", "cong them", "tang them", "bo sung", "bot", "giam", "tru", "giam di", "tang"])
    has_money = bool(_MONEY_PATTERN.search(text))
    has_prep = any(w in t for w in [" vao ", " cho ", " tu ", " vao khoang "])
    
    if has_mod_verb and has_money and has_prep:
        if not any(w in t for w in ["mua", "order", "thanh toan", "dong", "nap", "book", "chi"]):
            return True
            
    return False


def suggest_budget_action_type(text: str) -> str | None:
    t = _norm(text).replace("đ", "d")
    if any(k in t for k in _SUGGEST_KW):
        return "SUGGEST_BUDGET"
    return None


def report_general_action_type(text: str) -> str | None:
    if suggest_budget_action_type(text):
        return None
    if not is_action_query(text):
        return None
    t = _norm(text).replace("đ", "d")
    if re.search(
        r"bao\s+cao\s+(?:an\s+uong|di\s+lai|mua\s+sam|"
        r"giai\s+tri|dien\s+nuoc|hoc\s+phi|y\s+te|"
        r"nha\s+cua)\b",
        t,
    ):
        return "Report"
    return "REPORT_GENERAL"
