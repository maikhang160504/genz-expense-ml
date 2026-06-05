"""
Phát hiện câu hỏi báo cáo / tổng chi (intent Action) — chỉ dùng khi encoder chưa đủ tin.

Sau khi intent encoder retrain đạt smoke 20/20, có thể tắt bằng USE_ACTION_QUERY_GUARD=0.
"""
from __future__ import annotations

import os
import re
import unicodedata

_EXPENSE_RECORD = re.compile(
    r"^(mua|order|thanh\s+toán|thanh\s+toan|đóng|dong|nạp|nap|book)\b",
    re.I,
)
_CHI_WITH_MONEY = re.compile(r"^chi\s+.+\d", re.I)
_ACTION_QUERY = re.compile(
    r"(?:"
    r"tổng\s+chi|tong\s+chi|"
    r"(?:tháng|thang|tuần|tuan|hôm|hom|quý|quy)\s+nay\b.+(?:tiêu|tieu|chi\b).*(?:bao\s+nhiêu|bao\s+nhieu|hết|het|mấy|may|roi|rồi)|"
    r"(?:tiêu|tieu)\s+bao\s+nhiêu|"
    r"(?:đã|da)\s+(?:tiêu|tieu)|"
    r"thống\s+kê\s+chi|thong\s+ke\s+chi|"
    r"xem\s+(?:tổng\s+chi|tong\s+chi|báo\s+cáo\s+tổng|bao\s+cao\s+tong)|"
    r"chi\s+tiêu\s+(?:tháng|thang|tuần|tuan)|"
    r"mình\s+đã\s+tiêu|minh\s+da\s+tieu|"
    r"tiêu\s+hết\s+bao\s+nhiêu|tieu\s+het\s+bao\s+nhiêu"
    r")",
    re.I | re.UNICODE,
)

def _norm(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


_MONEY_PATTERN = re.compile(
    r"\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|tr|triệu|trieu|củ|cu)\b",
    re.I,
)


def is_action_query(text: str) -> bool:
    if os.environ.get("USE_ACTION_QUERY_GUARD", "1") != "1":
        return False
    t = _norm(text)
    if not t:
        return False
    if _MONEY_PATTERN.search(t):
        return False
    if _EXPENSE_RECORD.search(t):
        return False
    if _CHI_WITH_MONEY.search(t):
        return False
    if re.search(r"\b(me|mẹ|ma|bo|bố)\s+cho\b", t):
        return False
    return bool(_ACTION_QUERY.search(t))


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


def report_general_action_type(text: str) -> str | None:
    if not is_action_query(text):
        return None
    t = _norm(text)
    if re.search(
        r"báo\s+cáo\s+(?:ăn\s+uống|an\s+uong|đi\s+lại|di\s+lai|mua\s+sắm|mua\s+sam|"
        r"giải\s+trí|giai\s+tri|điện\s+nước|dien\s+nuoc|học\s+phí|hoc\s+phi|y\s+tế|y\s+te|"
        r"nhà\s+cửa|nha\s+cua)\b",
        t,
    ):
        return "Report"
    return "REPORT_GENERAL"
