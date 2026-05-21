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


def is_action_query(text: str) -> bool:
    if os.environ.get("USE_ACTION_QUERY_GUARD", "1") != "1":
        return False
    t = _norm(text)
    if not t:
        return False
    if _EXPENSE_RECORD.search(t):
        return False
    if _CHI_WITH_MONEY.search(t):
        return False
    if re.search(r"\b(me|mẹ|ma|bo|bố)\s+cho\b", t):
        return False
    return bool(_ACTION_QUERY.search(t))


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
