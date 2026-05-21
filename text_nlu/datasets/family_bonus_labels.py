"""
Nhãn Bonus cho thu từ người thân (mẹ cho, gia đình, ông bà, cô chú…).

Dùng chung: append_income_short_rows, fix_family_bonus_labels.
"""
from __future__ import annotations

import re
import unicodedata

# Thu từ người thân → Bonus (không phải Business)
FAMILY_INCOME_TEMPLATES: list[tuple[str, str]] = [
    ("mẹ cho {m}", "Bonus"),
    ("me cho {m}", "Bonus"),
    ("ma cho {m}", "Bonus"),
    ("bố cho {m}", "Bonus"),
    ("bo cho {m}", "Bonus"),
    ("ba cho {m}", "Bonus"),
    ("ông cho {m}", "Bonus"),
    ("ong cho {m}", "Bonus"),
    ("bà cho {m}", "Bonus"),
    ("anh cho {m}", "Bonus"),
    ("chị cho {m}", "Bonus"),
    ("chi cho {m}", "Bonus"),
    ("em cho {m}", "Bonus"),
    ("ảnh cho {m}", "Bonus"),
    ("cô cho {m}", "Bonus"),
    ("co cho {m}", "Bonus"),
    ("chú cho {m}", "Bonus"),
    ("chu cho {m}", "Bonus"),
    ("dì cho {m}", "Bonus"),
    ("di cho {m}", "Bonus"),
    ("bác cho {m}", "Bonus"),
    ("bac cho {m}", "Bonus"),
    ("ông bà cho {m}", "Bonus"),
    ("ong ba cho {m}", "Bonus"),
    ("cha mẹ cho {m}", "Bonus"),
    ("cha me cho {m}", "Bonus"),
    ("gia đình cho {m}", "Bonus"),
    ("gia dinh cho {m}", "Bonus"),
    ("gđ cho {m}", "Bonus"),
    ("gd cho {m}", "Bonus"),
    ("mẹ ck {m}", "Bonus"),
    ("me ck {m}", "Bonus"),
    ("ma ck {m}", "Bonus"),
    ("bố ck {m}", "Bonus"),
    ("bo ck {m}", "Bonus"),
    ("mẹ gửi {m}", "Bonus"),
    ("me gui {m}", "Bonus"),
    ("bố gửi {m}", "Bonus"),
    ("bo gui {m}", "Bonus"),
    ("bạn cho {m}", "Bonus"),
    ("ban cho {m}", "Bonus"),
    ("được mẹ cho {m}", "Bonus"),
    ("duoc me cho {m}", "Bonus"),
    ("được bố cho {m}", "Bonus"),
    ("duoc bo cho {m}", "Bonus"),
]

_FAMILY_BONUS_RE = re.compile(
    r"(?:duoc|được)\s+(?:me|mẹ|ma|bo|bố|ong|ông|anh|chi|chị)\s+cho\b|"
    r"(?:gia\s+dinh|gia\s+đình|gđ|gd)\s+cho\b|"
    r"(?:me|mẹ|ma|bo|bố|ba|ong|ông|anh|chi|chị|em|co|cô|chu|chú|di|dì|bac|bác|ảnh)\s+cho\b|"
    r"(?:ong\s+ba|ông\s+bà|cha\s+me|cha\s+mẹ)\s+cho\b|"
    r"(?:me|mẹ|ma|bo|bố|ba|ong|ông)\s+(?:ck|gui|gửi)\b|"
    r"(?:ban|bạn)\s+cho\b",
    re.IGNORECASE | re.UNICODE,
)


def normalize_text(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def is_family_gift_income(text: str) -> bool:
    """True nếu câu mô tả người thân cho/tặng/chuyển tiền (income)."""
    t = normalize_text(text)
    if not _FAMILY_BONUS_RE.search(t):
        return False
    # Tránh false positive: «cho xin», «đi chơi», «chốt kèo»
    if re.search(r"\bcho\s+xin\b|\bdi\s+choi\b|\bchot\s+keo\b", t):
        return False
    if re.search(r"\bmua\s+.*\s+cho\b", t):
        return False
    return True
