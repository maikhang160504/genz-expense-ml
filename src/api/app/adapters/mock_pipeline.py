"""Lightweight regex + keyword fallback used when real models are not loaded.

Goal: keep the service usable for demos even on machines without the heavy
PaddleOCR / PhoBERT artifacts. Output shape matches the real pipeline so the
service layer does not need to branch downstream.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


_MONEY_RE = re.compile(
    r"(\d+(?:[\.,]\d+)?)\s?(k|đ|d|vnđ|vnd|ngan|nghìn|nghin|triệu|trieu|củ|cu|tr)?",
    re.IGNORECASE,
)


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Food": (
        "an ", "ăn", "phở", "pho", "bun", "bún", "com", "cơm", "trà sữa", "tra sua",
        "cafe", "cà phê", "ca phe", "highlands", "phuc long", "starbucks",
        "grabfood", "ship đồ ăn", "ship do an", "banh", "bánh", "kem", "sữa",
    ),
    "Shopping": (
        "ao", "áo", "quan", "quần", "giày", "giay", "tui xach", "túi xách",
        "điện thoại", "dien thoai", "laptop", "tai nghe", "shopee", "lazada",
        "tiki",
    ),
    "Essentials": (
        "rau", "thịt", "thit", "gạo", "gao", "đi chợ", "di cho", "siêu thị",
        "sieu thi", "vinmart", "bach hoa", "bách hóa", "giấy vệ sinh",
        "giay ve sinh", "xà phòng", "nước giặt",
    ),
    "Transport": (
        "xăng", "xang", "grab bike", "grabbike", "grab car", "be ", "be xe",
        "taxi", "xe ôm", "xe om", "vé xe", "ve xe", "gửi xe", "gui xe",
    ),
    "Housing": (
        "tiền điện", "tien dien", "tiền nước", "tien nuoc", "internet", "wifi",
        "thuê nhà", "thue nha", "tiền nhà",
    ),
    "Health": (
        "thuốc", "thuoc", "khám bệnh", "kham benh", "bệnh viện", "benh vien",
        "nha thuoc", "nhà thuốc", "y tế",
    ),
    "Beauty": ("mỹ phẩm", "my pham", "son môi", "kem dưỡng", "dầu gội"),
    "Entertainment": (
        "netflix", "spotify", "xem phim", "cgv", "game", "youtube premium",
    ),
    "Education": ("học phí", "hoc phi", "sách", "sach", "khóa học", "khoa hoc"),
    "Social": ("quà", "qua tang", "sinh nhật", "sinh nhat", "đám cưới", "dam cuoi"),
}


_INCOME_KEYWORDS = (
    "lương", "luong", "thưởng", "thuong", "được cho", "duoc cho", "mẹ cho",
    "me cho", "ba cho", "bố cho", "bo cho", "tiền bán", "tien ban", "thu nhập",
    "thu nhap", "freelance", "nhận tiền", "nhan tien",
)


_ACTION_KEYWORDS = (
    "đặt", "dat", "thiết lập", "thiet lap", "đổi", "doi", "báo cáo", "bao cao",
    "tổng chi", "tong chi", "thống kê", "thong ke", "xem chi tiêu",
    "tổng thu", "tong thu", "thu nhập", "thu nhap", "tổng tiền gửi", "tong tien gui",
    "tổng tiền rút", "tong tien rut", "tiền gửi", "tien gui", "tiền rút", "tien rut",
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def _parse_amount_token(num: str, unit: str | None) -> float | None:
    try:
        value = float(num.replace(",", "."))
    except ValueError:
        return None
    unit = (unit or "").lower()
    if unit in ("k", "nghin", "nghìn", "ngan"):
        value *= 1_000
    elif unit in ("triệu", "trieu", "củ", "cu", "tr"):
        value *= 1_000_000
    if value < 1:
        return None
    return value


def extract_amounts_mock(text: str) -> list[float]:
    out: list[float] = []
    for m in _MONEY_RE.finditer(text):
        num, unit = m.group(1), m.group(2)
        val = _parse_amount_token(num, unit)
        if val is not None and val >= 1000:
            out.append(val)
    return out


def classify_category_mock(text: str) -> tuple[str, float]:
    norm = _normalize(text)
    scores: dict[str, int] = {}
    for cat, keys in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for k in keys if _normalize(k) in norm)
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "Others", 0.4
    total = sum(scores.values()) or 1
    return best, min(0.95, 0.55 + 0.1 * scores[best] / total + 0.05 * scores[best])


def classify_intent_mock(text: str) -> tuple[str, float]:
    norm = _normalize(text)
    has_money = bool(_MONEY_RE.search(text)) and extract_amounts_mock(text)
    if any(kw in norm for kw in (_normalize(k) for k in _ACTION_KEYWORDS)):
        return "Action", 0.7
    if has_money:
        return "Record", 0.85
    if len(norm.split()) <= 5 and not has_money:
        return "Chitchat", 0.6
    return "Chitchat", 0.5


def detect_record_type_mock(text: str) -> str:
    norm = _normalize(text)
    if any(_normalize(k) in norm for k in _INCOME_KEYWORDS):
        return "Income"
    return "Expense"


@dataclass
class MockNLUResult:
    text: str
    intent: str
    intent_confidence: float
    amount: float | None
    category: str | None
    record_type: str | None
    category_confidence: float
    multi_amounts: list[float]
    action_type: str | None = None
    time_range: dict | None = None


def run_nlu_mock(text: str) -> MockNLUResult:
    intent, intent_conf = classify_intent_mock(text)
    amounts = extract_amounts_mock(text)
    category, cat_conf = classify_category_mock(text)
    record_type = detect_record_type_mock(text) if intent == "Record" else None
    action_type = None
    if intent == "Action":
        norm = _normalize(text)
        if any(k in norm for k in ("tong thu nhap", "thu nhap", "tong thu")):
            action_type = "REPORT_INCOME"
        elif any(k in norm for k in ("tien gui", "tong tien gui", "gui tien")):
            action_type = "REPORT_SAVINGS"
        elif any(k in norm for k in ("tien rut", "tong tien rut", "rut tien")):
            action_type = "REPORT_SAVINGS"
        elif any(k in norm for k in ("tong chi", "bao cao", "thong ke", "xem chi tieu")):
            action_type = "REPORT_GENERAL"
        else:
            action_type = "REPORT_GENERAL"
    return MockNLUResult(
        text=text,
        intent=intent,
        intent_confidence=intent_conf,
        amount=amounts[0] if amounts else None,
        category=category if intent == "Record" else None,
        record_type=record_type,
        category_confidence=cat_conf,
        multi_amounts=amounts[1:] if len(amounts) > 1 else [],
        action_type=action_type,
        time_range=None,
    )


def mock_ocr_from_text(text: str) -> dict[str, Any]:
    """OCR mock: pretend we already extracted text and run rule-based summary."""
    amounts = extract_amounts_mock(text)
    category, conf = classify_category_mock(text)
    return {
        "text": text,
        "lines": [{"text": line.strip(), "bbox": None, "confidence": None}
                  for line in text.splitlines() if line.strip()],
        "suggestion": {
            "amount": max(amounts) if amounts else None,
            "category": category,
            "confidence": conf if amounts else 0.3,
            "currency": "VND",
        },
        "requires_confirmation": True,
        "backend": "mock",
    }
