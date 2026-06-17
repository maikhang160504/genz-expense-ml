"""
Tách một câu Record thành nhiều bản ghi đề xuất (nhiều thực thể → nhiều dòng).

Chiến lược v1: tách theo dấu phẩy/chấm phẩy/xuống dòng/từ \"và\" rồi với mỗi đoạn có số tiền,
chạy NER + category + record_type như một câu đơn.

Chỉ trả về danh sách khi có **ít nhất 2** đoạn hợp lệ (mỗi đoạn có ≥1 số tiền).
"""
from __future__ import annotations

import re
from typing import Any

from pipeline.text_preprocessing import clean_category_text

from src.nlu.ner import (
    extract_ner_slots,
    map_product_brand_hint,
    select_record_item_from_slots,
)
from src.nlu.text import extract_amounts
from src.nlu.models import predict_category_from_text

# Tách đoạn: tránh bẻ số thập phân kiểu 1,5 (hiếm trong chi tiêu) — ưu tiên ", " rõ ràng
_SEGMENT_RE = re.compile(
    r"(?:\s*,\s*|\s*;\s*|\s*\|\s*|\n\s*|\s+và\s+|\s+\+\s+)",
    re.IGNORECASE,
)


def split_record_segments(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in _SEGMENT_RE.split(t) if p and p.strip()]
    return parts if len(parts) > 1 else [t]


def _record_type_for_segment(seg: str, record_type_model: dict) -> str | None:
    if record_type_model.get("backend") == "encoder" and record_type_model.get("bundle"):
        from src.nlu.encoder_runtime import predict_record_type_encoder
        pred = predict_record_type_encoder(record_type_model["bundle"], seg)
        return "Income" if pred == "income" else "Expense"
    elif record_type_model.get("backend") == "tfidf" and record_type_model.get("vectorizer"):
        rec_vec = record_type_model["vectorizer"].transform([clean_category_text(seg)])
        rec_pred = str(record_type_model["model"].predict(rec_vec)[0]).lower()
        return "Income" if rec_pred == "income" else "Expense"
    return None


def _category_for_segment(seg: str, category_model: dict) -> Any:
    if category_model.get("backend") == "encoder" and category_model.get("bundle"):
        from src.nlu.encoder_runtime import predict_category_encoder
        return predict_category_encoder(category_model["bundle"], seg)
    elif category_model.get("backend") == "tfidf" and category_model.get("vectorizer"):
        cat_vec = category_model["vectorizer"].transform([clean_category_text(seg)])
        return category_model["model"].predict(cat_vec)[0]
    return None


def extract_multi_records(
    user_text: str,
    ner_model: Any,
    category_model: dict,
    record_type_model: dict,
) -> list[dict]:
    """
    Trả về [] nếu không phát hiện ≥2 khoản chi/hợp nhật rõ từ các đoạn tách được.
    Mỗi phần tử: segment, amount, category, item, category_ner, record_type.
    """
    segments = split_record_segments(user_text)
    rows: list[dict] = []
    for seg in segments:
        amounts = extract_amounts(seg)
        if not amounts:
            continue
        amount = amounts[0]
        slots = extract_ner_slots(seg, ner_model) if ner_model else None
        ner_item = select_record_item_from_slots(slots) if slots else None
        mapped = predict_category_from_text(ner_item, category_model)

        cat = _category_for_segment(seg, category_model)
        rt = _record_type_for_segment(seg, record_type_model)

        rows.append(
            {
                "segment": seg,
                "amount": amount,
                "category": cat,
                "item": ner_item,
                "category_ner": mapped,
                "record_type": rt,
            }
        )

    return rows if len(rows) >= 2 else []
