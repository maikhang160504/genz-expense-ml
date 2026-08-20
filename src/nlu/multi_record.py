"""
Tách một câu Record thành nhiều bản ghi đề xuất (nhiều thực thể → nhiều dòng).

Chiến lược v1: tách theo dấu phẩy/chấm phẩy/xuống dòng/từ \"và\" rồi với mỗi đoạn có số tiền,
chạy NER + category + record_type như một câu đơn.

Chỉ trả về danh sách khi có **ít nhất 2** đoạn hợp lệ (mỗi đoạn có ≥1 số tiền).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    from pipeline.text_preprocessing import clean_category_text
except ImportError:
    try:
        from text_nlu.pipeline.text_preprocessing import clean_category_text
    except ImportError:
        # Auto-add text_nlu to sys.path
        _text_nlu_dir = Path(__file__).resolve().parent.parent.parent / "text_nlu"
        if str(_text_nlu_dir) not in sys.path:
            sys.path.insert(0, str(_text_nlu_dir))
        from pipeline.text_preprocessing import clean_category_text

from src.nlu.ner import (
    extract_ner_slots,
    map_product_brand_hint,
    select_record_item_from_slots,
)
from src.nlu.text import extract_amounts, clean_content
from src.nlu.models import predict_category_from_text

# Tách đoạn: tránh bẻ số thập phân kiểu 1,5 (hiếm trong chi tiêu) — ưu tiên ", " rõ ràng
_SEGMENT_RE = re.compile(
    r"(?:\s*,\s*|\s*;\s*|\s*\|\s*|\n\s*|\s+và\s+|\s+\+\s+)",
    re.IGNORECASE,
)

_TFIDF_CATEGORY_CACHE = None
_TFIDF_RECORD_TYPE_CACHE = None
_NER_MODEL_CACHE = None


def _get_fallback_category_model() -> dict:
    global _TFIDF_CATEGORY_CACHE
    if _TFIDF_CATEGORY_CACHE is None:
        try:
            from src.config import settings
            from src.nlu.models import _get_path, _load_tfidf, _load_encoder, use_category_encoder_runtime
            cat_enc = _get_path(settings.CATEGORY_ENCODER_PATH)
            if use_category_encoder_runtime() and cat_enc.is_file():
                _TFIDF_CATEGORY_CACHE = _load_encoder(cat_enc)
            else:
                cat_model = _get_path(settings.CATEGORY_MODEL_PATH)
                if cat_model.is_file():
                    _TFIDF_CATEGORY_CACHE = _load_tfidf(cat_model)
                else:
                    _TFIDF_CATEGORY_CACHE = {}
        except Exception:
            _TFIDF_CATEGORY_CACHE = {}
    return _TFIDF_CATEGORY_CACHE


def _get_fallback_record_type_model() -> dict:
    global _TFIDF_RECORD_TYPE_CACHE
    if _TFIDF_RECORD_TYPE_CACHE is None:
        try:
            from src.config import settings
            from src.nlu.models import _get_path, _load_tfidf, _load_encoder, use_intent_encoder_runtime
            rec_enc = _get_path(settings.RECORD_TYPE_ENCODER_PATH)
            if use_intent_encoder_runtime() and rec_enc.is_file():
                _TFIDF_RECORD_TYPE_CACHE = _load_encoder(rec_enc)
            else:
                rec_model = _get_path(settings.RECORD_TYPE_MODEL_PATH)
                if rec_model.is_file():
                    _TFIDF_RECORD_TYPE_CACHE = _load_tfidf(rec_model)
                else:
                    _TFIDF_RECORD_TYPE_CACHE = {}
        except Exception:
            _TFIDF_RECORD_TYPE_CACHE = {}
    return _TFIDF_RECORD_TYPE_CACHE


def _get_fallback_ner_model() -> Any:
    global _NER_MODEL_CACHE
    if _NER_MODEL_CACHE is None:
        try:
            from src.nlu.ner import load_ner_model
            _NER_MODEL_CACHE = load_ner_model()
        except Exception:
            _NER_MODEL_CACHE = None
    return _NER_MODEL_CACHE


def split_record_segments(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in _SEGMENT_RE.split(t) if p and p.strip()]
    return parts if len(parts) > 1 else [t]


def _record_type_for_segment(seg: str, record_type_model: dict | None) -> str | None:
    if not record_type_model or record_type_model.get("backend") == "llm":
        record_type_model = _get_fallback_record_type_model()
    if not record_type_model:
        return "Expense"
    if record_type_model.get("backend") == "encoder" and record_type_model.get("bundle"):
        from src.nlu.encoder_runtime import predict_record_type_encoder
        pred = predict_record_type_encoder(record_type_model["bundle"], seg)
        return "Income" if pred == "income" else "Expense"
    elif record_type_model.get("backend") == "tfidf" and record_type_model.get("vectorizer"):
        rec_vec = record_type_model["vectorizer"].transform([clean_category_text(seg)])
        rec_pred = str(record_type_model["model"].predict(rec_vec)[0]).lower()
        return "Income" if rec_pred == "income" else "Expense"
    return "Expense"


def _category_for_segment(seg: str, category_model: dict | None) -> Any:
    if not category_model or category_model.get("backend") == "llm":
        category_model = _get_fallback_category_model()
    if not category_model:
        return None
    if category_model.get("backend") == "encoder" and category_model.get("bundle"):
        from src.nlu.encoder_runtime import predict_category_encoder
        return predict_category_encoder(category_model["bundle"], seg)
    elif category_model.get("backend") == "tfidf" and category_model.get("vectorizer"):
        cat_vec = category_model["vectorizer"].transform([clean_category_text(seg)])
        return category_model["model"].predict(cat_vec)[0]
    return None


def extract_multi_records(
    user_text: str,
    ner_model: Any = None,
    category_model: dict | None = None,
    record_type_model: dict | None = None,
) -> list[dict]:
    """
    Trả về [] nếu không phát hiện ≥2 khoản chi/thu rõ từ các đoạn tách được.
    Mỗi phần tử: segment, text, amount, category, item, category_ner, record_type.
    """
    segments = split_record_segments(user_text)
    if len(segments) < 2:
        return []

    if ner_model is None:
        ner_model = _get_fallback_ner_model()
    if category_model is None or category_model.get("backend") == "llm":
        category_model = _get_fallback_category_model()
    if record_type_model is None or record_type_model.get("backend") == "llm":
        record_type_model = _get_fallback_record_type_model()

    rows: list[dict] = []
    for seg in segments:
        amounts = extract_amounts(seg)
        if not amounts:
            continue
        amount = amounts[0]
        slots = extract_ner_slots(seg, ner_model) if ner_model else None
        ner_item = select_record_item_from_slots(slots) if slots else None
        if not ner_item:
            ner_item = clean_content(seg)

        mapped = predict_category_from_text(ner_item, category_model) if category_model else None
        cat = _category_for_segment(seg, category_model)
        rt = _record_type_for_segment(seg, record_type_model)
        final_cat = cat or mapped or "Other"

        rows.append(
            {
                "segment": seg,
                "text": seg,
                "amount": amount,
                "category": final_cat,
                "item": ner_item,
                "category_ner": mapped,
                "record_type": rt or "Expense",
            }
        )

    return rows if len(rows) >= 2 else []
