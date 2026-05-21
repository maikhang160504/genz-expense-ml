from __future__ import annotations

from typing import Any

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

from .text import normalize_text

CATEGORY_LABELS = {
    "ăn uống",
    "đi lại",
    "mua sắm",
    "giải trí",
    "điện nước",
    "cà phê",
    "xăng xe",
    "học phí",
    "thuê nhà",
    "y tế",
    "gia đình",
    "đầu tư",
    "linh tinh",
    "xem phim",
    "mỹ phẩm",
    "quà tặng",
    "sửa xe",
    "đi chợ",
    "du lịch",
    "cắt tóc",
    "điện thoại",
    "internet",
    "bảo hiểm",
    "nội thất",
    "hội họp",
}

CATEGORY_MAP = {
    "ăn uống": "Food",
    "cà phê": "Food",
    "đi chợ": "Essentials",
    "gia đình": "Social",
    "đi lại": "Transport",
    "xăng xe": "Transport",
    "sửa xe": "Transport",
    "mua sắm": "Shopping",
    "điện thoại": "Shopping",
    "nội thất": "Housing",
    "thuê nhà": "Housing",
    "điện nước": "Housing",
    "internet": "Housing",
    "y tế": "Health",
    "mỹ phẩm": "Beauty",
    "cắt tóc": "Beauty",
    "học phí": "Education",
    "hội họp": "Social",
    "quà tặng": "Social",
    "du lịch": "Entertainment",
    "xem phim": "Entertainment",
    "giải trí": "Entertainment",
    "đầu tư": "Investment",
    "bảo hiểm": "Investment",
    "linh tinh": "Others",
}

CATEGORY_STOPWORDS = {
    "đi",
    "ăn",
    "đi ăn",
}

PRODUCT_BRAND_HINT = {
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "youtube": "Entertainment",
    "chatgpt": "Others",
    "momo": "Shopping",
    "zalopay": "Shopping",
    "shopee": "Shopping",
    "grab": "Transport",
    "grabfood": "Food",
    "temu": "Shopping",
    "lazada": "Shopping",
    "canva": "Shopping",
    "icloud": "Housing",
    "capcut": "Entertainment",
    "viettel": "Housing",
    "garena": "Entertainment",
}


def map_product_brand_hint(name: str | None) -> str | None:
    if not name:
        return None
    k = normalize_text(name).lower()
    for key, lab in PRODUCT_BRAND_HINT.items():
        if key in k:
            return lab
    return None


def select_record_item_from_slots(slots: dict[str, list[str]] | None) -> str | None:
    if not slots:
        return None
    cat = select_category_entity(slots.get("CATEGORY", []))
    if cat:
        return cat
    prods = [p.strip() for p in slots.get("PRODUCT", []) if p and str(p).strip()]
    if not prods:
        return None
    prods.sort(key=len)
    return prods[-1]


def load_ner_model(model_dir) -> Any:
    if spacy is None:
        return None
    if model_dir.exists():
        return spacy.load(str(model_dir))
    return None


def extract_ner_slots(text: str, nlp) -> dict:
    doc = nlp(text)
    slots: dict[str, list[str]] = {}
    for ent in doc.ents:
        slots.setdefault(ent.label_, []).append(ent.text)
    return slots


def select_category_entity(categories: list[str]) -> str | None:
    if not categories:
        return None
    cleaned = [c.strip() for c in categories if c and c.strip()]
    if not cleaned:
        return None
    filtered = [c for c in cleaned if normalize_text(c) not in CATEGORY_STOPWORDS]
    pool = filtered or cleaned
    pool.sort(key=lambda item: (len(item), cleaned.index(item)))
    return pool[-1]


def map_category_to_label(category: str | None) -> str | None:
    if not category:
        return None
    key = normalize_text(category)
    return CATEGORY_MAP.get(key)
