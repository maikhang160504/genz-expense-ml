from __future__ import annotations

from typing import Any

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

from .text import normalize_text

VALID_LABELS = {
    "Food", "Essentials", "Social", "Transport", "Shopping", "Housing", 
    "Health", "Beauty", "Education", "Entertainment", "Investment", "Others"
}




def map_product_brand_hint(name: str | None) -> str | None:
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
    # Use sorted() to avoid mutating `cleaned` during comparison
    pool = sorted(cleaned, key=lambda item: (len(item), cleaned.index(item)))
    return pool[-1]



