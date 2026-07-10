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
    try:
        from pathlib import Path
        import json
        import os
        reg_paths = [
            Path("/storage/nlu_models/nlu_model_registry.json"),
            Path(__file__).resolve().parents[2] / "text_nlu" / "models" / "nlu_model_registry.json"
        ]
        backend = "llm"
        for p in reg_paths:
            if p.exists() and p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    backend = str(data.get("inference_backend", "llm")).strip().lower()
                    break
                except Exception:
                    pass
        if backend == "llm":
            return None
    except Exception:
        pass
    if spacy is None:
        return None
    from pathlib import Path
    storage_path = Path("/storage/nlu_models/ner_model/model-best")
    if storage_path.is_dir():
        model_dir = storage_path
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



