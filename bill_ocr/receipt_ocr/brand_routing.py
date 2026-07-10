"""Brand → category routing for single-category merchants."""
from __future__ import annotations

import re
import unicodedata

BRAND_CATEGORY_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"highlands|phuc\s*long|starbucks|the\s*coffee|ca\s*phe|coffee\s*house|gong\s*cha|mixue|katinat|phuc\s*loc\s*tho|kfc|mcdonald|jollibee|lotteria|pizza|quan\s*an|nha\s*hang|tra\s*sua", re.I), "Food"),
    (re.compile(r"pharmacity|long\s*chau|guardian|medicare|phano|an\s*khang|nha\s*thuoc", re.I), "Health"),
    (re.compile(r"zara|uniqlo|hm\b|mango\b|canifa|thoi\s*trang|quan\s*ao", re.I), "Shopping"),
    (re.compile(r"circle\s*k|gs25|family\s*mart|ministop|7\s*-\s*eleven|7\s*eleven", re.I), "Food"),
    (re.compile(r"winmart|vinmart|co\.?\s*op|coopmart|lotte|bach\s*hoa|aeon|big\s*c|sieu\s*thi", re.I), "Essentials"),
    (re.compile(r"grab|gojek|xanh\s*sm|taxi|mai\s*linh|vinasun|petrolimex|xang\s*dau", re.I), "Transport"),
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def route_brand_category(seller_text: str | None) -> str | None:
    """Return fixed category for known single-category brands, else None."""
    if not seller_text or not str(seller_text).strip():
        return None
    blob = _normalize(str(seller_text))
    for pattern, category in BRAND_CATEGORY_RULES:
        if pattern.search(blob):
            return category
    return None
