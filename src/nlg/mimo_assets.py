"""Tên file PNG mascot — một nguồn chuẩn cho NLG, AI service, mobile."""

from __future__ import annotations

MIMO_ASSET_NAMES = frozenset({
    "Alert", "Angry", "Approved", "Celebrate", "Chill", "Cooking", "Cool",
    "Determined", "Error", "Excited", "Giggle", "Happy", "Hello", "Loading",
    "Love", "Proud", "Relax", "Sad", "Sleepy", "Sassy", "Shopping", "Travel",
    "Sorry", "Success", "Taunting", "Thankful", "Thinking", "Working", "Worried",
})

# Persona NLG (prompts.json) — không phải tên file PNG
NLG_PERSONA_KEYS = frozenset({
    "vui", "dan_doi"
})


def coerce_mimo_asset(value: str | None) -> str | None:
    """Khớp tên LLM với MIMO_ASSET_NAMES (không phân biệt hoa thường)."""
    if not value or not str(value).strip():
        return None
    v = str(value).strip()
    v_lower = v.lower()
    
    # Map persona keys directly to visual mascot emotions
    PERSONA_TO_EMOTION = {
        "vui":     "Happy",   # Năng lượng cao
        "dan_doi": "Angry", # Lo lắng nhẹ
    }
    if v_lower in PERSONA_TO_EMOTION:
        return PERSONA_TO_EMOTION[v_lower]

    if v in NLG_PERSONA_KEYS:
        return None
    if v in MIMO_ASSET_NAMES:
        return v
    lower = v.lower()
    for name in MIMO_ASSET_NAMES:
        if name.lower() == lower:
            return name
    return None
