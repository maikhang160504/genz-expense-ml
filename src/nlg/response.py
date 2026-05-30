import json
import unicodedata

STATUS_ALLOWED = {"vui", "buon", "canh_bao", "trung_lap"}

EMOTION_ALLOWED = {
    "Alert", "Angry", "Approved", "Celebrate", "Chill", "Cooking", "Cool",
    "Determined", "Error", "Excited", "Giggle", "Happy", "Hello", "Loading",
    "Love", "Proud", "Relax", "Sad", "Sleepy", "Sassy", "Shopping", "Travel",
    "Sorry", "Success", "Taunting", "Thankful", "Thinking", "Working", "Worried",
}

_EMOTION_LOWER_MAP: dict[str, str] = {e.lower(): e for e in EMOTION_ALLOWED}

_EMOTION_ALIASES: dict[str, str] = {
    "vui": "Happy",
    "buon": "Sad",
    "canh_bao": "Thinking",
    "trung_lap": "Chill",
    "dan_doi": "Sad",
    "cham_choc": "Taunting",
    "hai_huoc": "Sassy",
    "dong_cam": "Approved",
    "nghiem_tuc": "Thinking",
    "can_than": "Alert",
    "canh_bao_cam": "Alert",
    "khong_ro": "Chill",
}


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def normalize_status(value: str | None, is_triggered: bool) -> str:
    if not value:
        return "canh_bao" if is_triggered else "trung_lap"
    lowered = value.strip().lower()
    lowered = _strip_accents(lowered)
    lowered = lowered.replace(" ", "_")
    if lowered in STATUS_ALLOWED:
        return lowered
    if lowered in {"canhbao", "canh-bao"}:
        return "canh_bao"
    if lowered in {"trunglap", "trung-lap"}:
        return "trung_lap"
    return "canh_bao" if is_triggered else "trung_lap"


def normalize_emotion(value: str | None, fallback: str = "Chill") -> str:
    """Map LLM-returned emotion string → PascalCase Flutter asset name (one of EMOTION_ALLOWED)."""
    if not value:
        return fallback
    v = value.strip()
    if v in EMOTION_ALLOWED:
        return v
    lowered = v.lower().replace(" ", "_").replace("-", "_")
    if lowered in _EMOTION_LOWER_MAP:
        return _EMOTION_LOWER_MAP[lowered]
    if lowered in _EMOTION_ALIASES:
        return _EMOTION_ALIASES[lowered]
    no_accent = _strip_accents(lowered)
    if no_accent in _EMOTION_LOWER_MAP:
        return _EMOTION_LOWER_MAP[no_accent]
    if no_accent in _EMOTION_ALIASES:
        return _EMOTION_ALIASES[no_accent]
    return fallback


def _parse_json_text(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_llm_response(response: dict, source: str) -> dict | None:
    if not response:
        return None
    if source == "gemini":
        parsed = response.get("parsed")
        if isinstance(parsed, dict):
            return parsed
        candidates = response.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return _parse_json_text(parts[0].get("text", ""))
        return None

    if source == "llama":
        choices = response.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
            return _parse_json_text(text)
    return None
