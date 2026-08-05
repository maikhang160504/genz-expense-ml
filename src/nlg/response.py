import json

from src.nlg.mimo_assets import MIMO_ASSET_NAMES, coerce_mimo_asset


def intent_mimo_fallback(intent: str | None, record_type: str | None = None) -> str:
    """Chỉ khi LLM không trả mimo_emotion hợp lệ."""
    if intent == "Record":
        if record_type == "Income":
            return "Celebrate"
        return "Success"
    if intent == "Action":
        return "Approved"
    return "Hello"


def extract_mimo_emotion_from_llm_block(block: dict | None) -> str | None:
    """Chỉ đọc mimo_emotion / emotion (PascalCase) — không dùng status tiếng Việt cũ."""
    if not block:
        return None
    for key in ("mimo_emotion", "emotion"):
        coerced = coerce_mimo_asset(block.get(key))
        if coerced:
            return coerced
    return None


def normalize_mimo_emotion(value: str | None, fallback: str = "Hello") -> str:
    return coerce_mimo_asset(value) or fallback


def normalize_emotion(value: str | None, fallback: str = "Hello") -> str:
    return normalize_mimo_emotion(value, fallback)


def _parse_json_text(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_llm_response(response: dict | str, source: str = "qwen") -> dict | None:
    if not response:
        return None
    if isinstance(response, str):
        return _parse_json_text(response)
    if source == "llama":
        choices = response.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
            return _parse_json_text(text)
    return None


# Backward-compat exports
EMOTION_ALLOWED = set(MIMO_ASSET_NAMES)
