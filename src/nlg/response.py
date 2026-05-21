import json
import unicodedata

STATUS_ALLOWED = {"vui", "buon", "canh_bao", "trung_lap"}


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
