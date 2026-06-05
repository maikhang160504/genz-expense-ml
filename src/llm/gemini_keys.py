"""Round-robin Gemini API keys: v1 → v2 → v3 on 429/quota (không retry cùng key)."""
from __future__ import annotations

import os
from typing import Any

from src.llm.client import call_gemini


def list_gemini_api_keys() -> list[str]:
    """Thứ tự ưu tiên: v1, v2, v3 rồi alias legacy."""
    seen: set[str] = set()
    keys: list[str] = []
    for name in (
        "gemini_API_v1",
        "gemini_API_v2",
        "gemini_API_v3",
        "gemini_API",
        "GEMINI_API",
    ):
        val = (os.environ.get(name) or "").strip()
        if val and val not in seen:
            seen.add(val)
            keys.append(val)
    return keys


def _is_rotate_key_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        x in msg
        for x in ("429", "resource exhausted", "resource_exhausted", "quota", "rate limit")
    )


def call_gemini_with_key_fallback(model: str, payload: dict[str, Any]) -> dict:
    keys = list_gemini_api_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini API key configured (set gemini_API_v1 / v2 / v3 in .env)"
        )
    last_exc: BaseException | None = None
    for idx, api_key in enumerate(keys):
        label = f"gemini_API_v{idx + 1}" if idx < 3 else f"gemini_API_{idx + 1}"
        try:
            return call_gemini(api_key, model, payload)
        except Exception as exc:
            last_exc = exc
            if _is_rotate_key_error(exc) and idx < len(keys) - 1:
                print(
                    f"[gemini] {label} quota/rate limit — switch to next key in .env",
                    flush=True,
                )
                continue
            print(f"[gemini] {label} failed: {exc!s:.160}", flush=True)
            if idx < len(keys) - 1:
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Gemini call failed")
