"""Round-robin Gemini and ChatGPT API keys: v1 → v2 → v3 on 429/quota (không retry cùng key)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from src.llm.client import call_gemini


def get_all_llm_keys() -> list[dict[str, str]]:
    """Thứ tự ưu tiên: 2 Gemini keys trước, rồi đến ChatGPT keys."""
    seen: set[str] = set()
    keys: list[dict[str, str]] = []
    
    # 1. Gemini Keys
    gemini_names = ["gemini_API_v1", "gemini_API_v2", "gemini_API_v3"]
    has_v1_v2 = any((os.environ.get(name) or "").strip() for name in gemini_names)
    if not has_v1_v2:
        gemini_names = ["gemini_API", "GEMINI_API"]

    for name in gemini_names:
        val = (os.environ.get(name) or "").strip()
        if val and val not in seen:
            seen.add(val)
            keys.append({
                "provider": "gemini",
                "key": val,
                "label": name
            })
            
    # 2. ChatGPT Keys
    for name in (
        "chatgpt_API_v1",
        "chatgpt_API_v2",
        "chatgpt_API_v3",
        "chatgpt_API",
        "CHATGPT_API",
        "OPENAI_API_KEY",
    ):
        val = (os.environ.get(name) or "").strip()
        if val and val not in seen:
            seen.add(val)
            keys.append({
                "provider": "chatgpt",
                "key": val,
                "label": name
            })
            
    return keys


def list_gemini_api_keys() -> list[str]:
    """Legacy compatibility: returns list of Gemini API keys only."""
    seen: set[str] = set()
    keys: list[str] = []
    gemini_names = ["gemini_API_v1", "gemini_API_v2", "gemini_API_v3"]
    has_v1_v2 = any((os.environ.get(name) or "").strip() for name in gemini_names)
    if not has_v1_v2:
        gemini_names = ["gemini_API", "GEMINI_API"]

    for name in gemini_names:
        val = (os.environ.get(name) or "").strip()
        if val and val not in seen:
            seen.add(val)
            keys.append(val)
    return keys


def _is_rotate_key_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        x in msg
        for x in ("429", "resource exhausted", "resource_exhausted", "quota", "rate limit", "insufficient_quota")
    )


def _convert_gemini_payload_to_openai(model: str, payload: dict) -> dict:
    # 1. System Prompt
    system_prompt = ""
    sys_instruction = payload.get("systemInstruction")
    if sys_instruction:
        if isinstance(sys_instruction, str):
            system_prompt = sys_instruction
        elif isinstance(sys_instruction, dict):
            parts = sys_instruction.get("parts") or []
            system_prompt = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

    # 2. User Prompt
    user_prompt = ""
    contents = payload.get("contents") or []
    if isinstance(contents, list) and contents:
        parts_list = []
        for c in contents:
            if isinstance(c, dict):
                parts = c.get("parts") or []
                for p in parts:
                    if isinstance(p, dict) and p.get("text"):
                        parts_list.append(p.get("text"))
                    elif isinstance(p, str):
                        parts_list.append(p)
        user_prompt = "\n".join(parts_list).strip()

    # 3. Build message list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    # 4. generationConfig conversion
    gen_config = payload.get("generationConfig") or {}
    temperature = gen_config.get("temperature", 0.7)
    
    # 5. response format (JSON mode)
    response_format = None
    if gen_config.get("responseMimeType") == "application/json":
        response_format = {"type": "json_object"}
        schema = gen_config.get("responseSchema")
        if schema:
            schema_str = json.dumps(schema, ensure_ascii=False)
            if messages:
                messages[-1]["content"] += f"\n\nCRITICAL: You MUST respond in JSON conforming to this schema:\n{schema_str}"

    openai_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        openai_payload["response_format"] = response_format

    return openai_payload


def _call_openai_api(api_key: str, model: str, payload: dict) -> dict:
    openai_payload = _convert_gemini_payload_to_openai(model, payload)
    data = json.dumps(openai_payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_json = json.loads(resp.read().decode("utf-8"))
            content = ""
            choices = response_json.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": content
                                }
                            ]
                        }
                    }
                ]
            }
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {err_body}")
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")


def call_gemini_with_key_fallback(model: str, payload: dict[str, Any]) -> dict:
    keys = get_all_llm_keys()
    if not keys:
        raise RuntimeError(
            "No LLM API keys configured (set gemini_API_v1/v2 or chatgpt_API_v1/v2/v3 in .env)"
        )

    # Phân nhóm keys theo provider
    gemini_keys = [k for k in keys if k["provider"] == "gemini"]
    chatgpt_keys = [k for k in keys if k["provider"] == "chatgpt"]

    # Xây dựng danh sách các mô hình Gemini xoay tua (rotate)
    gemini_models = []
    seen_models = set()
    
    # Ưu tiên mô hình đầu vào được truyền trực tiếp
    if model:
        m_clean = model.strip()
        if m_clean and m_clean not in seen_models:
            seen_models.add(m_clean)
            gemini_models.append(m_clean)
            
    # Đọc các model xoay vòng từ .env (GEMINI_MODEL_v1, GEMINI_MODEL_v2, v3, v4, ...)
    for i in range(1, 11):
        val = (os.environ.get(f"GEMINI_MODEL_v{i}") or "").strip()
        if val and val not in seen_models:
            seen_models.add(val)
            gemini_models.append(val)
            
    # Thêm GEMINI_MODEL mặc định từ env nếu chưa có
    for name in ("GEMINI_MODEL", "gemini_MODEL"):
        val = (os.environ.get(name) or "").strip()
        if val and val not in seen_models:
            seen_models.add(val)
            gemini_models.append(val)
            
    # Hàng đợi mô hình dự phòng mặc định (fallback defaults)
    defaults = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-3-flash"]
    for m in defaults:
        if m not in seen_models:
            seen_models.add(m)
            gemini_models.append(m)

    last_exc: BaseException | None = None

    # 1. Thử gọi Gemini với cơ chế xoay tua mô hình và keys
    if gemini_keys:
        for model_name in gemini_models:
            for key_info in gemini_keys:
                api_key = key_info["key"]
                label = key_info["label"]
                try:
                    print(f"[gemini] Trying model {model_name} with key {label}...", flush=True)
                    return call_gemini(api_key, model_name, payload)
                except Exception as exc:
                    last_exc = exc
                    if _is_rotate_key_error(exc):
                        print(
                            f"[gemini] {label} hit quota/rate limit on model {model_name}. Switching to next key/model...",
                            flush=True,
                        )
                        continue
                    print(f"[gemini] {label} failed on model {model_name}: {exc!s:.160}", flush=True)
                    continue

    # 2. Thử gọi ChatGPT nếu tất cả các cấu hình Gemini thất bại
    if chatgpt_keys:
        chatgpt_model = os.environ.get("CHATGPT_MODEL", "gpt-4o-mini")
        for key_info in chatgpt_keys:
            api_key = key_info["key"]
            label = key_info["label"]
            try:
                print(f"[chatgpt] Fallback: Trying model {chatgpt_model} with key {label}...", flush=True)
                return _call_openai_api(api_key, chatgpt_model, payload)
            except Exception as exc:
                last_exc = exc
                print(f"[chatgpt] Key {label} failed on model {chatgpt_model}: {exc!s:.160}", flush=True)
                continue

    if last_exc:
        raise last_exc
    raise RuntimeError("All LLM models and keys failed to execute")
