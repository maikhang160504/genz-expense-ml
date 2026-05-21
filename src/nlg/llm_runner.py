"""Gọi Gemini/Groq sau NLU — dùng chung CLI và API."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.llm.client import call_gemini, call_groq, ensure_gemini_system_instruction
from src.nlg.prompt import build_nlg_prompt
from src.nlg.response import normalize_status, parse_llm_response


def load_request_template(path: Path) -> dict | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8-sig").strip()
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    if not blocks:
        return None
    try:
        return json.loads(blocks[0])
    except json.JSONDecodeError:
        return None


def load_prompts(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_fusion_text(nlu_result: dict, context_metadata: dict, user_text: str = "") -> str:
    intent = nlu_result.get("intent")
    if intent == "Chitchat":
        return (
            f"Câu người dùng: \"{user_text}\". Intent: Chitchat. "
            f"Ngữ cảnh app: {json.dumps(context_metadata, ensure_ascii=False)}. "
            "Trả lời đúng chủ đề, chọn status từ câu user, gợi ý nhẹ 1 CTA (ghi chi / xem báo cáo)."
        )
    return (
        f"Dữ liệu fusion: {json.dumps(nlu_result, ensure_ascii=False)}. "
        f"Ngữ cảnh: {json.dumps(context_metadata, ensure_ascii=False)}. "
        "Hãy tạo một story ngắn, bắt trend, đúng vai trò theo system prompt."
    )


def should_run_llm(intent: str) -> bool:
    if os.environ.get("RUN_LLM", "0") == "1":
        return True
    if intent == "Chitchat" and os.environ.get("RUN_LLM_CHITCHAT", "1") == "1":
        return True
    return False


def attach_nlg_and_llm(
    result: dict[str, Any],
    *,
    user_text: str,
    nlu_result: dict[str, Any],
    context_metadata: dict[str, Any],
    prompts_config: dict[str, Any],
    request_template: dict | None,
    emotion: str = "hai_huoc",
) -> dict[str, Any]:
    """Bổ sung nlg_prompt, gemini_json, llama_json vào result (in-place + return)."""
    result["emotion"] = emotion
    result["nlg_prompt"] = build_nlg_prompt(prompts_config, emotion, nlu_result, context_metadata)

    result["gemini_json"] = None
    result["llama_json"] = None
    result.pop("gemini_error", None)
    result.pop("llama_error", None)

    if not should_run_llm(str(result.get("intent", ""))):
        return result
    if not request_template:
        result["llm_skipped"] = "no_request_template"
        return result

    fusion_text = build_fusion_text(nlu_result, context_metadata, user_text)
    gemini_api = os.environ.get("gemini_API") or os.environ.get("GEMINI_API")
    llama_api = os.environ.get("Llama_API")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    llama_model = os.environ.get("LLAMA_MODEL", "llama-3.1-8b-instant")
    llm_mode = os.environ.get("LLM_MODE", "both")

    if gemini_api and llm_mode in {"gemini", "both"}:
        payload = json.loads(json.dumps(request_template))
        payload["contents"][0]["parts"][0]["text"] = fusion_text
        ensure_gemini_system_instruction(payload, result["nlg_prompt"]["system"])
        try:
            raw = call_gemini(gemini_api, gemini_model, payload)
            result["gemini_response"] = raw
            gemini_json = parse_llm_response(raw, "gemini")
            if gemini_json:
                gemini_json["status"] = normalize_status(
                    gemini_json.get("status"),
                    context_metadata.get("is_triggered", False),
                )
            result["gemini_json"] = gemini_json
        except Exception as exc:
            result["gemini_error"] = str(exc)

    if llama_api and llm_mode in {"llama", "both"}:
        try:
            raw = call_groq(
                llama_api,
                llama_model,
                result["nlg_prompt"]["system"],
                f"{result['nlg_prompt']['user']}\n\n{fusion_text}",
                temperature=0.6,
                max_tokens=160,
            )
            result["llama_response"] = raw
            llama_json = parse_llm_response(raw, "llama")
            if llama_json:
                llama_json["status"] = normalize_status(
                    llama_json.get("status"),
                    context_metadata.get("is_triggered", False),
                )
            result["llama_json"] = llama_json
        except Exception as exc:
            result["llama_error"] = str(exc)

    return result
