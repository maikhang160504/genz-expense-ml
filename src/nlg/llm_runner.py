"""Gọi Gemini/Groq sau NLU — dùng chung CLI và API."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.llm.client import call_groq, ensure_gemini_system_instruction
from src.llm.gemini_keys import call_gemini_with_key_fallback
from src.nlg.context_meta import filter_context_metadata_for_prompt
from src.nlg.prompt import build_nlg_prompt
from src.nlg.response import (
    extract_mimo_emotion_from_llm_block,
    intent_mimo_fallback,
    parse_llm_response,
)


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
    """Bổ sung ngữ cảnh NLU — chỉ các trường đã lọc, tránh dump metadata thô."""
    intent = nlu_result.get("intent")
    ctx_json = json.dumps(context_metadata, ensure_ascii=False) if context_metadata else None
    if intent == "Chitchat":
        if ctx_json:
            return f'Câu người dùng: "{user_text}". Intent: Chitchat. Ngữ cảnh (đã lọc): {ctx_json}.'
        return f'Câu người dùng: "{user_text}". Intent: Chitchat.'
    slim_nlu = {
        k: nlu_result[k]
        for k in (
            "intent",
            "text",
            "item",
            "category",
            "amount",
            "record_type",
            "is_expense",
            "income_type",
            "action_type",
        )
        if nlu_result.get(k) is not None
    }
    nlu_json = json.dumps(slim_nlu, ensure_ascii=False)
    if ctx_json:
        return (
            f'Câu người dùng: "{user_text}". '
            f"Dữ liệu NLU: {nlu_json}. "
            f"Ngữ cảnh (đã lọc): {ctx_json}."
        )
    return (
        f'Câu người dùng: "{user_text}". '
        f"Dữ liệu NLU: {nlu_json}."
    )


def _apply_gemini_nlg_schema(payload: dict) -> None:
    """Ép Gemini trả đúng response + mimo_emotion (không dùng schema cũ story/status)."""
    cfg = payload.setdefault("generationConfig", {})
    cfg["responseMimeType"] = "application/json"
    cfg["responseSchema"] = {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
            "mimo_emotion": {"type": "string"},
        },
        "required": ["response", "mimo_emotion"],
    }


def should_run_llm(intent: str) -> bool:
    if os.environ.get("RUN_LLM", "0") == "1":
        return True
    if intent == "Chitchat" and os.environ.get("RUN_LLM_CHITCHAT", "1") == "1":
        return True
    return False


def _finalize_llm_block(
    llm_json: dict,
    *,
    intent: str,
    _is_triggered: bool,
    record_type: str | None = None,
) -> None:
    if "story" in llm_json and "response" not in llm_json:
        llm_json["response"] = llm_json["story"]
    if "response" in llm_json and "story" not in llm_json:
        llm_json["story"] = llm_json["response"]
    llm_json.pop("status", None)
    from_llm = extract_mimo_emotion_from_llm_block(llm_json)
    asset = from_llm or intent_mimo_fallback(intent, record_type)
    llm_json["mimo_emotion"] = asset
    llm_json["emotion"] = asset
    if os.environ.get("LOG_MIMO_EMOTION", "0") == "1":
        print(
            f"[mimo-emotion] intent={intent} from_llm={from_llm} resolved={asset} "
            f"keys={list(llm_json.keys())}",
            flush=True,
        )


def attach_nlg_and_llm(
    result: dict[str, Any],
    *,
    user_text: str,
    nlu_result: dict[str, Any],
    context_metadata: dict[str, Any],
    prompts_config: dict[str, Any],
    request_template: dict | None,
    nlg_persona: str = "hai_huoc",
    emotion: str | None = None,  # deprecated alias
    chat_history: list | None = None,
    chat_summary: str | None = None,
) -> dict[str, Any]:
    """Bổ sung nlg_prompt, gemini_json, llama_json vào result (in-place + return)."""
    persona = nlg_persona or emotion or "hai_huoc"
    intent = str(result.get("intent", ""))
    prompt_persona = "nghiem_tuc" if intent == "Action" else persona
    result["nlg_persona"] = prompt_persona
    result.pop("emotion", None)  # tránh nhầm với mimo_emotion asset
    prompt_meta = filter_context_metadata_for_prompt(
        context_metadata, nlu_result, user_text
    )
    result["nlg_prompt"] = build_nlg_prompt(
        prompts_config,
        prompt_persona,
        nlu_result,
        prompt_meta,
        chat_history=chat_history,
        chat_summary=chat_summary,
    )

    result["gemini_json"] = None
    result["llama_json"] = None
    result.pop("gemini_error", None)
    result.pop("llama_error", None)

    if not should_run_llm(str(result.get("intent", ""))):
        return result
    if not request_template:
        result["llm_skipped"] = "no_request_template"
        return result

    fusion_text = build_fusion_text(nlu_result, prompt_meta, user_text)
    nlg_user = result["nlg_prompt"]["user"]
    llm_user_text = f"{nlg_user}\n\n{fusion_text}"

    llama_api = os.environ.get("Llama_API")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    llama_model = os.environ.get("LLAMA_MODEL", "llama-3.1-8b-instant")
    llm_mode = os.environ.get("LLM_MODE", "both")
    is_triggered = bool(context_metadata.get("is_triggered", False))

    if llm_mode in {"gemini", "both"}:
        payload = json.loads(json.dumps(request_template))
        payload["contents"][0]["parts"][0]["text"] = llm_user_text
        _apply_gemini_nlg_schema(payload)
        ensure_gemini_system_instruction(payload, result["nlg_prompt"]["system"])
        try:
            raw = call_gemini_with_key_fallback(gemini_model, payload)
            result["gemini_response"] = raw
            gemini_json = parse_llm_response(raw, "gemini")
            if gemini_json:
                _finalize_llm_block(
                    gemini_json,
                    intent=intent,
                    _is_triggered=is_triggered,
                    record_type=nlu_result.get("record_type"),
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
                llm_user_text,
                temperature=0.6,
                max_tokens=160,
            )
            result["llama_response"] = raw
            llama_json = parse_llm_response(raw, "llama")
            if llama_json:
                _finalize_llm_block(
                    llama_json,
                    intent=intent,
                    _is_triggered=is_triggered,
                    record_type=nlu_result.get("record_type"),
                )
            result["llama_json"] = llama_json
        except Exception as exc:
            result["llama_error"] = str(exc)

    gemini = result.get("gemini_json")
    if isinstance(gemini, dict):
        top = extract_mimo_emotion_from_llm_block(gemini) or intent_mimo_fallback(
            intent, nlu_result.get("record_type")
        )
        result["mimo_emotion"] = top
        result["llm_emotion"] = top
        result["mascot_mood"] = top

    return result
