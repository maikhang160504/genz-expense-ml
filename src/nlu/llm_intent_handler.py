"""
LLM-based intent handler: fallback khi encoder confidence thấp.

Sử dụng LLM local (qua API endpoint) hoặc Gemini để:
1. Classify intent (Record / Action / Chitchat)
2. Extract slots tương ứng

Logic phải khớp 100% với action.md.
"""
from __future__ import annotations

import json
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

from src.prompts.llm_prompts import (
    UNIFIED_NLU_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    ACTION_SLOT_EXTRACTION_PROMPT,
    RECORD_SLOT_EXTRACTION_PROMPT
)

_PROMPTS_CACHE = None

def _load_prompts_json() -> dict:
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    try:
        from pathlib import Path
        prompts_path = Path(__file__).resolve().parent.parent / "prompts" / "prompts.json"
        with open(prompts_path, "r", encoding="utf-8") as f:
            _PROMPTS_CACHE = json.load(f)
    except Exception as e:
        logger.error(f"Error loading prompts.json: {e}")
        _PROMPTS_CACHE = {}
    return _PROMPTS_CACHE



# ── Default confidence threshold ──
DEFAULT_CONFIDENCE_THRESHOLD = 0.65


def _repair_truncated_json(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # If starts with { but doesn't end with }, try to fix
    if s.startswith("{") and not s.endswith("}"):
        # If inside a string value (odd number of quotes), close the string value first
        if s.count('"') % 2 == 1:
            s += '"'
        # Count open vs close braces and balance them
        open_braces = s.count("{")
        close_braces = s.count("}")
        if open_braces > close_braces:
            s += "}" * (open_braces - close_braces)
    return s


def _extract_first_json(text: str) -> str | None:
    """Find the first matching JSON block {...} in text."""
    start_idx = text.find("{")
    if start_idx == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
        escape = False
    return None


def _parse_llm_json(text: str) -> dict | None:
    """Parse JSON từ LLM response, handle markdown code blocks and auto-repair truncation."""
    text = text.strip()
    # Remove markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    
    # Try normal parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first complete {...} block
    extracted = _extract_first_json(text)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            # Try to repair truncated JSON inside extracted block
            repaired = _repair_truncated_json(extracted)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Try to repair the whole text
    repaired = _repair_truncated_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    logger.warning("Failed to parse LLM JSON response (even after repair/regex): %s", text[:200])
    return None


def _sanitize_nlg_response(text: str) -> str:
    """Strip CJK characters, code snippets and repeated words from LLM NLG response."""
    import re
    if not text:
        return text
    # 1. Remove CJK / Chinese / Japanese / Korean characters (U+4E00–U+9FFF, U+3000–U+303F, etc.)
    text = re.sub(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+", "", text)
    # 2. Remove code snippets patterns like .addAction(...), {key: val}, (), function calls
    text = re.sub(r"\.\w+\([^)]*\)", "", text)           # .addAction("X", {...})
    text = re.sub(r"\{[^}]{0,200}\}", "", text)           # {...} blocks
    text = re.sub(r"//.*", "", text)                      # // comments
    # 3. Remove excessive repeated words (e.g. "mascot mascot" → "mascot")
    text = re.sub(r"\b(\w{3,})\s+\1\b", r"\1", text)
    # 4. Strip trailing/leading whitespace and punctuation artifacts
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.rstrip("，。！？,.")
    return text if text else "Mimo đã ghi nhận rồi nha!"


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for p in (root, root / "text_nlu"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    use_local = os.environ.get("USE_LOCAL_PHOGPT") == "1" and os.environ.get("IS_MODAL") != "true"
    use_modal = os.environ.get("USE_MODAL_PHOGPT") == "1" or os.environ.get("IS_MODAL") == "true"

    if use_local or use_modal:
        if use_modal:
            try:
                logger.info("Attempting to call remote QwenModel on Modal...")
                try:
                    # Direct import works inside 'modal run' (ephemeral app)
                    from modal_app import QwenModel
                    model_client = QwenModel()
                    text = model_client.generate.remote(system_prompt, user_prompt)
                except Exception as e_inner:
                    logger.warning(f"Direct import QwenModel failed ({e_inner}), trying from_name...")
                    # Fallback for deployed external calls
                    import modal
                    QwenModel = modal.Cls.from_name("expense-ocr-nlu", "QwenModel")
                    model_client = QwenModel()
                    text = model_client.generate.remote(system_prompt, user_prompt)
                
                if text:
                    logger.info("Successfully received response from remote QwenModel.")
                    return text
            except Exception as e:
                logger.warning("Remote QwenModel call failed: %s. Falling back...", e)

        # Fallback to local HuggingFace loading if requested
        if use_local:
            try:
                logger.info("Calling local PhoGPT-7B5 model...")
                from src.nlu.local_llm import run_local_phogpt_inference
                text = run_local_phogpt_inference(system_prompt, user_prompt)
                if text:
                    return text
            except Exception as ex:
                logger.warning("Local PhoGPT-7B5 inference failed: %s", ex)

    local_url = os.environ.get("LOCAL_LLM_URL")
    local_model = os.environ.get("LOCAL_LLM_MODEL") or "qwen2.5-3b-instruct"
    if local_url:
        try:
            from pipeline.llm_module import call_lmstudio, extract_chat_text
            logger.info("Calling local LLM at %s with model %s", local_url, local_model)
            resp = call_lmstudio(local_url, local_model, system_prompt, user_prompt, temperature=0.15)
            text = extract_chat_text(resp)
            if text:
                return text
        except Exception as e:
            logger.warning("Local LLM call failed: %s. Falling back to API...", e)

    try:
        from src.llm.gemini_keys import call_gemini_with_key_fallback
        from pipeline.llm_module import extract_chat_text
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        payload = {
            "systemInstruction": system_prompt,
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
            }
        }
        logger.info("Calling Gemini key rotation fallback for NLU...")
        resp = call_gemini_with_key_fallback(gemini_model, payload)
        return extract_chat_text(resp)
    except Exception as e:
        logger.error("All LLM NLU calls failed: %s", e)
        return ""


def classify_intent_llm(
    text: str,
    *,
    llm_call_fn=None,
    api_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, float]:
    """
    Classify intent using LLM.
    
    Returns: (intent, confidence)
    """
    if llm_call_fn is None:
        llm_call_fn = _call_llm

    try:
        response = llm_call_fn(
            system_prompt=INTENT_CLASSIFICATION_PROMPT,
            user_prompt=f"Câu nói: \"{text}\"",
        )
        
        parsed = _parse_llm_json(str(response))
        if parsed:
            intent = parsed.get("intent", "Chitchat")
            confidence = float(parsed.get("confidence", 0.5))
            if intent not in ("Record", "Action", "Chitchat"):
                intent = "Chitchat"
            return intent, confidence
    except Exception as e:
        logger.warning("LLM intent classification failed: %s", e)
    
    return "Chitchat", 0.0


def extract_action_slots_llm(
    text: str,
    *,
    llm_call_fn=None,
) -> dict[str, Any]:
    """
    Extract action slots using LLM.
    
    Returns: dict with action_type and slot values
    """
    if llm_call_fn is None:
        llm_call_fn = _call_llm

    try:
        response = llm_call_fn(
            system_prompt=ACTION_SLOT_EXTRACTION_PROMPT,
            user_prompt=f"Câu nói: \"{text}\"",
        )
        
        parsed = _parse_llm_json(str(response))
        if parsed:
            # Clean None values
            return {k: v for k, v in parsed.items() if v is not None}
    except Exception as e:
        logger.warning("LLM action slot extraction failed: %s", e)
    
    return {}


def extract_record_slots_llm(
    text: str,
    *,
    llm_call_fn=None,
) -> dict[str, Any]:
    """
    Extract record slots using LLM.
    
    Returns: dict with type, label, amount, item
    """
    if llm_call_fn is None:
        llm_call_fn = _call_llm

    try:
        response = llm_call_fn(
            system_prompt=RECORD_SLOT_EXTRACTION_PROMPT,
            user_prompt=f"Câu nói: \"{text}\"",
        )
        
        parsed = _parse_llm_json(str(response))
        if parsed:
            return {k: v for k, v in parsed.items() if v is not None}
    except Exception as e:
        logger.warning("LLM record slot extraction failed: %s", e)
    
    return {}


def run_llm_fallback(
    text: str,
    encoder_intent: str | None = None,
    encoder_confidence: float | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    *,
    llm_call_fn=None,
) -> dict[str, Any] | None:
    """
    Run LLM fallback when encoder confidence is below threshold.
    
    Returns:
        dict with LLM-enhanced NLU results if fallback was triggered,
        None if encoder confidence was sufficient.
    """
    # If encoder is confident enough, don't fallback
    if encoder_confidence is not None and encoder_confidence >= confidence_threshold:
        return None

    logger.info(
        "Encoder confidence %.2f < threshold %.2f, running LLM fallback for: %s",
        encoder_confidence or 0.0,
        confidence_threshold,
        text[:60],
    )

    # Step 1: Classify intent via LLM
    llm_intent, llm_conf = classify_intent_llm(text, llm_call_fn=llm_call_fn)
    
    result: dict[str, Any] = {
        "intent": llm_intent,
        "intent_confidence": llm_conf,
        "intent_backend": "llm_fallback",
    }

    # Step 2: Extract slots based on intent
    if llm_intent == "Action":
        slots = extract_action_slots_llm(text, llm_call_fn=llm_call_fn)
        result["action_type"] = slots.get("action_type")
        result["action_details"] = {
            k: v for k, v in slots.items()
            if k != "action_type"
        }
    elif llm_intent == "Record":
        slots = extract_record_slots_llm(text, llm_call_fn=llm_call_fn)
        result["record_type"] = "Income" if slots.get("type") == "income" else "Expense"
        result["category"] = slots.get("label")
        result["amount_spent"] = slots.get("amount")
        result["item"] = slots.get("item")

    return result


def run_llm_nlu(
    text: str,
    context_metadata: dict[str, Any] | None = None,
    run_llm: bool = False,
    override_prompt: str | None = None,
    nlg_persona: str | None = None,
) -> dict[str, Any]:
    try:
        system_tone_addition = ""
        user_tone_addition = ""
        prompts_config = _load_prompts_json()
        
        if nlg_persona:
            emotions = prompts_config.get("emotions", {})
            persona_key = nlg_persona.lower()
            if persona_key in emotions:
                persona_config = emotions[persona_key]
                sys_msg = persona_config.get("system", "")
                user_msg = persona_config.get("user", "")
                slangs = ", ".join(persona_config.get("slang_pool", []))
                
                system_tone_addition = f"\n\n[QUY TẮC PHONG CÁCH (Persona): {nlg_persona}]\n{sys_msg}\nTừ lóng được phép dùng: {slangs}"
                user_tone_addition = f"\n\n[YÊU CẦU PHẢN HỒI]: {user_msg}\n(TUYỆT ĐỐI KHÔNG DÙNG TIẾNG TRUNG/NGOẠI QUỐC)"
            else:
                system_tone_addition = f"\n\n[QUY TẮC PHONG CÁCH (Persona): {nlg_persona}]\nHãy đóng vai và trả lời theo phong cách này."
                user_tone_addition = "\n\n(TUYỆT ĐỐI KHÔNG DÙNG TIẾNG TRUNG/NGOẠI QUỐC)"

        # Check relationship override
        text_lower = text.lower()
        relationship_rule = ""
        rel_override = prompts_config.get("relationship_override", {})
        
        # Keywords for CHA_ME
        if any(kw in text_lower for kw in ["cha", "mẹ", "me", "ba", "má", "ma", "bố", "bo", "ông", "ong", "bà"]):
            cha_me = rel_override.get("CHA_ME", {})
            if "rule" in cha_me:
                relationship_rule = f"\n\n[ĐẶC BIỆT - QUAN HỆ NGƯỜI THÂN]: {cha_me['rule']}"
                
        # Keywords for NGUOI_YEU
        elif any(kw in text_lower for kw in ["người yêu", "nguoi yeu", "bồ", "bo", "vợ", "vo", "chồng", "chong", "gấu", "gau", "crush"]):
            nguoi_yeu = rel_override.get("NGUOI_YEU", {})
            is_expense = any(kw in text_lower for kw in ["mua", "tặng", "chuyển", "trả", "bao", "đãi", "đưa", "ăn"])
            is_income = any(kw in text_lower for kw in ["nhận", "được", "cho", "đòi"])
            if not is_income and not is_expense:
                is_expense = True # default
            
            if is_expense and "rule_happy" in nguoi_yeu:
                relationship_rule = f"\n\n[ĐẶC BIỆT - QUAN HỆ NGƯỜI YÊU]: {nguoi_yeu['rule_happy']}"
            elif is_income and "rule_sad" in nguoi_yeu:
                relationship_rule = f"\n\n[ĐẶC BIỆT - QUAN HỆ NGƯỜI YÊU]: {nguoi_yeu['rule_sad']}"

        user_tone_addition += relationship_rule

        if context_metadata:
            context_meta_str = json.dumps(context_metadata, ensure_ascii=False)
            user_prompt = f"Ngữ cảnh hệ thống (CONTEXT_META): {context_meta_str}\nCâu thoại của người dùng: {text}{user_tone_addition}"
        else:
            context_meta_str = "null"
            user_prompt = f"Ngữ cảnh hệ thống (CONTEXT_META): null\nCâu thoại của người dùng: {text}{user_tone_addition}"

        # 1. Check if we are in the second-pass for Action commentary
        if context_metadata and "action_facts" in context_metadata:
            system_prompt = (
                "Bạn là Mimo, trợ lý tài chính cá nhân của hệ thống spending-diary. "
                "Hãy phân tích dữ liệu thực tế được cung cấp trong trường 'action_facts' của 'Ngữ cảnh hệ thống (CONTEXT_META)' và câu nói của người dùng để trả về một cấu trúc JSON hợp lệ có dạng:\n"
                "{\n"
                '  "intent": "Action",\n'
                '  "emotion": "Alert" | "Angry" | "Approved" | "Celebrate" | "Chill" | "Cooking" | "Cool" | "Determined" | "Error" | "Excited" | "Giggle" | "Happy" | "Hello" | "Love" | "Proud" | "Relax" | "Sad" | "Sleepy" | "Sassy" | "Shopping" | "Travel" | "Sorry" | "Success" | "Taunting" | "Thankful" | "Thinking" | "Working" | "Worried",\n'
                '  "response": "<lời nhận xét, phân tích số liệu chi tiêu/thu nhập/kết quả tìm kiếm bằng tiếng Việt kiểu Gen Z, tối đa 2-3 câu ngắn>"\n'
                "}\n"
                "Quy tắc:\n"
                "1. Lời phản hồi 'response' PHẢI dựa trực tiếp trên số liệu thực tế được cung cấp trong 'action_facts' (ví dụ: tổng số tiền, số lượng giao dịch, danh mục chi nhiều nhất). Không được bịa đặt hoặc đoán mò số liệu nằm ngoài 'action_facts'.\n"
                "2. Nếu 'action_facts' trống hoặc chỉ ra không có dữ liệu chi tiêu (ví dụ: tổng chi tiêu = 0đ hoặc danh sách rỗng), hãy viết câu phản hồi hài hước nhẹ nhàng thông báo rằng người dùng chưa có giao dịch nào ghi nhận trong khoảng thời gian này và khuyên họ hãy bắt đầu ghi chép bằng cách nhắn 'ăn trưa 50k' hoặc tương tự nhé. TUYỆT ĐỐI KHÔNG được bịa đặt so sánh hay ảo giác nói rằng người dùng đã chi tiêu nhiều hơn hay bớt đi khi số liệu thực tế là 0đ.\n"
                "3. 'response' PHẢI là câu NHẬN XÉT CỤ THỂ, TỰ NHIÊN về dữ liệu. KHÔNG lặp lại các từ cửa miệng vô nghĩa. Phản hồi phải tuân thủ chặt chẽ QUY TẮC PHONG CÁCH bên dưới.\n"
                "4. 'response' phải viết bằng tiếng Việt 100% tự nhiên. TUYỆT ĐỐI không chứa bất kỳ chữ cái, từ ngữ nước ngoài nào khác (không tiếng Nga, không tiếng Anh, không tiếng Trung).\n"
                "5. Các con số tiền phải được viết rõ ràng định dạng phân cách hàng nghìn bằng dấu chấm (ví dụ: '1.200.000đ', '600.000đ', '400.000đ'). Không viết kiểu '1 triệu hai trăm nghìn'.\n"
                "Chỉ trả về JSON, không giải thích."
            )
            system_prompt += system_tone_addition
        else:
            system_prompt = override_prompt if override_prompt else UNIFIED_NLU_PROMPT
            system_prompt += system_tone_addition

        response = _call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        parsed = _parse_llm_json(str(response))
        if parsed:
            intent = parsed.get("intent", "Chitchat")
            if context_metadata and "action_facts" in context_metadata:
                intent = "Action"
            if isinstance(intent, str):
                intent_lower = intent.strip().lower()
                if intent_lower in ("record", "log_expense", "log expense", "log transaction", "log_transaction", "log"):
                    intent = "Record"
                elif intent_lower in ("action", "system"):
                    intent = "Action"
                elif intent_lower in ("chitchat", "chat"):
                    intent = "Chitchat"
                elif intent_lower in ("unknown",):
                    intent = "Unknown"
                else:
                    intent = "Chitchat"
            slots = parsed.get("slots") or {}
            
            # Determine record_type: prioritize top-level, then slots.type fallback
            raw_record_type = parsed.get("record_type")
            if not raw_record_type or raw_record_type not in ("Income", "Expense"):
                slot_type = slots.get("type", "").strip().lower() if slots.get("type") else ""
                raw_record_type = "Income" if slot_type == "income" else "Expense"

            # Determine action_type fallback
            act_type = parsed.get("action_type") if intent == "Action" else None
            if intent == "Action" and not act_type:
                text_lower = text.lower().strip()
                is_report = any(kw in text_lower for kw in ("tổng chi", "báo cáo", "thống kê", "chi tiêu tuần", "chi tiêu tháng", "chi tiêu hôm nay", "thu nhập tuần", "thu nhập tháng", "thu nhập hôm nay"))
                act_type = "REPORT_GENERAL" if is_report else "SEARCH_RECORD"

            result = {
                "intent": intent,
                "intent_confidence": 1.0,
                "text": text,
                "item": slots.get("item"),  # Always Vietnamese item name from LLM
                "category": slots.get("category"),
                "amount": slots.get("amount") or slots.get("value"),
                "record_type": raw_record_type if intent == "Record" else None,
                "action_type": act_type,
                "action_details": slots if intent == "Action" else {},
                "time_range": slots.get("time_range"),
                "mimo_emotion": parsed.get("emotion") or "neutral",
                "llm_emotion": parsed.get("emotion") or "neutral",
                "mascot_mood": parsed.get("emotion") or "neutral",
                "nlg_response": _sanitize_nlg_response(parsed.get("response") or "Mimo đã ghi nhận rồi nha."),
                "suggested_actions": parsed.get("suggested_actions") or (["Thêm giao dịch", "Xem báo cáo", "Quét hóa đơn"] if intent == "Chitchat" else None),
                "backend": "llm_unified"
            }
            if result["action_type"] in ("REPORT_GENERAL", "REPORT_COMPARE"):
                from src.nlu.time_parser import parse_time_range
                import re
                t_slots = slots.get("time_range")
                
                if isinstance(t_slots, str):
                    parts = re.split(r'(?i)\s+vs\s+|\s+với\s+', t_slots)
                    parsed_list = []
                    for p in parts:
                        pt = parse_time_range(text, [p.strip()])
                        if pt:
                            parsed_list.append(pt)
                    if result["action_type"] == "REPORT_COMPARE":
                        result["time_range"] = parsed_list
                    else:
                        result["time_range"] = parsed_list[0] if parsed_list else None
                elif isinstance(t_slots, list):
                    if result["action_type"] == "REPORT_COMPARE":
                        parsed_list = [parse_time_range(text, [str(t).strip()]) for t in t_slots]
                        parsed_list = [p for p in parsed_list if p]
                        result["time_range"] = parsed_list
                    else:
                        # REPORT_GENERAL: nếu LLM trả về danh sách 2 mốc như ["ngày 1", "ngày 10"], thử ghép lại bằng " đến "
                        joined_text = " đến ".join([str(t).strip() for t in t_slots])
                        pt_joined = parse_time_range(text, [joined_text, text])
                        if pt_joined:
                            result["time_range"] = pt_joined
                        else:
                            parsed_list = [parse_time_range(text, [str(t).strip()]) for t in t_slots]
                            parsed_list = [p for p in parsed_list if p]
                            result["time_range"] = parsed_list[0] if parsed_list else None
                else:
                    result["time_range"] = None
            
            return result
    except Exception as e:
        logger.error("Unified LLM NLU extraction failed: %s", e)
        
    return {
        "intent": "Chitchat",
        "intent_confidence": 0.0,
        "text": text,
        "category": None,
        "amount": None,
        "record_type": None,
        "action_type": None,
        "action_details": {},
        "mimo_emotion": "neutral",
        "nlg_response": "Mimo gặp chút trục trặc kết nối rồi, bạn thử lại sau nha.",
        "backend": "llm_fallback"
    }
