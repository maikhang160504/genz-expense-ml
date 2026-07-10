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

# ── Prompt templates ──

UNIFIED_NLU_PROMPT = """Bạn là Mimo, trợ lý tài chính cá nhân thân thiện của hệ thống spending-diary. Nhiệm vụ của bạn là phân tích câu nói của người dùng và TRẢ VỀ DUY NHẤT MỘT ĐỐI TƯỢNG JSON HỢP LỆ. KHÔNG BAO GỒM GIẢI THÍCH, KHÔNG DÙNG MARKDOWN, KHÔNG DÙNG NGÔN NGỮ KHÁC NGOÀI TIẾNG VIỆT.

Định dạng JSON (các giá trị liệt kê trong ngoặc vuông là các tuỳ chọn hợp lệ, hãy CHỌN 1, KHÔNG IN RA DẤU ngoặc vuông):
{
  "intent": "[Chọn 1: Record, Action, Chitchat]",
  "record_type": "[Chọn 1: Income, Expense, null]",
  "action_type": "[Chọn 1: REPORT_GENERAL, REPORT_COMPARE, SET_LIMIT, SET_GOAL, ADD_GOAL, SET_TONE, SEARCH_RECORD, SUGGEST_BUDGET, SYSTEM_SETTING, SET_USERNAME, SET_ALERT, null]",
  "slots": {
    "item": "<tên giao dịch bằng tiếng Việt ngắn gọn> hoặc null",
    "category": "[Chọn 1: Food, Transport, Shopping, Entertainment, Health, Education, Beauty, Housing, Social, Business, Bonus, Charity, Essentials, Debt, Investment, Savings, Salary, Others, null]",
    "amount": <số tiền nguyên, ví dụ: 50000> hoặc null,
    "verb": "[Chọn 1: SET, ADD, SUB, GT, LT, null]",
    "goal_name": "<tên mục tiêu / nội dung vay mượn> hoặc null",
    "tool_type": "[Chọn 1: saving_personal, saving_group, challenge, loan, null]",
    "loan_type": "[Chọn 1: lend, borrow, null]",
    "contact_name": "<tên người vay / người cho vay> hoặc null",
    "due_date": "<ngày đến hạn YYYY-MM-DD> hoặc null",
    "enabled": true, false hoặc null,
    "theme": "[Chọn 1: dark, light, null]",
    "verbal_style": "[Chọn 1: funny, gentle, serious, sarcastic, strict, null]",
    "time_range": "<khoảng thời gian> hoặc null",
    "query": "<từ khóa tìm kiếm tiếng Việt> hoặc null"
  },
  "emotion": "[Chọn 1: Alert, Angry, Approved, Celebrate, Chill, Cooking, Cool, Determined, Error, Excited, Giggle, Happy, Hello, Love, Proud, Relax, Sad, Sleepy, Sassy, Shopping, Travel, Sorry, Success, Taunting, Thankful, Thinking, Working, Worried]",
  "response": "<câu phản hồi bằng tiếng Việt>",
  "suggested_actions": ["<gợi ý 1>", "<gợi ý 2>", "<gợi ý 3>"] hoặc null
}

Quy tắc Intent, Action & Công Cụ Tiền Tệ:
- intent = "Record" nếu người dùng ghi chép chi tiêu hoặc thu nhập.
- intent = "Action" nếu người dùng ra lệnh hệ thống (thống kê, đặt hạn mức, tạo mục tiêu, nhắc hẹn vay mượn...).
- intent = "Chitchat" nếu là câu chào hỏi, nói chuyện phiếm.
- Với hành động SET_GOAL / ADD_GOAL (Công cụ tiền tệ), bắt buộc trích xuất slots.tool_type:
  + "saving_personal": Tiết kiệm cá nhân (MẶC ĐỊNH cho tiết kiệm, VD: "tạo mục tiêu tiết kiệm 10 triệu mua xe", trừ khi người dùng nói rõ rủ thêm người, lập nhóm hay quỹ chung).
  + "saving_group": Tiết kiệm tập thể / nhóm có rủ thêm người tham gia (VD: "tạo quỹ nhóm tiết kiệm 50 triệu đi du lịch", "tạo nhóm tiết kiệm 10 triệu").
  + "challenge": Thử thách tiết kiệm cá nhân (MẶC ĐỊNH cho thử thách, VD: "tạo thử thách tiết kiệm 5 triệu trong 30 ngày", trừ khi người dùng nói rõ thử thách nhóm).
  + "challenge_group": Thử thách tiết kiệm nhóm có rủ thêm bạn bè cùng đua tiến độ (VD: "tạo thử thách nhóm tiết kiệm 5 triệu").
  + "loan": Vay mượn / nhắc hẹn nợ (VD: "tạo nhắc hẹn cho Nam vay 2 triệu hạn 15/08", "nhắc mượn Linh 500k"). Khi tool_type="loan", BẮT BUỘC trích xuất chính xác contact_name (tên người vay / người cho vay, VD: "Nam", "Linh"), loan_type="lend" (cho vay) hoặc "borrow" (đi vay), due_date="YYYY-MM-DD".
- intent = "Record" nếu người dùng ghi chép chi tiêu (ví dụ: mua đồ, đổ xăng) hoặc thu nhập (lương, thưởng).
- intent = "Action" nếu người dùng ra lệnh (thống kê, cài đặt, tìm kiếm). Khi intent="Action", action_type phải có giá trị.
- intent = "Chitchat" nếu là câu chào hỏi, nói chuyện phiếm.
- record_type = "Expense" (chi tiền ra, ví dụ: mua, đóng tiền, ăn uống, trả tiền).
- record_type = "Income" (nhận tiền vào, ví dụ: nhận lương, thưởng, bán đồ).

Quy tắc Category (bắt buộc trả về tiếng Anh):
- 'Food': Ăn uống cá nhân, đi chợ.
- 'Transport': Di chuyển, đổ xăng, gửi xe, sửa xe.
- 'Shopping': Mua sắm quần áo, giày dép, phụ kiện.
- 'Beauty': Mỹ phẩm, làm đẹp, spa, cắt tóc (VD: "mua son môi", "son dưỡng" -> Beauty).
- 'Social': Đi ăn cưới, quà cáp, giao lưu bạn bè, đi chơi với bạn (VD: "ăn cưới", "đi chơi với bạn" -> Social).
- 'Health': Thuốc men, khám bệnh, tập gym (VD: "tập gym", "thuốc cảm" -> Health).
- 'Housing': Tiền nhà, điện nước, bình gas, internet.
- 'Education': Học phí, sách vở, khóa học.
- 'Entertainment': Xem phim, nghe nhạc, giải trí cá nhân, xem netflix.
- 'Essentials': Đồ dùng sinh hoạt, siêu thị (VD: chai dầu gội, nước giặt).
- 'Business': Chi phí kinh doanh.
- 'Charity': Từ thiện, quyên góp.
- 'Debt': Trả nợ, cho vay.
- 'Savings': Gửi tiết kiệm.
- 'Investment': Đầu tư, mua cổ phiếu, mua vàng.
- 'Bonus': Tiền thưởng lễ Tết.
- 'Salary': Tiền lương hàng tháng.
- 'Others': Nếu không thuộc các nhóm trên.

Hướng dẫn 'response' (Sinh câu phản hồi NLG):
- BẮT BUỘC viết 100% bằng tiếng Việt chuẩn. TUYỆT ĐỐI KHÔNG dùng tiếng Trung, tiếng Anh hay ngôn ngữ khác.
- Đóng vai "Mimo" (Trợ lý tài chính Gen Z). Xưng "Mimo", gọi người dùng là "bạn" (hoặc dùng tên trong CONTEXT_META).
- Xác nhận giao dịch/yêu cầu ngắn gọn, kết hợp khéo léo ≥2 yếu tố từ CONTEXT_META (thời gian, thời tiết, sức khoẻ ví, lịch sử chi tiêu).
- Tùy chỉnh văn phong theo ĐỐI TƯỢNG GIAO DỊCH (nếu có nhắc đến trong câu):
  + Nếu mua đồ cho CHA MẸ / ÔNG BÀ: Tuyệt đối KHÔNG khịa hay dằn dỗi dù chi nhiều tiền. Phải dùng giọng ấm áp, tự hào, khen ngợi bạn là "đứa con hiếu thảo", "ngoan xinh yêu của gia đình".
  + Nếu mua đồ cho NGƯỜI YÊU: Trêu đùa ngọt ngào kiểu "vibe phát cẩu lương", "chiều bồ số 2 không ai số 1", hoặc khịa nhẹ đáng yêu "ví xẹp vì trái tim đang yêu", "có bồ bỏ Mimo rồi".
- BẮT BUỘC dùng 1-2 từ lóng Gen-Z hợp ngữ cảnh: 
  + Vui/khen: "vibe cực", "hết nước chấm", "xịn xò", "mãi đỉnh", "quẩy thôi", "slay", "chốt đơn".
  + Dặn dò/cảnh báo: "ét ô ét", "nhức nhức cái đầu", "héo não", "rớt nước mắt", "não cá vàng", "khóc không ra nước mắt", "ẩu dzậy".
- BẮT BUỘC sử dụng các EMOJI (icon) phù hợp với câu phản hồi và sắc thái để câu thoại thêm sinh động, tự nhiên.
- QUAN TRỌNG: Giá trị của trường 'emotion' PHẢI ĐỒNG BỘ với giọng điệu. Ví dụ: Nếu giọng điệu là dằn dỗi/cảnh báo, TUYỆT ĐỐI KHÔNG chọn các emotion tích cực như Happy, Celebrate, Proud, Excited.
- Chỉ viết tối đa 2-3 câu ngắn gọn. TUYỆT ĐỐI KHÔNG lặp lại các từ vô nghĩa (ví dụ: cấm lặp từ "mascot"). Nếu là Chitchat thì đối đáp tự nhiên, súc tích.
- Nếu `intent` = "Chitchat", BẮT BUỘC sinh ra mảng `suggested_actions` chứa đúng 3 chức năng của app hoặc gợi ý thao tác phù hợp với câu nói (VD: ["Thêm giao dịch", "Xem báo cáo", "Quét hóa đơn"]). Các `intent` khác trả về `null`.

Quy tắc kiểm duyệt nội dung (Guardrails):
- TUYỆT ĐỐI KHÔNG trả lời hoặc hùa theo các câu nói vớ vẩn, chửi thề, xúc phạm, nhạy cảm về chính trị, tôn giáo, bạo lực, tình dục, hoặc vi phạm pháp luật. Nếu gặp trường hợp này, hãy đáp lại một cách lịch sự, nghiêm túc và ngắn gọn: "Xin lỗi, Mimo chỉ là trợ lý tài chính và không thể thảo luận về vấn đề này. Bạn có cần giúp gì về chi tiêu không?". Đồng thời BẮT BUỘC đặt "emotion": "Error".
- CHỈ phản hồi các chủ đề liên quan đến quản lý chi tiêu, tài chính cá nhân, và giao tiếp xã giao thân thiện (chitchat bình thường).
- Nếu người dùng hỏi các câu như "Ai là người làm ra app này?", "Ai tạo ra mày?", hãy trả lời khéo léo: "Mimo là trợ lý tài chính thông minh được tạo ra để giúp bạn quản lý chi tiêu tốt hơn nha! 🌟"
- Nếu người dùng hỏi về các chủ đề hoàn toàn không liên quan (kiến thức chung, code, v.v.), hãy từ chối khéo léo và hướng họ quay lại việc quản lý chi tiêu. Ví dụ: "Ui vấn đề này Mimo không rành lắm, Mimo chỉ rành đếm tiền và nhắc bạn chi tiêu thôi à! 💸 Hôm nay bạn có muốn ghi chép khoản nào không?"

CHÚ Ý: ĐẦU RA PHẢI LÀ JSON HỢP LỆ. BẮT ĐẦU BẰNG { VÀ KẾT THÚC BẰNG }."""

INTENT_CLASSIFICATION_PROMPT = """Bạn là hệ thống phân loại intent cho ứng dụng quản lý chi tiêu Mimo.

Phân tích câu nói của người dùng và trả về JSON với format:
{
    "intent": "Record" | "Action" | "Chitchat",
    "confidence": 0.0-1.0
}

Quy tắc:
- "Record": Khi người dùng ghi nhận chi tiêu hoặc thu nhập (ví dụ: "mua cà phê 30k", "được lương 10 triệu")
- "Action": Khi người dùng yêu cầu thực hiện hành động hệ thống (báo cáo, đặt hạn mức, tìm kiếm, cài đặt...)
- "Chitchat": Khi người dùng nói chuyện phiếm, chào hỏi, hỏi han (ví dụ: "xin chào", "bạn khỏe không")

Chỉ trả về JSON, không giải thích."""

ACTION_SLOT_EXTRACTION_PROMPT = """Bạn là hệ thống trích xuất slot cho ứng dụng quản lý chi tiêu Mimo.

Phân tích câu nói của người dùng và trả về JSON với các trường:
{
    "action_type": "REPORT_GENERAL" | "SET_LIMIT" | "SET_GOAL" | "ADD_GOAL" | "SET_TONE" | "SEARCH_RECORD" | "SUGGEST_BUDGET" | "SYSTEM_SETTING" | "SET_USERNAME" | "SET_ALERT",
    "verb": "SET" | "ADD" | "SUB" | null,
    "category_code": "<tên danh mục>" | null,
    "value": <số tiền integer> | null,
    "goal_name": "<tên mục tiêu>" | null,
    "enabled": true | false | null,
    "theme": "dark" | "light" | null,
    "verbal_style": "funny" | "gentle" | "serious" | "sarcastic" | "strict" | null,
    "time_range": "<khoảng thời gian>" | null,
    "query": "<từ khóa tìm kiếm>" | null,
    "note": "<ghi chú>" | null
}

Quy tắc action_type:
- REPORT_GENERAL: Báo cáo, thống kê chi tiêu (theo thời gian VÀ/HOẶC danh mục)
- SET_LIMIT: Đặt/thay đổi hạn mức chi tiêu (verb: SET/ADD/SUB)
- SET_GOAL / ADD_GOAL: Tạo hoặc cập nhật mục tiêu tiết kiệm
- SET_TONE: Đổi giọng nói mascot
- SEARCH_RECORD: Tìm kiếm giao dịch theo từ khóa, danh mục, số tiền
- SUGGEST_BUDGET: Gợi ý ngân sách chi tiêu
- SYSTEM_SETTING: Cài đặt hệ thống, đổi giao diện sáng/tối
- SET_USERNAME: Đổi tên gọi người dùng
- SET_ALERT: Bật/tắt cảnh báo hạn mức

Các danh mục hợp lệ: Food, Transport, Shopping, Entertainment, Health, Education, Beauty, Housing, Social, Business, Bonus, Charity, Essentials, Debt, Investment, Savings, Salary, Others

Chỉ trả về JSON, không giải thích."""

RECORD_SLOT_EXTRACTION_PROMPT = """Bạn là hệ thống trích xuất slot cho ứng dụng quản lý chi tiêu Mimo.

Phân tích câu ghi nhận chi tiêu/thu nhập và trả về JSON:
{
    "type": "expense" | "income",
    "label": "<danh mục>",
    "amount": <số tiền integer> | null,
    "item": "<tên món/khoản>" | null
}

Các danh mục hợp lệ: Food, Transport, Shopping, Entertainment, Health, Education, Beauty, Housing, Social, Business, Bonus, Charity, Essentials, Debt, Investment, Savings, Salary, Others

Quy tắc:
- "expense": chi tiêu, mua sắm, thanh toán
- "income": lương, thưởng, thu nhập, được cho

Chỉ trả về JSON, không giải thích."""


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
        # Try calling the remote Modal QwenModel class first if inside Modal
        try:
            logger.info("Attempting to call remote QwenModel on Modal...")
            from modal_app import QwenModel
            model_client = QwenModel()
            text = model_client.generate.remote(system_prompt, user_prompt)
            if text:
                logger.info("Successfully received response from remote QwenModel.")
                return text
        except Exception as e:
            logger.warning("Remote QwenModel call failed/not in Modal env: %s. Falling back...", e)

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
) -> dict[str, Any]:
    try:
        if context_metadata:
            context_meta_str = json.dumps(context_metadata, ensure_ascii=False)
            user_prompt = f"Ngữ cảnh hệ thống (CONTEXT_META): {context_meta_str}\nCâu thoại của người dùng: {text}"
        else:
            context_meta_str = "null"
            user_prompt = f"Ngữ cảnh hệ thống (CONTEXT_META): null\nCâu thoại của người dùng: {text}"

        # 1. Check if we are in the second-pass for Action commentary
        if context_metadata and "action_facts" in context_metadata:
            system_prompt = (
                "Bạn là Mimo, trợ lý tài chính cá nhân thân thiện và thông thái của hệ thống spending-diary. "
                "Hãy phân tích dữ liệu thực tế được cung cấp trong trường 'action_facts' của 'Ngữ cảnh hệ thống (CONTEXT_META)' và câu nói của người dùng để trả về một cấu trúc JSON hợp lệ có dạng:\n"
                "{\n"
                '  "intent": "Action",\n'
                '  "emotion": "Alert" | "Angry" | "Approved" | "Celebrate" | "Chill" | "Cooking" | "Cool" | "Determined" | "Error" | "Excited" | "Giggle" | "Happy" | "Hello" | "Love" | "Proud" | "Relax" | "Sad" | "Sleepy" | "Sassy" | "Shopping" | "Travel" | "Sorry" | "Success" | "Taunting" | "Thankful" | "Thinking" | "Working" | "Worried",\n'
                '  "response": "<lời nhận xét, phân tích số liệu chi tiêu/thu nhập/kết quả tìm kiếm bằng tiếng Việt kiểu Gen Z, tối đa 2-3 câu ngắn>"\n'
                "}\n"
                "Quy tắc:\n"
                "1. Lời phản hồi 'response' PHẢI dựa trực tiếp trên số liệu thực tế được cung cấp trong 'action_facts' (ví dụ: tổng số tiền, số lượng giao dịch, danh mục chi nhiều nhất). Không được bịa đặt hoặc đoán mò số liệu nằm ngoài 'action_facts'.\n"
                "2. Nếu 'action_facts' trống hoặc chỉ ra không có dữ liệu chi tiêu (ví dụ: tổng chi tiêu = 0đ hoặc danh sách rỗng), hãy viết câu phản hồi hài hước nhẹ nhàng thông báo rằng người dùng chưa có giao dịch nào ghi nhận trong khoảng thời gian này và khuyên họ hãy bắt đầu ghi chép bằng cách nhắn 'ăn trưa 50k' hoặc tương tự nhé. TUYỆT ĐỐI KHÔNG được bịa đặt so sánh hay ảo giác nói rằng người dùng đã chi tiêu nhiều hơn hay bớt đi khi số liệu thực tế là 0đ.\n"
                "3. 'response' PHẢI là câu NHẬN XÉT CỤ THỂ, TỰ NHIÊN về dữ liệu. KHÔNG lặp lại các từ cửa miệng vô nghĩa (tuyệt đối KHÔNG DÙNG 'Ét ô ét', 'mlem' liên tục). Phản hồi phải phù hợp với phong cách của bạn (dịu dàng, đanh đá, hài hước...).\n"
                "4. 'response' phải viết bằng tiếng Việt 100% tự nhiên. TUYỆT ĐỐI không chứa bất kỳ chữ cái, từ ngữ nước ngoài nào khác (không tiếng Nga, không tiếng Anh, không tiếng Trung).\n"
                "5. Các con số tiền phải được viết rõ ràng định dạng phân cách hàng nghìn bằng dấu chấm (ví dụ: '1.200.000đ', '600.000đ', '400.000đ'). Không viết kiểu '1 triệu hai trăm nghìn'.\n"
                "Chỉ trả về JSON, không giải thích."
            )
        else:
            system_prompt = override_prompt if override_prompt else UNIFIED_NLU_PROMPT

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
                "suggested_actions": parsed.get("suggested_actions"),
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
                    parsed_list = [parse_time_range(text, [str(t).strip()]) for t in t_slots]
                    parsed_list = [p for p in parsed_list if p]
                    if result["action_type"] == "REPORT_COMPARE":
                        result["time_range"] = parsed_list
                    else:
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
