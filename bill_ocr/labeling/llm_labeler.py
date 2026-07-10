"""
LLM labeling module: auto-label bill items, correct TOTAL_COST, and classify category.
Supports google-genai library or direct HTTP fallback using standard urllib.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Standard categories defined in Mimo
CATEGORIES = [
    "Food", "Transport", "Shopping", "Entertainment", "Health", 
    "Education", "Beauty", "Housing", "Social", "Business", 
    "Essentials", "Others"
]

BILL_LABEL_PROMPT = """Bạn là hệ thống gán nhãn hóa đơn cho ứng dụng quản lý chi tiêu Mimo.

KIE Fields trích xuất từ mô hình PICK KIE:
{kie_fields}

Danh sách toàn bộ các dòng chữ OCR:
{ocr_lines}

Nhiệm vụ của bạn:
1. Trích xuất danh sách sản phẩm/món hàng chi tiết dạng: tên gốc (raw), tên chuẩn hóa (label), đơn giá/thành tiền (price).
2. Kiểm tra lại giá trị TOTAL_COST từ PICK. Nếu tổng tiền của các sản phẩm cộng lại không khớp với TOTAL_COST gốc, hoặc TOTAL_COST gốc bị nhận diện sai/thiếu chữ số (ví dụ 8,000 thay vì 80,000), hãy tự sửa và đưa ra giá trị đúng (total_cost_corrected).
3. Phân tích danh sách sản phẩm để nhận dạng categoryCode tổng thể của hóa đơn (Food, Transport, Shopping, etc.).

Trả về định dạng JSON thuần túy như sau:
{{
    "items": [
        {{"raw": "tên sản phẩm gốc", "label": "tên chuẩn hóa tiếng Việt sạch", "price": 45000}}
    ],
    "total_cost_original": 45000,
    "total_cost_corrected": 45000,
    "total_cost_fixed": false,
    "bill_category": "Food",
    "confidence": 0.95
}}

Chú ý: Chỉ trả về JSON duy nhất, không giải thích, không kèm markdown block ```json."""


def _call_gemini_http(prompt: str, system_instruction: str = "") -> str:
    """Call Gemini using raw HTTP request to minimize external library dependencies."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Try gemini-2.5-flash as default, fallback to gemini-1.5-flash
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    contents = []
    if system_instruction:
        contents.append({
            "role": "user",
            "parts": [{"text": f"System Instruction: {system_instruction}\n\nUser Request: {prompt}"}]
        })
    else:
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

    payload = {
        "contents": contents,
        "generationConfig": {
            "responseMimeType": "application/json" if "JSON" in prompt else "text/plain"
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            raise ValueError(f"Unexpected API response shape: {res_data}")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        logger.error("Gemini HTTP Error %d: %s", e.code, error_body)
        raise RuntimeError(f"Gemini API returned error {e.code}: {error_body}") from e


def auto_label_bill(ocr_lines: list[str], kie_fields: dict[str, Any]) -> dict[str, Any]:
    """
    Run LLM auto-labeler.
    
    Args:
        ocr_lines: List of clean text lines from OCR
        kie_fields: Dict containing KIE extracted fields (e.g. SELLER, TOTAL_COST, etc.)
        
    Returns:
        Auto-labeling result dict.
    """
    ocr_text = "\n".join(f"- {line}" for line in ocr_lines)
    kie_text = json.dumps(kie_fields, ensure_ascii=False, indent=2)
    
    prompt = BILL_LABEL_PROMPT.format(kie_fields=kie_text, ocr_lines=ocr_text)
    
    # Try calling via client if installed, or fallback to HTTP
    try:
        try:
            import google.genai as genai
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if api_key:
                client = genai.Client(api_key=api_key)
                model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                response_text = response.text
            else:
                response_text = _call_gemini_http(prompt)
        except ImportError:
            response_text = _call_gemini_http(prompt)
            
        # Parse JSON
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(lines)
            
        result = json.loads(response_text)
        
        # Validate category
        category = result.get("bill_category")
        if category not in CATEGORIES:
            result["bill_category"] = "Others"
            
        return result
        
    except Exception as e:
        logger.error("LLM Auto labeling failed: %s", e)
        # Fallback heuristic
        total_cost = None
        if kie_fields.get("TOTAL_COST"):
            try:
                # Basic normalization of total cost
                num_str = re.sub(r"[^\d]", "", str(kie_fields["TOTAL_COST"]))
                total_cost = int(num_str) if num_str else None
            except Exception:
                pass
                
        return {
            "items": [],
            "total_cost_original": total_cost,
            "total_cost_corrected": total_cost,
            "total_cost_fixed": False,
            "bill_category": "Others",
            "confidence": 0.0,
            "error": str(e)
        }
