import json
import os
import sys
import unicodedata
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.llm.gemini_keys import call_gemini_with_key_fallback


def _no_accent(s: str) -> str:
    """Normalize string: lowercase, strip accents, replace đ with d, remove extra whitespaces."""
    s = s.lower().strip()
    s = s.replace("đ", "d").replace("Đ", "d")
    # Strip diacritics
    s = "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )
    # Clean multiple spaces or punctuation
    s = " ".join(s.split())
    return s


def main() -> None:
    # Load environment variables
    load_env_file(settings.ENV_PATH)

    print("=== Fetching Expanded Disambiguation Keywords from Gemini API ===")

    # Prepare request
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3-flash")

    system_prompt = (
        "Bạn là một chuyên gia ngôn ngữ học Tiếng Việt và chuyên gia phân tích hội thoại Chatbot FinTech.\n"
        "Hãy tìm kiếm và mở rộng danh sách các từ khóa tiếng Việt (bao gồm cả từ chuẩn, tiếng lóng, từ viết tắt, tiếng Anh thông dụng, teen code) dùng để mô tả:\n"
        "1. Động từ/hành vi tụ tập, đi chơi, hẹn hò, giao lưu xã hội (ví dụ: đi, hẹn, tụ tập, giao lưu, gặp, khao, mời, nhậu, quẩy, họp lớp, đi chơi...).\n"
        "2. Hoạt động giải trí, địa điểm, đồ uống hoặc đồ ăn liên quan đến tụ tập giải trí xã hội (ví dụ: cà phê, cafe, trà sữa, trà chanh, trà đào, quán ốc, quán lẩu, quán nướng, bida, karaoke, bar, pub, club, rạp phim, xem phim, xem bóng đá, dã ngoại, sinh nhật, tiệc tùng...).\n"
        "3. Từ chỉ đối tượng đi cùng, bạn bè, đồng nghiệp hoặc các mối quan hệ xã hội (ví dụ: với, cùng, bạn, bồ, crush, người yêu, ny, đồng nghiệp, anh em, chiến hữu, cạ cứng, bạn hiền, bạn học, sếp, đồng môn, nhóm, team, lớp...).\n\n"
        "Lưu ý quan trọng:\n"
        "- Danh sách từ khóa trả về phải là tiếng Việt có dấu chuẩn chỉnh.\n"
        "- Hãy tập trung vào các từ khóa xuất hiện trong các câu chat ghi chép chi tiêu tự nhiên của giới trẻ Việt Nam.\n"
        "- Trả về kết quả khớp hoàn toàn với cấu trúc JSON schema yêu cầu."
    )

    user_prompt = "Hãy cung cấp các từ khóa rộng rãi cho các hoạt động tụ tập giải trí xã hội."

    payload = {
        "contents": [
            {"parts": [{"text": user_prompt}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {
                    "verbs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Các động từ hoặc hành động rủ rê, đi chơi, tụ tập, khao, bao, hẹn hò."
                    },
                    "activities_places": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Các địa điểm ăn chơi, ăn nhậu, giải trí, trò chơi, rạp phim, trà sữa, cafe hoặc các món ăn/đồ uống đi kèm hội họp."
                    },
                    "companions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Từ chỉ mối quan hệ, bạn bè, đồng nghiệp, đối tượng đi cùng."
                    }
                },
                "required": ["verbs", "activities_places", "companions"]
            }
        },
        "systemInstruction": system_prompt
    }

    try:
        response_dict = call_gemini_with_key_fallback(gemini_model, payload)
        
        # Extract the text content from response
        # The structure returned by call_gemini is usually a dictionary containing candidate choices
        candidates = response_dict.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates found in Gemini response: {response_dict}")
        
        content_text = candidates[0]["content"]["parts"][0]["text"]
        result_json = json.loads(content_text)
        
        # Base/default lists to ensure we never lose current working matches
        base_verbs = ["di", "hen", "tu tap", "giao luu", "gap", "gap mat", "an uong", "party", "di choi", "di an", "di nhau", "lien hoan", "khao", "moi", "ru", "quay", "ca hat", "hat ho"]
        base_activities = ["ca phe", "cafe", "cf", "tra sua", "quan", "nhau", "bida", "phim", "bar", "pub", "club", "rap", "karaoke", "cinema", "tiem net", "net", "game", "xem phim", "lau", "nuong", "buffet", "quan oc", "an vat"]
        base_companions = ["voi", "cung", "ban", "ghe", "bo", "crush", "nguoi yeu", "ny", "dong nghiep", "anh em", "be", "gia dinh", "vo", "chong", "ox", "bx", "sep", "lop", "nhom", "team", "chien huu", "ca cung", "ban hien"]

        # Process and normalize the generated keywords
        raw_verbs = result_json.get("verbs", [])
        raw_activities = result_json.get("activities_places", [])
        raw_companions = result_json.get("companions", [])

        print(f"Generated raw from Gemini: {len(raw_verbs)} verbs, {len(raw_activities)} activities/places, {len(raw_companions)} companions.")

        normalized_verbs = sorted(list(set(base_verbs + [_no_accent(v) for v in raw_verbs if v.strip()])))
        normalized_activities = sorted(list(set(base_activities + [_no_accent(a) for a in raw_activities if a.strip()])))
        normalized_companions = sorted(list(set(base_companions + [_no_accent(c) for c in raw_companions if c.strip()])))

        output_data = {
            "raw_verbs": raw_verbs,
            "raw_activities_places": raw_activities,
            "raw_companions": raw_companions,
            "normalized_verbs": normalized_verbs,
            "normalized_activities_places": normalized_activities,
            "normalized_companions": normalized_companions
        }

        output_path = ROOT / "src" / "nlu" / "disambiguation_keywords.json"
        output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print(f"Successfully saved {len(normalized_verbs)} verbs, {len(normalized_activities)} activities, and {len(normalized_companions)} companion terms to {output_path}")

    except Exception as e:
        print(f"Error calling Gemini or processing result: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
