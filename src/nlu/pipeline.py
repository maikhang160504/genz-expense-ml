import re
import unicodedata

from pyvi import ViTokenizer

from src.nlu.text import clean_content, extract_amounts, normalize_text
from src.nlu.ner import (
    CATEGORY_LABELS,
    extract_ner_slots,
    map_category_to_label,
    map_product_brand_hint,
    select_record_item_from_slots,
)
from src.nlu.multi_record import extract_multi_records
from src.nlu.action_query import is_action_query, report_general_action_type
from src.nlu.income_phrase import is_clear_income_phrase
from pipeline.text_preprocessing import clean_category_text

MONEY_RE = re.compile(
    r"(\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|triệu|trieu|củ|cu))",
    re.IGNORECASE,
)

_CHA_ME_KEYWORDS = {
    "mẹ", "má", "ba", "bố", "cha", "cụ", "ông bà",
    "mẹ ruột", "bố ruột", "ba ruột", "bà nội", "bà ngoại", "ông nội", "ông ngoại",
    "báo hiếu", "phụng dưỡng", "cho mẹ", "cho ba", "cho bố", "cho cha",
    "tặng mẹ", "tặng ba", "tặng bố", "tiền mẹ", "tiền ba", "tiền bố",
    "mẹ ơi", "ba ơi", "bố ơi", "nuôi mẹ", "nuôi ba", "lo cho mẹ",
    "thuốc cho mẹ", "thuốc cho ba", "viện phí", "bệnh viện",
}

_NGUOI_YEU_KEYWORDS = {
    "bồ", "người yêu", "ny", "crush", "bạn gái", "bạn trai",
    "bạn ghệ", "ghệ", "gf", "bf", "partner",
    "em yêu", "anh yêu", "yêu ơi", "baby", "babe",
    "hẹn hò", "date", "kỷ niệm", "anniversary", "valentine",
    "dẫn bồ", "đưa bồ", "cho bồ", "tặng bồ", "mua cho bồ",
    "dẫn người yêu", "đưa người yêu", "mua cho người yêu",
}


def _no_accent(s: str) -> str:
    """Strip diacritics for accent-insensitive matching."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


_CHA_ME_NORM = {_no_accent(kw) for kw in _CHA_ME_KEYWORDS}
_NGUOI_YEU_NORM = {_no_accent(kw) for kw in _NGUOI_YEU_KEYWORDS}


def detect_relationship_tag(text: str) -> str | None:
    """Quét từ khóa nhạy cảm để gán tag ẩn CHA_ME hoặc NGUOI_YEU.
    Hỗ trợ cả text có dấu lẫn không dấu (GenZ typing).
    Bill scan KHÔNG gọi hàm này (không có text chủ quan của user).
    """
    lowered = text.lower()
    norm = _no_accent(lowered)
    for kw in _NGUOI_YEU_NORM:
        if kw in norm:
            return "NGUOI_YEU"
    for kw in _CHA_ME_NORM:
        if kw in norm:
            return "CHA_ME"
    return None


def classify_intent(text: str, intent_model: dict) -> tuple[str, float | None, dict[str, float]]:
    """Trả về (intent, độ tin cậy đã hiệu chỉnh nếu encoder, phân phối xác suất)."""
    if intent_model.get("backend") == "encoder":
        from src.nlu.encoder_runtime import predict_intent_encoder

        return predict_intent_encoder(intent_model["bundle"], text, MONEY_RE)

    vectorizer, model = intent_model["vectorizer"], intent_model["model"]
    vec = vectorizer.transform([text])
    pred_raw = str(model.predict(vec)[0])
    dist: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        for c, p in zip(model.classes_, proba):
            dist[str(c)] = float(p)
    intent = pred_raw
    if MONEY_RE.search(text) and pred_raw == "Chitchat":
        intent = "Record"
    if dist:
        conf = dist.get(str(intent), max(dist.values()))
    else:
        conf = None
    return intent, conf, dist


def parse_action_details(text: str) -> dict:
    normalized = normalize_text(ViTokenizer.tokenize(text).replace("_", " "))
    verbs_set = ["dat", "đặt", "dat lai", "đặt lại", "thiet lap", "thiết lập", "chot", "chốt"]
    verbs_inc = ["tang", "tăng", "them", "thêm", "nang", "nâng"]
    verbs_dec = ["giam", "giảm", "ha", "hạ", "bot", "bớt"]

    verb = "SET"
    if any(v in normalized for v in verbs_inc):
        verb = "INCREASE"
    elif any(v in normalized for v in verbs_dec):
        verb = "DECREASE"
    elif any(v in normalized for v in verbs_set):
        verb = "SET"

    categories = list(CATEGORY_LABELS)
    target = None
    for cat in categories:
        if cat in normalized:
            target = cat
            break

    amounts = extract_amounts(text)
    value = amounts[0] if amounts else None

    target_type = None
    if any(k in normalized for k in ["han muc", "hạn mức", "gioi han", "giới hạn", "ngan sach", "ngân sách"]):
        target_type = "LIMIT"
    elif any(k in normalized for k in ["muc tieu", "mục tiêu", "tiet kiem", "tiết kiệm"]):
        target_type = "GOAL"
    elif any(k in normalized for k in ["phong cach", "phong cách", "giong", "giọng", "tone"]):
        target_type = "STYLE"
    elif any(k in normalized for k in ["thu nhap", "thu nhập", "luong", "lương"]):
        target_type = "INCOME"
    elif any(k in normalized for k in ["doi ten", "đổi tên", "ten", "tên", "goi", "gọi"]):
        target_type = "USERNAME"

    style_value = None
    for style in ["dận dữ", "tức giận", "vui vẻ", "vui ve", "dễ thương", "de thuong", "nghiêm túc", "nghiem tuc"]:
        if style in normalized:
            style_value = style
            break

    name_value = None
    name_match = re.search(
        r"\b(?:goi minh la|gọi mình là|ten cua toi la|tên của tôi là|doi ten thanh|đổi tên thành)\s+(?P<name>.+)$",
        normalized,
    )
    if name_match:
        name_value = name_match.group("name").strip()

    return {
        "verb": verb,
        "target": target,
        "target_type": target_type,
        "value": value,
        "unit": "VND",
        "style": style_value,
        "name": name_value,
    }


INCOME_TYPE_LABELS = {"salary", "bonus", "investment", "business"}


def build_action_details_from_slots(slots: dict | None, text: str, category_mapper=None) -> dict:
    slots = slots or {}
    verb = slots.get("VERB", [None])[0]
    target = slots.get("CATEGORY", [None])[0]
    if category_mapper and target:
        mapped_target = category_mapper(target)
        if mapped_target:
            target = mapped_target
    target_raw = slots.get("TARGET", [None])[0]
    action_type_token = slots.get("ACTION_TYPE", [None])[0]
    time_values = slots.get("TIME", [])
    amount_text = slots.get("AMOUNT", [None])[0]
    amount_values = extract_amounts(amount_text or text)
    value = amount_values[0] if amount_values else None

    target_type = None
    if target_raw in {"hạn mức", "giới hạn", "ngân sách"}:
        target_type = "LIMIT"
    elif target_raw == "mục tiêu":
        target_type = "GOAL"

    return {
        "verb": verb,
        "target": target,
        "target_type": target_type,
        "value": value,
        "unit": "VND" if value is not None else None,
        "time": time_values or None,
        "action_type_token": action_type_token,
    }


def run_nlu(
    user_text: str,
    intent_model,
    category_model,
    action_type_model,
    record_type_model,
    sentiment_model,
    ner_model,
    *,
    run_llm: bool = False,
    user_id: str | None = None,
) -> dict:
    intent, intent_conf, intent_proba = classify_intent(user_text, intent_model)
    if is_action_query(user_text) and intent != "Action":
        intent = "Action"
    amounts = extract_amounts(user_text)
    content = clean_content(user_text)

    result = {
        "text": user_text,
        "intent": intent,
        "intent_backend": intent_model.get("backend"),
        "intent_confidence": intent_conf,
        "intent_proba": intent_proba or None,
        "clean_content": content,
        "multi_records": [],
        "multi_record_task": False,
        "relationship_tag": detect_relationship_tag(user_text),
    }

    if intent == "Record":
        result["amount_spent"] = amounts[0] if amounts else None
        rec_pred: str | None = None
        if record_type_model.get("backend") == "encoder" and record_type_model.get("bundle"):
            from src.nlu.encoder_runtime import predict_record_type_encoder

            rec_pred = predict_record_type_encoder(record_type_model["bundle"], user_text)
            result["record_type_backend"] = "encoder"
        elif record_type_model.get("backend") == "tfidf" and record_type_model.get("vectorizer"):
            rec_vec = record_type_model["vectorizer"].transform([clean_category_text(user_text)])
            rec_pred = str(record_type_model["model"].predict(rec_vec)[0]).lower()
            result["record_type_backend"] = "tfidf"
        if rec_pred is not None:
            result["record_type"] = "Income" if rec_pred == "income" else "Expense"
        else:
            result["record_type"] = None
        if is_clear_income_phrase(user_text):
            result["record_type"] = "Income"
        result["income_type"] = None

        slots = None
        if ner_model:
            slots = extract_ner_slots(user_text, ner_model)
            if slots:
                result["slots"] = slots
        ner_category = None
        if slots:
            ner_category = select_record_item_from_slots(slots)
        mapped_category = map_category_to_label(ner_category) or map_product_brand_hint(ner_category)
        result["item"] = ner_category

        raw_category = None
        if category_model.get("backend") == "encoder" and category_model.get("bundle"):
            from src.nlu.encoder_runtime import predict_category_encoder

            raw_category = predict_category_encoder(category_model["bundle"], user_text)
            result["category_backend"] = "encoder"
        elif category_model.get("backend") == "tfidf" and category_model.get("vectorizer"):
            cat_input = clean_category_text(user_text)
            cat_vec = category_model["vectorizer"].transform([cat_input])
            raw_category = category_model["model"].predict(cat_vec)[0]
            result["category_backend"] = "tfidf"
        if raw_category is not None:
            result["category"] = raw_category
            if result["record_type"] == "Income" and raw_category:
                key = normalize_text(str(raw_category))
                if key in INCOME_TYPE_LABELS:
                    result["income_type"] = key
        else:
            result["category"] = None

        result["category_ner"] = mapped_category
        if result["category"] and mapped_category:
            result["category_agreement"] = str(result["category"]).lower() == str(mapped_category).lower()
        else:
            result["category_agreement"] = None

        multi = extract_multi_records(user_text, ner_model, category_model, record_type_model)
        result["multi_records"] = multi
        result["multi_record_task"] = len(multi) >= 2

    elif intent == "Action":
        if action_type_model.get("backend") == "encoder":
            from src.nlu.encoder_runtime import predict_action_type_encoder

            result["action_type"] = predict_action_type_encoder(action_type_model["bundle"], user_text)
        elif action_type_model.get("backend") == "tfidf" and action_type_model.get("vectorizer"):
            act_vec = action_type_model["vectorizer"].transform([user_text])
            result["action_type"] = str(action_type_model["model"].predict(act_vec)[0])
        else:
            result["action_type"] = None
        if str(result.get("action_type")) == "SYSTEM_SETTING":
            result["action_type"] = "Setting"
        rg = report_general_action_type(user_text)
        if rg and str(result.get("action_type")) != rg:
            result["action_type"] = rg
        result["action_type_backend"] = action_type_model.get("backend")
        result["action_param"] = amounts[0] if amounts else None
        slots = None
        if ner_model:
            slots = extract_ner_slots(user_text, ner_model)
            if slots:
                result["slots"] = slots
            result["action_details"] = build_action_details_from_slots(slots, user_text, map_category_to_label)
        else:
            result["action_details"] = None

    else:
        # Chitchat: tone + gợi ý hành động do LLM (không phân Positive/Negative/Neutral bằng NLU)
        result["sentiment"] = None
        result["sentiment_backend"] = "llm"
        result["chitchat_response_via"] = "llm"

    return result
