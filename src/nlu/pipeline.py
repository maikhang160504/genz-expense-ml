import re
import unicodedata
import json
from pathlib import Path

from pyvi import ViTokenizer

from src.nlu.text import clean_content, extract_amounts, normalize_text
from src.nlu.ner import (
    extract_ner_slots,
    map_product_brand_hint,
    select_record_item_from_slots,
)
from src.nlu.multi_record import extract_multi_records
from src.nlu.action_query import (
    is_action_query,
    report_general_action_type,
    suggest_budget_action_type,
    is_limit_or_goal_action,
    is_system_or_delete_action,
)
from src.nlu.time_parser import parse_time_range
from src.nlu.income_phrase import is_clear_income_phrase
from pipeline.text_preprocessing import clean_category_text



MONEY_RE = re.compile(
    r"(\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|tr|triệu|trieu|củ|cu))",
    re.IGNORECASE,
)

def _no_accent(s: str) -> str:
    """Strip diacritics for accent-insensitive matching."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


def detect_relationship_tag(text: str, slots: dict | None = None) -> str | None:
    """Phát hiện quan hệ CHA_ME hoặc NGUOI_YEU dựa trên slots COMPANION hoặc fallback."""
    if slots and "COMPANION" in slots:
        for val in slots["COMPANION"]:
            norm = _no_accent(val.lower())
            if any(w in norm for w in ["me", "ma", "ba", "bo", "cha", "cu", "ong ba", "bo me", "ba me"]):
                return "CHA_ME"
            if any(w in norm for w in ["bo", "nguoi yeu", "ny", "crush", "ban gai", "ban trai", "ghe", "gf", "bf"]):
                return "NGUOI_YEU"
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

from src.nlu.models import predict_category_from_text


INCOME_TYPE_LABELS = {"salary", "bonus", "investment", "business"}


def build_action_details_from_slots(slots: dict | None, text: str, category_mapper=None) -> dict:
    slots = slots or {}
    raw_verb = slots.get("VERB", [None])[0]
    
    norm_text = _no_accent(text.lower()).replace("đ", "d")
    verb = "SET"
    
    if any(kw in norm_text for kw in ["cong them", "tang them", "bo sung", "them", "tang", "bu vao"]):
        verb = "ADD"
    elif any(kw in norm_text for kw in ["bot", "giam", "tru di", "giam di"]):
        verb = "SUB"
    elif any(kw in norm_text for kw in ["thay doi thanh", "dat lai", "thiet lap", "dat", "thanh", "la", "chot"]):
        verb = "SET"
    elif raw_verb:
        norm_v = _no_accent(raw_verb.lower()).replace("đ", "d")
        if any(kw in norm_v for kw in ["them", "cong", "tang", "bu", "bo sung"]):
            verb = "ADD"
        elif any(kw in norm_v for kw in ["bot", "giam", "tru"]):
            verb = "SUB"
        else:
            verb = "SET"
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


def find_matching_correction(
    user_text: str,
    user_corrections: list[dict],
    vectorizer,
    threshold: float = 0.85,
) -> dict | None:
    if not user_corrections:
        return None

    cleaned_input = clean_category_text(user_text).strip().lower()

    # 1. Layer 1: Exact Match
    for c in user_corrections:
        c_text = c.get("text", "").strip().lower()
        if c_text == cleaned_input:
            match = c.copy()
            match["match_type"] = "exact"
            return match

    # 2. Layer 2: Semantic Similarity
    if not vectorizer:
        return None

    texts_to_compare = []
    valid_corrections = []
    for c in user_corrections:
        c_text = c.get("text", "").strip().lower()
        if c_text:
            texts_to_compare.append(c_text)
            valid_corrections.append(c)

    if not texts_to_compare:
        return None

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        # Transform both using the loaded global NLU vectorizer
        input_vec = vectorizer.transform([cleaned_input])
        corrections_vec = vectorizer.transform(texts_to_compare)

        # Calculate cosine similarity
        similarities = cosine_similarity(input_vec, corrections_vec)[0]

        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            match = valid_corrections[best_idx].copy()
            match["match_type"] = "similarity"
            match["similarity_score"] = best_score
            return match
    except Exception as e:
        print(f"[NLU Personalization] Error calculating similarity: {e}")

    return None


def is_entertainment_cafe(text: str) -> bool:
    norm = _no_accent(text.lower()).replace("đ", "d")
    # Matches "di/hen/tu tap ca phe/cafe/cf" or "ca phe/cafe/cf ... voi/ban/be/bo/ny/crush/nguoi yeu"
    pattern = re.compile(
        r"\b(di|hen|tu\s+tap)\s+(ca\s+phe|cafe|cf)\b|\b(ca\s+phe|cafe|cf)\b.*\b(voi|ban|be|bo|ny|crush|nguoi\s+yeu)\b",
        re.IGNORECASE
    )
    return bool(pattern.search(norm))


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
    user_corrections: list[dict] | None = None,
) -> dict:
    intent, intent_conf, intent_proba = classify_intent(user_text, intent_model)
    if is_action_query(user_text) and intent != "Action":
        intent = "Action"
    if is_limit_or_goal_action(user_text) and intent != "Action":
        intent = "Action"
    if is_system_or_delete_action(user_text) and intent != "Action":
        intent = "Action"

    # Personalization Hybrid Layer: exact match or semantic similarity match
    match = None
    if user_corrections:
        vectorizer = category_model.get("vectorizer")
        match = find_matching_correction(user_text, user_corrections, vectorizer)

    if match:
        if match.get("intent"):
            intent = match["intent"]
            intent_conf = match.get("similarity_score", 1.0)

    amounts = extract_amounts(user_text)
    content = clean_content(user_text)

    # Extract slots early for all intents (useful for relationship tags and actions)
    slots = None
    if ner_model:
        slots = extract_ner_slots(user_text, ner_model)

    result = {
        "text": user_text,
        "intent": intent,
        "intent_backend": intent_model.get("backend") if not match or not match.get("intent") else f"user_{match['match_type']}",
        "intent_confidence": intent_conf,
        "intent_proba": intent_proba or None,
        "clean_content": content,
        "multi_records": [],
        "multi_record_task": False,
        "relationship_tag": detect_relationship_tag(user_text, slots),
    }
    if slots:
        result["slots"] = slots

    if intent == "Record":
        result["amount_spent"] = amounts[0] if amounts else None
        rec_pred: str | None = None
        if match and match.get("record_type"):
            result["record_type"] = match["record_type"]
            result["record_type_backend"] = f"user_{match['match_type']}"
        else:
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

        ner_category = None
        if slots:
            ner_category = select_record_item_from_slots(slots)
        mapped_category = predict_category_from_text(ner_category, category_model)
        result["item"] = ner_category

        raw_category = None
        if match and match.get("category_code"):
            raw_category = match["category_code"]
            result["category_backend"] = f"user_{match['match_type']}"
        else:
            if category_model.get("backend") == "encoder" and category_model.get("bundle"):
                from src.nlu.encoder_runtime import predict_category_encoder

                raw_category = predict_category_encoder(category_model["bundle"], user_text)
                result["category_backend"] = "encoder"
            elif category_model.get("backend") == "tfidf" and category_model.get("vectorizer"):
                cat_input = clean_category_text(user_text)
                cat_vec = category_model["vectorizer"].transform([cat_input])
                raw_category = category_model["model"].predict(cat_vec)[0]
                result["category_backend"] = "tfidf"

        # Resolve category: prefer mapped_category from NER slots, fallback to raw_category from full text
        final_category = mapped_category if mapped_category else raw_category
        if is_entertainment_cafe(user_text):
            final_category = "Entertainment"
                
        if final_category is not None:
            result["category"] = final_category
            if result["record_type"] == "Income" and final_category:
                key = normalize_text(str(final_category))
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
        sb = suggest_budget_action_type(user_text)
        if sb:
            result["action_type"] = sb
        else:
            rg = report_general_action_type(user_text)
            if rg and str(result.get("action_type")) != rg:
                result["action_type"] = rg
        result["action_type_backend"] = action_type_model.get("backend")
        result["action_param"] = amounts[0] if amounts else None
        
        if ner_model:
            result["action_details"] = build_action_details_from_slots(
                slots, user_text, lambda x: predict_category_from_text(x, category_model)
            )
        else:
            result["action_details"] = None
        time_slots = (result.get("action_details") or {}).get("time")
        if isinstance(time_slots, str):
            time_slots = [time_slots]
        result["time_range"] = parse_time_range(user_text, time_slots)

    else:
        # Chitchat: tone + gợi ý hành động do LLM (không phân Positive/Negative/Neutral bằng NLU)
        result["sentiment"] = None
        result["sentiment_backend"] = "llm"
        result["chitchat_response_via"] = "llm"

    return result
