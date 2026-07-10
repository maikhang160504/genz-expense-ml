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
from src.nlu.time_parser import parse_time_range
from src.nlu.llm_intent_handler import run_llm_fallback
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
    """Chỉ dùng NER slot COMPANION — không keyword fallback."""
    if not slots or "COMPANION" not in slots:
        return None
    # Model NER gán COMPANION; backend chỉ pass-through tag nếu đã có trong slot meta
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
    if dist:
        conf = dist.get(str(intent), max(dist.values()))
    else:
        conf = None
    return intent, conf, dist

from src.nlu.models import predict_category_from_text
from src.nlu.action_slots import predict_action_details


INCOME_TYPE_LABELS = {"salary", "bonus", "investment", "business"}


def _sanitize_match_intent(match: dict | None) -> dict | None:
    if not match:
        return match
    intent = match.get("intent")
    if intent:
        intent_lower = str(intent).strip().lower()
        if intent_lower in ("record", "log_expense", "log expense", "log transaction", "log_transaction", "log"):
            match["intent"] = "Record"
        elif intent_lower in ("action", "system"):
            match["intent"] = "Action"
        elif intent_lower in ("chitchat", "chat"):
            match["intent"] = "Chitchat"
        elif intent_lower in ("unknown",):
            match["intent"] = "Unknown"
        else:
            match["intent"] = "Record"
            
    rt = match.get("record_type")
    if rt:
        rt_lower = str(rt).strip().lower()
        if rt_lower in ("expense", "spending", "chi"):
            match["record_type"] = "Expense"
        elif rt_lower in ("income", "earnings", "thu"):
            match["record_type"] = "Income"
    return match


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
            return _sanitize_match_intent(match)

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
            return _sanitize_match_intent(match)
    except Exception as e:
        print(f"[NLU Personalization] Error calculating similarity: {e}")

    return None


def run_nlu(
    user_text: str,
    intent_model,
    category_model,
    action_type_model,
    record_type_model,
    sentiment_model,
    ner_model,
    action_slots_model=None,
    *,
    run_llm: bool = False,
    user_id: str | None = None,
    user_corrections: list[dict] | None = None,
    profile: dict | None = None,
) -> dict:
    from src.nlu.models import get_inference_backend
    if get_inference_backend() == "llm":
        from src.nlu.llm_intent_handler import run_llm_nlu
        from src.nlg.context_meta import build_unified_llm_context
        
        # Priority 1: Check personalization layer before calling the heavy LLM
        match = None
        if user_corrections:
            vectorizer = category_model.get("vectorizer")
            match = find_matching_correction(user_text, user_corrections, vectorizer)
            
        if match:
            intent = match.get("intent", "Record")
            category = match.get("category_code")
            record_type = match.get("record_type")
            amounts = extract_amounts(user_text)
            amount = amounts[0] if amounts else None
            
            vietCategoryMap = {
                'Food': 'Ăn uống', 'Transport': 'Di chuyển', 'Housing': 'Nhà ở', 'Shopping': 'Mua sắm',
                'Entertainment': 'Giải trí', 'Health': 'Sức khỏe', 'Education': 'Giáo dục', 'Others': 'Tiêu dùng khác',
                'Other': 'Tiêu dùng khác', 'Essentials': 'Thiết yếu', 'Beauty': 'Làm đẹp', 'Social': 'Xã hội',
                'Salary': 'Lương', 'Bonus': 'Thưởng', 'Business': 'Kinh doanh'
            }
            vietCat = vietCategoryMap.get(category, category or "khoản chi")
            
            if amount:
                formatted_amt = f"{amount:,}".replace(",", ".") + "đ"
                if record_type == "Income":
                    resp = f"Tuyệt vời! Mimo đã ghi nhận khoản thu nhập {formatted_amt} vào danh mục {vietCat}."
                else:
                    resp = f"Mimo đã ghi nhận khoản chi {formatted_amt} cho {vietCat} vào ví của bạn."
            else:
                if record_type == "Income":
                    resp = f"Mimo đã chuẩn bị sẵn phiếu ghi nhận thu nhập cho danh mục {vietCat}. Bạn hãy nhập số tiền và bấm lưu nhé!"
                else:
                    resp = f"Mimo đã chuẩn bị sẵn phiếu ghi nhận chi tiêu cho danh mục {vietCat}. Bạn hãy nhập số tiền và bấm lưu nhé!"
            
            return {
                "intent": intent,
                "intent_confidence": 1.0,
                "intent_backend": f"user_{match['match_type']}",
                "text": user_text,
                "item": clean_content(user_text),
                "category": category,
                "amount": amount,
                "record_type": record_type,
                "is_expense": record_type == "Expense",
                "nlg_response": resp,
                "mimo_emotion": "Success" if record_type == "Income" else "Chill",
                "backend": f"user_{match['match_type']}",
            }

        # Priority 2: Action Heuristic Rules (First Pass Action queries bypass LLM)
        user_text_lower = user_text.lower().strip()
        is_suggest = any(kw in user_text_lower for kw in ("gợi ý", "đề xuất", "khuyên", "suggest", "recommend"))
        is_report = any(kw in user_text_lower for kw in ("tổng chi", "báo cáo", "thống kê", "chi tiêu tuần", "chi tiêu tháng", "chi tiêu hôm nay", "thu nhập tuần", "thu nhập tháng", "thu nhập hôm nay"))
        is_search = any(kw in user_text_lower for kw in ("tìm kiếm", "tra cứu", "tìm giao dịch", "tìm khoản"))
        
        if is_suggest or is_report or is_search:
            action_type = "SUGGEST_BUDGET" if is_suggest else ("REPORT_GENERAL" if is_report else "SEARCH_RECORD")
            # If it's a second pass (has action_facts in profile), let it fall through to Qwen
            if profile and "action_facts" in profile:
                pass
            else:
                # First pass: response is empty
                return {
                    "intent": "Action",
                    "intent_confidence": 1.0,
                    "intent_backend": "rule_heuristic",
                    "text": user_text,
                    "item": None,
                    "category": None,
                    "amount": None,
                    "record_type": None,
                    "is_expense": None,
                    "action_type": action_type,
                    "action_details": {
                        "time_range": parse_time_range(user_text, []) if is_report else None
                    },
                    "nlg_response": "",
                    "mimo_emotion": "Chill",
                    "backend": "llm_unified",
                }

        context_metadata = None
        if profile:
            try:
                context_metadata = build_unified_llm_context(profile)
            except Exception as e:
                print(f"[NLU pipeline] Error building unified context: {e}")

        # Fallback to Qwen LLM NLU if personalization not matched
        result = run_llm_nlu(user_text, context_metadata=context_metadata, run_llm=True)
        return result

    intent, intent_conf, intent_proba = classify_intent(user_text, intent_model)

    # Personalization Hybrid Layer: exact match or semantic similarity match
    match = None
    if user_corrections:
        vectorizer = category_model.get("vectorizer")
        match = find_matching_correction(user_text, user_corrections, vectorizer)

    if match:
        if match.get("intent"):
            intent = match["intent"]
            intent_conf = match.get("similarity_score", 1.0)

    # LLM Fallback: khi encoder confidence thấp → dùng LLM để re-classify
    llm_fallback_result = None
    if run_llm and not match:
        llm_fallback_result = run_llm_fallback(
            user_text,
            encoder_intent=intent,
            encoder_confidence=intent_conf,
        )
        if llm_fallback_result:
            intent = llm_fallback_result["intent"]
            intent_conf = llm_fallback_result.get("intent_confidence", intent_conf)

    amounts = extract_amounts(user_text)
    content = clean_content(user_text)

    # Extract slots early for all intents (useful for relationship tags and actions)
    slots = None
    if ner_model:
        slots = extract_ner_slots(user_text, ner_model)

    result = {
        "text": user_text,
        "intent": intent,
        "intent_backend": (
            "llm_fallback" if llm_fallback_result
            else f"user_{match['match_type']}" if match and match.get("intent")
            else intent_model.get("backend")
        ),
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

        # Full sentence vs NER span: nếu hai model khác nhau → ưu tiên câu đủ (có ngữ cảnh xã hội)
        if raw_category and mapped_category:
            if str(raw_category).lower() != str(mapped_category).lower():
                final_category = raw_category
            else:
                final_category = mapped_category
        else:
            final_category = mapped_category or raw_category

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
        result["action_type_backend"] = action_type_model.get("backend")
        result["action_param"] = amounts[0] if amounts else None

        slots_bundle = None
        if action_slots_model and action_slots_model.get("backend") == "slots":
            slots_bundle = action_slots_model.get("bundle")
        result["action_details"] = predict_action_details(
            user_text,
            result.get("action_type"),
            slots_bundle,
            ner_slots=slots,
        )
        result["action_details_backend"] = (
            "slots_model" if slots_bundle else "missing_slots_model"
        )
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


def infer_with_tfidf(user_text: str, bundle: dict) -> dict:
    import os
    old_env = os.environ.get("NLU_USE_ENCODER")
    os.environ["NLU_USE_ENCODER"] = "0"
    
    from src.nlu import models
    old_backend_fn = models._registry_inference_backend
    models._registry_inference_backend = lambda: "tfidf"
    
    try:
        return run_nlu(
            user_text,
            bundle["intent"],
            bundle["category"],
            bundle["action_type"],
            bundle["record_type"],
            bundle["sentiment"],
            bundle["ner"],
            bundle.get("action_slots"),
            run_llm=False
        )
    finally:
        if old_env is not None:
            os.environ["NLU_USE_ENCODER"] = old_env
        else:
            os.environ.pop("NLU_USE_ENCODER", None)
        models._registry_inference_backend = old_backend_fn


def infer_with_phobert(user_text: str, bundle: dict) -> dict:
    import os
    old_env = os.environ.get("NLU_USE_ENCODER")
    os.environ["NLU_USE_ENCODER"] = "1"
    
    from src.nlu import models
    old_backend_fn = models._registry_inference_backend
    models._registry_inference_backend = lambda: "encoder"
    
    try:
        return run_nlu(
            user_text,
            bundle["intent"],
            bundle["category"],
            bundle["action_type"],
            bundle["record_type"],
            bundle["sentiment"],
            bundle["ner"],
            bundle.get("action_slots"),
            run_llm=False
        )
    finally:
        if old_env is not None:
            os.environ["NLU_USE_ENCODER"] = old_env
        else:
            os.environ.pop("NLU_USE_ENCODER", None)
        models._registry_inference_backend = old_backend_fn


infer_with_encoder = infer_with_phobert
