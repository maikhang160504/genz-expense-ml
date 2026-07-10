"""Dự đoán action_details từ model slot (không rule runtime)."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.nlu.text import extract_amounts


def load_action_slots_model(path) -> dict | None:
    import joblib

    if not path.is_file():
        return None
    return joblib.load(path)


def _predict_field(bundle: dict, field: str, text: str) -> Any:
    spec = bundle.get("fields", {}).get(field)
    if not spec:
        return None
    exact = spec.get("exact") or {}
    if text in exact:
        return exact[text]
    if spec.get("type") == "exact":
        return None
    model = spec.get("model")
    if spec.get("type") == "classifier" and model is not None:
        pred = model.predict([text])[0]
        return str(pred).strip() if pred is not None else None
    if spec.get("type") == "regressor" and model is not None:
        log_v = float(model.predict([text])[0])
        return int(round(np.expm1(log_v)))
    return None


def predict_action_details(
    text: str,
    action_type: str | None,
    slots_bundle: dict | None,
    ner_slots: dict | None = None,
) -> dict:
    """
    Chỉ dùng:
    - slot classifiers/regressors (train từ intent_action.csv)
    - NER AMOUNT span → value (model NER + parse số)
    """
    act = str(action_type or "").upper()
    if act == "SETTING":
        act = "SYSTEM_SETTING"

    details: dict[str, Any] = {
        "verb": None,
        "target": None,
        "target_type": None,
        "goal_name": None,
        "tool_type": None,
        "loan_type": None,
        "contact_name": None,
        "due_date": None,
        "value": None,
        "unit": None,
        "time": None,
        "enabled": None,
        "theme": None,
        "verbal_style": None,
        "query": None,
        "note": None,
    }

    if not slots_bundle:
        return {k: v for k, v in details.items() if v is not None}

    fields = slots_bundle.get("slots_by_action", {}).get(act, [])
    for field in fields:
        if field == "value":
            continue
        val = _predict_field(slots_bundle, field, text)
        if val is None:
            continue
        if field == "goal_name":
            details["goal_name"] = val
            details["target"] = val
        elif field == "category_code":
            details["target"] = val
        elif field == "time_range":
            details["time"] = [val]
        elif field == "enabled":
            details["enabled"] = val
        else:
            details[field] = val

    if act in {"ADD_GOAL", "SET_LIMIT"} and not details.get("verb"):
        details["verb"] = _predict_field(slots_bundle, "verb", text)

    # value: NER AMOUNT (model) trước, fallback slot regressor / text classifier
    if "value" in fields or act in {"SET_LIMIT", "SET_GOAL", "ADD_GOAL", "SET_INCOME", "SET_USERNAME"}:
        if ner_slots and ner_slots.get("AMOUNT"):
            amounts = extract_amounts(ner_slots["AMOUNT"][0])
            if amounts:
                details["value"] = amounts[0]
        if details["value"] is None:
            exact = (slots_bundle.get("fields", {}).get("value") or {}).get("exact") or {}
            if text in exact:
                details["value"] = exact[text]
        if details["value"] is None:
            details["value"] = _predict_field(slots_bundle, "value", text)
        if details["value"] is None and act == "SET_USERNAME":
            details["value"] = _predict_field(slots_bundle, "value_text", text)

    if details["value"] is not None:
        details["unit"] = "VND"

    if act in {"SET_LIMIT"}:
        details["target_type"] = "LIMIT"
    elif act in {"SET_GOAL", "ADD_GOAL"}:
        details["target_type"] = "GOAL"

    return {k: v for k, v in details.items() if v is not None}
