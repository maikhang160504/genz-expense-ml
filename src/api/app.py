import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.nlg.context_meta import build_context_metadata
from src.nlg.llm_runner import (
    attach_nlg_and_llm,
    load_prompts,
    load_request_template,
)
from src.nlg.response import intent_mimo_fallback, parse_llm_response
from src.nlu.models import (
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.action_executor import describe_action_execution
from src.nlu.ner import load_ner_model
from src.nlu.json_sanitize import json_sanitize
from src.nlu.pipeline import run_nlu

app = FastAPI()

load_env_file(settings.ENV_PATH)
INTENT_MODEL = load_intent_model()
CATEGORY_MODEL = load_category_model()
ACTION_TYPE_MODEL = load_action_type_model()
RECORD_TYPE_MODEL = load_record_type_model()
SENTIMENT_MODEL = load_chitchat_sentiment_model()
NER_MODEL = load_ner_model(settings.NER_MODEL_DIR)
PROMPTS_CONFIG = load_prompts(settings.PROMPTS_PATH)
REQUEST_TEMPLATE = load_request_template(settings.REQUEST_TEMPLATE_PATH)


@app.on_event("startup")
def startup_event():
    print("Warming up NLU models with a dummy inference...", flush=True)
    try:
        dummy_result = run_nlu(
            "Tôi đã chi 50k ăn trưa",
            INTENT_MODEL,
            CATEGORY_MODEL,
            ACTION_TYPE_MODEL,
            RECORD_TYPE_MODEL,
            SENTIMENT_MODEL,
            NER_MODEL,
        )
        print(f"Model warm-up successful. Intent: {dummy_result.get('intent')}", flush=True)
    except Exception as e:
        print(f"Error during model warm-up: {e}", flush=True)


def _nlu_result_from_pipeline(result: dict, user_text: str) -> dict:
    return {
        "intent": result.get("intent"),
        "text": user_text,
        "item": result.get("item"),
        "category": result.get("category"),
        "amount": result.get("amount_spent") if result.get("intent") == "Record" else result.get("action_param"),
        "record_type": result.get("record_type") if result.get("intent") == "Record" else None,
        "is_expense": result.get("record_type") == "Expense" if result.get("intent") == "Record" else None,
        "income_type": result.get("income_type") if result.get("intent") == "Record" else None,
        "action_type": result.get("action_type"),
        "value": result.get("action_param"),
        "relationship_tag": result.get("relationship_tag"),
    }


@app.post("/infer")
def infer(payload: dict):
    user_text = payload.get("text", "")
    if not user_text:
        return {"error": "missing text"}

    result = run_nlu(
        user_text,
        INTENT_MODEL,
        CATEGORY_MODEL,
        ACTION_TYPE_MODEL,
        RECORD_TYPE_MODEL,
        SENTIMENT_MODEL,
        NER_MODEL,
    )
    if result.get("intent") == "Action":
        result["demo_execution_lines"] = describe_action_execution(result)
    else:
        result["demo_execution_lines"] = []

    nlu_result = _nlu_result_from_pipeline(result, user_text)
    context_metadata = build_context_metadata(nlu_result, payload.get("profile", {}))
    emotion = payload.get("emotion") or "hai_huoc"

    run_llm_flag = payload.get("run_llm")
    if run_llm_flag is True or str(run_llm_flag).lower() == "true":
        os.environ["RUN_LLM"] = "1"
    if result.get("intent") == "Chitchat":
        os.environ.setdefault("RUN_LLM_CHITCHAT", "1")

    chat_history = payload.get("chat_history")
    chat_summary = payload.get("chat_summary")

    attach_nlg_and_llm(
        result,
        user_text=user_text,
        nlu_result=nlu_result,
        context_metadata=context_metadata,
        prompts_config=payload.get("prompts") or PROMPTS_CONFIG,
        request_template=REQUEST_TEMPLATE,
        emotion=emotion,
        chat_history=chat_history,
        chat_summary=chat_summary,
    )

    if result.get("intent") == "Action" and not result.get("gemini_json"):
        ack_emotion = (
            "Worried"
            if context_metadata.get("is_triggered")
            else intent_mimo_fallback("Action")
        )
        result["action_ack"] = {
            "response": "Đã ghi nhận yêu cầu và sẽ cập nhật thiết lập.",
            "mimo_emotion": ack_emotion,
        }

    # Legacy: client vẫn có thể gửi gemini_response đã parse sẵn
    if payload.get("gemini_response") and not result.get("gemini_json"):
        result["gemini_json"] = parse_llm_response(
            payload.get("gemini_response", {}), "gemini"
        )

    return json_sanitize(result)
