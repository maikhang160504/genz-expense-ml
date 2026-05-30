"""Quick smoke test for new prompts + context_meta fields + mascot_mood mapping."""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.nlg.context_meta import build_mock_context_metadata
from src.nlg.llm_runner import attach_nlg_and_llm, load_prompts, load_request_template
from src.nlu.models import (
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.ner import load_ner_model
from src.nlu.pipeline import run_nlu

STATUS_TO_ASSET = {
    "vui": "Happy",
    "buon": "Sad",
    "canh_bao": "Thinking",
    "trung_lap": "Chill",
}

TEST_CASES = [
    ("an pho 45k", "hai_huoc"),
    ("mua giay nike 500k", "cham_choc"),
    ("gui tien cho me 200k", "dong_cam"),
    ("dat han muc an uong 2 trieu", "nghiem_tuc"),
    ("hom nay stress qua khong muon ghi chi tieu", "dan_doi"),
    ("mua tra sua cho nguoi yeu 60k", "vui"),
]


def main() -> None:
    load_env_file(settings.ENV_PATH)
    request_template = load_request_template(settings.REQUEST_TEMPLATE_PATH)
    prompts_config = load_prompts(settings.PROMPTS_PATH)

    intent_model = load_intent_model()
    category_model = load_category_model()
    action_type_model = load_action_type_model()
    record_type_model = load_record_type_model()
    sentiment_model = load_chitchat_sentiment_model()
    ner_model = load_ner_model(settings.NER_MODEL_DIR)

    print("=== Demo new prompts + context_meta + mascot_mood ===\n")
    for text, emotion in TEST_CASES:
        result = run_nlu(
            text,
            intent_model,
            category_model,
            action_type_model,
            record_type_model,
            sentiment_model,
            ner_model,
        )
        nlu_result = {
            "intent": result.get("intent"),
            "text": text,
            "item": result.get("item"),
            "category": result.get("category"),
            "amount": result.get("amount_spent"),
            "action_type": result.get("action_type"),
            "relationship_tag": result.get("relationship_tag"),
        }
        ctx = build_mock_context_metadata(nlu_result)

        attach_nlg_and_llm(
            result,
            user_text=text,
            nlu_result=nlu_result,
            context_metadata=ctx,
            prompts_config=prompts_config,
            request_template=request_template,
            emotion=emotion,
        )

        g = result.get("gemini_json") or {}
        rtag = result.get("relationship_tag") or "-"
        print(f"[{emotion}] {text}")
        print(f"  intent={result.get('intent')}  rel_tag={rtag}  time={ctx.get('time_of_day')}  wallet={ctx.get('wallet_health')}  payday_in={ctx.get('days_to_payday')}d")
        print(f"  weather: {ctx.get('weather')}")
        print(f"  fact: {ctx.get('historical_fact')}")
        if g.get("story"):
            api_status = g.get("status", "?")
            mascot = STATUS_TO_ASSET.get(api_status, "?")
            print(f"  story: {g['story']}")
            print(f"  status={api_status} → mascot_mood={mascot}")
        else:
            err = result.get("gemini_error") or "LLM tắt hoặc không có API key"
            print(f"  [no LLM] {err}")
        print()


if __name__ == "__main__":
    main()
