import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.nlg.context_meta import build_context_metadata, build_mock_context_metadata
from src.nlg.llm_runner import attach_nlg_and_llm, load_prompts, load_request_template
from src.nlu.models import (
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.action_executor import describe_action_execution
from src.nlu.ner import load_ner_model
from src.nlu.pipeline import run_nlu


def _pick_llm_text(gemini_json: dict | None, llama_json: dict | None) -> tuple[str | None, str | None]:
    story = None
    source = None
    if gemini_json and gemini_json.get("story"):
        story = str(gemini_json.get("story"))
        source = "Gemini"
    elif llama_json and llama_json.get("story"):
        story = str(llama_json.get("story"))
        source = "Groq"
    return story, source


def _nlu_result_from_pipeline(result: dict, user_text: str) -> dict:
    return {
        "intent": result.get("intent"),
        "text": user_text,
        "item": result.get("item"),
        "category": result.get("category"),
        "amount": result.get("amount_spent") if result.get("intent") == "Record" else result.get("action_param"),
        "is_expense": result.get("record_type") == "Expense" if result.get("intent") == "Record" else None,
        "income_type": result.get("income_type") if result.get("intent") == "Record" else None,
        "action_type": result.get("action_type"),
        "value": result.get("action_param"),
    }


def _print_demo_summary(
    user_text: str,
    result: dict,
    nlu_result: dict,
    context_metadata: dict,
    gemini_json: dict | None,
    llama_json: dict | None,
) -> None:
    intent = result.get("intent")
    lines = [
        "",
        "==========",
        f"Input: {user_text}",
        f"Intent: {intent}",
    ]
    if result.get("intent_confidence") is not None:
        lines.append(f"  → intent_confidence={result.get('intent_confidence'):.3f}")
    if intent == "Record":
        lines.append(
            f"  → category={result.get('category')}, amount={result.get('amount_spent')}, "
            f"record_type={result.get('record_type')}, income_type={result.get('income_type')}, "
            f"item={result.get('item')}"
        )
    elif intent == "Action":
        lines.append(f"  → action_type={result.get('action_type')}, action_param={result.get('action_param')}")
    else:
        lines.append("  → chitchat via LLM (no NLU sentiment)")

    lines.append(f"context_meta ({context_metadata.get('source', '?')}): type={context_metadata.get('type')}")

    story, src = _pick_llm_text(gemini_json, llama_json)
    if story:
        status = (gemini_json or llama_json or {}).get("status")
        lines.append(f"LLM ({src}) — story: {story}")
        if status is not None:
            lines.append(f"LLM ({src}) — status: {status}")
    elif intent == "Action" and result.get("action_ack"):
        ack = result["action_ack"]
        lines.append(f"Phản hồi (ack cục bộ): {ack.get('story')} [{ack.get('status')}]")
    else:
        lines.append("LLM: (tắt — đặt RUN_LLM=1 và cấu hình API để xem phản hồi)")

    for ex in result.get("demo_execution_lines") or []:
        lines.append(ex)

    lines.append("==========\n")
    print("\n".join(lines))


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

    use_mock_context = os.environ.get("USE_MOCK_CONTEXT", "1") == "1"
    print("Demo NLU + NLG (gọn)")
    print("- USE_MOCK_CONTEXT=1 (mặc định): ngữ cảnh prompt = mock ngẫu nhiên.")
    print("- USE_MOCK_CONTEXT=0: dùng profile cố định như trước.")
    print("- RUN_LLM=1: gọi Gemini/Groq nếu có API trong .env")
    print("- RUN_LLM_CHITCHAT=1 (mặc định): Chitchat luôn gọi LLM")
    print("Gõ exit để thoát.\n")

    while True:
        user_text = input("Input: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        result = run_nlu(
            user_text,
            intent_model,
            category_model,
            action_type_model,
            record_type_model,
            sentiment_model,
            ner_model,
        )
        if result.get("intent") == "Action":
            result["demo_execution_lines"] = describe_action_execution(result)
        else:
            result["demo_execution_lines"] = []

        nlu_result = _nlu_result_from_pipeline(result, user_text)

        demo_profile = {
            "budget_total": 1_000_000,
            "budget_remain": 150_000,
            "frequency_week": 5,
            "avg_amount": 35_000,
            "amount_threshold": 50_000,
            "old_value": 1_000_000,
        }
        if use_mock_context:
            context_metadata = build_mock_context_metadata(nlu_result)
        else:
            context_metadata = build_context_metadata(nlu_result, demo_profile)

        attach_nlg_and_llm(
            result,
            user_text=user_text,
            nlu_result=nlu_result,
            context_metadata=context_metadata,
            prompts_config=prompts_config,
            request_template=request_template,
            emotion="hai_huoc",
        )

        if result.get("intent") == "Action" and not result.get("gemini_json"):
            result["action_ack"] = {
                "story": "Đã ghi nhận yêu cầu và sẽ cập nhật thiết lập.",
                "status": "canh_bao" if context_metadata.get("is_triggered") else "trung_lap",
            }

        _print_demo_summary(
            user_text,
            result,
            nlu_result,
            context_metadata,
            result.get("gemini_json"),
            result.get("llama_json"),
        )


if __name__ == "__main__":
    main()
