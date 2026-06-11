import sys
from pathlib import Path
import json

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.nlg import llm_runner, context_meta

def test_case(text, record_type, category, amount, income_type, persona):
    print(f"\n==================================================")
    print(f"RUNNING TEST CASE: text='{text}', persona='{persona}'")
    print(f"==================================================")
    sys.stdout.flush()

    load_env_file(settings.ENV_PATH)
    prompts_config = llm_runner.load_prompts(settings.PROMPTS_PATH)
    request_template = llm_runner.load_request_template(settings.REQUEST_TEMPLATE_PATH)

    nlu_for_meta = {
        "intent": "Record",
        "text": text,
        "item": text.replace("mua ", "").replace("hết ", "").replace(" 45k", "").replace("về 15tr", ""),
        "category": category,
        "amount": amount,
        "record_type": record_type,
        "is_expense": record_type == "Expense",
        "income_type": income_type,
        "action_type": None,
        "value": None,
    }

    result = {
        "text": text,
        "intent": "Record",
        "item": nlu_for_meta["item"],
        "category": category,
        "amount_spent": amount if record_type == "Expense" else None,
        "record_type": record_type,
        "is_expense": nlu_for_meta["is_expense"],
        "income_type": income_type,
    }

    # Mock context metadata
    profile = {
        "budget_total": 5000000,
        "budget_remain": 1200000,
        "wallet_health": "can_than"
    }
    context_metadata = context_meta.build_context_metadata(nlu_for_meta, profile)

    llm_runner.attach_nlg_and_llm(
        result,
        user_text=text,
        nlu_result=nlu_for_meta,
        context_metadata=context_metadata,
        prompts_config=prompts_config,
        request_template=request_template,
        nlg_persona=persona,
        run_llm=True,
    )

    print("\n1. FULL PROMPT SENT TO GEMINI:")
    print("--------------------------------------------------")
    nlg_prompt = result.get("nlg_prompt")
    if nlg_prompt:
        print(">>> SYSTEM PROMPT:")
        print(nlg_prompt.get("system"))
        print("\n>>> USER PROMPT:")
        print(nlg_prompt.get("user"))
    else:
        print("No prompt was built!")

    print("\n2. RESPONSE AND PARSED JSON FROM GEMINI:")
    print("--------------------------------------------------")
    print(">>> Raw Gemini API Response:")
    print(result.get("gemini_response"))
    print("\n>>> Parsed Gemini JSON:")
    print(json.dumps(result.get("gemini_json"), indent=2, ensure_ascii=False))

    print("\n>>> Final mimo_emotion / mascot_mood:")
    print(result.get("mimo_emotion"))
    print("==================================================")
    sys.stdout.flush()

def main():
    # Test case 1: Expense with dan_doi persona (slangs and emotion check)
    test_case(
        text="mua trà sữa hết 45k",
        record_type="Expense",
        category="Food",
        amount=45000,
        income_type=None,
        persona="dan_doi"
    )

    # Test case 2: Income with vui persona
    test_case(
        text="lương về 15tr",
        record_type="Income",
        category="Others",
        amount=15000000,
        income_type="salary",
        persona="vui"
    )

if __name__ == "__main__":
    main()
