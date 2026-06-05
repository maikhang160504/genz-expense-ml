import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.nlu.models import (
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.ner import load_ner_model
from src.nlu.pipeline import run_nlu


def main() -> None:
    load_env_file(settings.ENV_PATH)
    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    rec_m = load_record_type_model()
    sent_m = load_chitchat_sentiment_model()
    ner = load_ner_model(settings.NER_MODEL_DIR)

    test_cases = [
        # Problem 1: Food vs Entertainment Cafe
        ("Mua cà phê 19k", "Record", "Food", None),
        ("Mua cafe sữa đá 25k", "Record", "Food", None),
        ("Đi cà phê với bạn 19k", "Record", "Entertainment", None),
        ("đi cf với bạn bè hết 50k", "Record", "Entertainment", None),
        ("hẹn đi cafe sữa đá với bồ 45k", "Record", "Entertainment", None),

        # Problem 2: Action vs Record
        ("Đặt giới hạn chi tiêu thành 2tr", "Action", None, "SET_LIMIT"),
        ("đặt hạn mức chi tiêu 5tr", "Action", None, "SET_LIMIT"),
        ("Mới tiêu 2tr", "Record", None, None),
        ("hôm nay tiêu hết 2tr", "Record", None, None),
        ("thanh toán 2tr tiền nhà", "Record", "Housing", None),

        # Problem 3: Action Math Operators
        ("Thêm 200k vào ăn uống", "Action", None, "SET_LIMIT", "ADD"),
        ("cộng thêm 100k vào giới hạn đi lại", "Action", None, "SET_LIMIT", "ADD"),
        ("bớt 50k từ hạn mức giải trí", "Action", None, "SET_LIMIT", "SUB"),
        ("giảm giới hạn ăn uống đi 100k", "Action", None, "SET_LIMIT", "SUB"),
        ("Đặt lại giới hạn thành 500k", "Action", None, "SET_LIMIT", "SET"),
    ]

    failed = 0
    print("=== Running NLU Fixes Verification Tests ===")
    for test_idx, case in enumerate(test_cases):
        text = case[0]
        expected_intent = case[1]
        expected_category = case[2]
        expected_action_type = case[3]
        expected_verb = case[4] if len(case) > 4 else None

        nlu = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, ner)
        intent = nlu.get("intent")
        category = nlu.get("category")
        action_type = nlu.get("action_type")
        details = nlu.get("action_details") or {}
        verb = details.get("verb")

        print(f"\nTest #{test_idx + 1}: '{text}'")
        print(f"  Predicted: intent={intent}, category={category}, action_type={action_type}, verb={verb}")
        
        errors = []
        if expected_intent and intent != expected_intent:
            errors.append(f"Expected intent={expected_intent}, got {intent}")
        if expected_category and category != expected_category:
            errors.append(f"Expected category={expected_category}, got {category}")
        if expected_action_type and action_type != expected_action_type:
            errors.append(f"Expected action_type={expected_action_type}, got {action_type}")
        if expected_verb and verb != expected_verb:
            errors.append(f"Expected verb={expected_verb}, got {verb}")

        if errors:
            print(f"  ❌ FAILED: {', '.join(errors)}")
            failed += 1
        else:
            print("  ✅ PASSED")

    print("\n=== Verification Summary ===")
    if failed == 0:
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"💥 {failed} TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
