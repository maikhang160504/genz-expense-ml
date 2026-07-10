"""
Kiểm tra hồi quy sau khi gỡ keyword guard runtime.

- Action: phải pass qua model (intent + action_type + slot).
- Record category (cafe/social): thuộc intent_record — báo riêng, không ghi vào action CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.nlu.models import (
    load_action_type_model,
    load_action_slots_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.ner import load_ner_model
from src.nlu.pipeline import run_nlu

# (text, intent, action_type|None, category|None, verb|None, note)
ACTION_CASES: list[tuple] = [
    # Record vs Action (từng dùng is_action_query / money guard)
    ("Mới tiêu 2tr", "Record", None, None, None, "record_not_action"),
    ("hôm nay tiêu hết 2tr", "Record", None, None, None, "record_not_action"),
    ("Đặt giới hạn chi tiêu thành 2tr", "Action", "SET_LIMIT", None, "SET", "limit_set"),
    ("đặt hạn mức chi tiêu 5tr", "Action", "SET_LIMIT", None, "SET", "limit_set"),
    ("Thêm 200k vào ăn uống", "Action", "SET_LIMIT", None, "ADD", "limit_add"),
    ("bớt 50k từ hạn mức giải trí", "Action", "SET_LIMIT", None, "SUB", "limit_sub"),
    # SET_ALERT vs SYSTEM_SETTING (từng relabel SYSTEM_SETTING→SET_ALERT)
    ("bật cảnh báo vượt hạn mức cho ăn uống", "Action", "SET_ALERT", None, None, "alert_not_setting"),
    ("tắt thông báo chi tiêu", "Action", "SET_ALERT", None, None, "alert_off"),
    ("Bật nhắc nhở chi tiêu cho mục giải trí đi bro", "Action", "SET_ALERT", None, None, "alert_genz"),
    ("bật thông báo khi chi quá hạn mức mua sắm", "Action", "SET_ALERT", None, None, "alert_on"),
    # Gen-Z SET_LIMIT (hay nhầm SYSTEM_SETTING / Chitchat)
    ("Đừng cho tui xài quá 500k cho đồ ăn mỗi tháng nha Mimo", "Action", "SET_LIMIT", "Food", None, "limit_genz"),
    ("đừng cho tui chi quá 1tr tiền ăn mỗi tháng", "Action", "SET_LIMIT", "Food", None, "limit_genz"),
    ("hạn chế chi tiêu giải trí còn 300k thôi", "Action", "SET_LIMIT", "Entertainment", None, "limit_genz"),
    # SET_GOAL / ADD_GOAL
    ("tạo mục tiêu mua laptop mới 15 triệu", "Action", "SET_GOAL", None, None, "goal_set"),
    ("Tui muốn tích cóp 10 củ để sắm con lap mới á Mimo", "Action", "SET_GOAL", None, None, "goal_genz"),
    ("Tui mún tích cóp 5 củ để mua con laptop mới", "Action", "ADD_GOAL", None, "ADD", "goal_add_genz"),
    ("bù 2tr vào mục tiêu mua nhà", "Action", "ADD_GOAL", None, "ADD", "goal_add"),
    # SUGGEST vs REPORT (từng dùng suggest_budget_action_type / report_general)
    ("gợi ý chi tiêu tuần này", "Action", "SUGGEST_BUDGET", None, None, "suggest"),
    ("goi y chi tieu thang nay", "Action", "SUGGEST_BUDGET", None, None, "suggest"),
    ("tổng chi tháng này", "Action", "REPORT_GENERAL", None, None, "report"),
    ("thống kê chi tiêu ăn uống tháng này", "Action", "REPORT_GENERAL", None, None, "report"),
    # SYSTEM_SETTING thật
    ("chuyển sang giao diện tối", "Action", "SYSTEM_SETTING", None, None, "setting"),
    ("Chuyển giao diện sang tông tối nha Mimo", "Action", "SYSTEM_SETTING", None, None, "setting"),
    # DELETE / UPDATE / SEARCH
    ("xóa giao dịch vừa nhập", "Action", "DELETE_RECORD", None, None, "delete"),
    ("sửa số tiền giao dịch vừa rồi thành 50k", "Action", "UPDATE_RECORD", "Food", None, "update"),
    ("tìm các giao dịch mua sắm", "Action", "SEARCH_RECORD", "Shopping", None, "search"),
    ("SET_TONE", "Action", "SET_TONE", None, None, "tone"),
]

# Sửa typo SET_TONE case
ACTION_CASES[-1] = ("đổi sang giọng nói châm chọc nhé", "Action", "SET_TONE", None, None, "tone")

RECORD_CATEGORY_CASES: list[tuple] = [
    ("Mua cà phê 19k", "Record", "Food"),
    ("Mua cafe sữa đá 25k", "Record", "Food"),
    ("Đi cà phê với bạn 19k", "Record", "Entertainment"),
    ("đi cf với bạn bè hết 50k", "Record", "Entertainment"),
    ("hẹn đi cafe sữa đá với bồ 45k", "Record", "Entertainment"),
]


def _load():
    load_env_file(settings.ENV_PATH)
    return {
        "intent": load_intent_model(),
        "cat": load_category_model(),
        "act": load_action_type_model(),
        "slots": load_action_slots_model(),
        "rec": load_record_type_model(),
        "sent": load_chitchat_sentiment_model(),
        "ner": load_ner_model(settings.NER_MODEL_DIR),
    }


def main() -> int:
    models = _load()
    action_failed: list[dict] = []
    record_failed: list[dict] = []

    print("=== ACTION regression (no keyword guards) ===")
    for text, exp_intent, exp_at, exp_cat, exp_verb, tag in ACTION_CASES:
        nlu = run_nlu(
            text,
            models["intent"],
            models["cat"],
            models["act"],
            models["rec"],
            models["sent"],
            models["ner"],
            models["slots"],
        )
        intent = nlu.get("intent")
        at = nlu.get("action_type")
        cat = nlu.get("category")
        verb = (nlu.get("action_details") or {}).get("verb")
        target = (nlu.get("action_details") or {}).get("target")

        errs: list[str] = []
        if intent != exp_intent:
            errs.append(f"intent={intent}!={exp_intent}")
        if exp_at and at not in (exp_at, "Setting" if exp_at == "SYSTEM_SETTING" else exp_at):
            errs.append(f"action_type={at}!={exp_at}")
        if exp_cat and cat != exp_cat and target != exp_cat:
            # slot target có thể thay category trên Action
            if not (target and exp_cat.lower() in str(target).lower()):
                errs.append(f"category/target={cat}/{target}!={exp_cat}")
        if exp_verb and verb != exp_verb:
            errs.append(f"verb={verb}!={exp_verb}")

        status = "OK" if not errs else "FAIL"
        print(f"[{status}] ({tag}) {text}")
        if errs:
            print(f"       {', '.join(errs)}")
            action_failed.append(
                {
                    "text": text,
                    "tag": tag,
                    "expected": {
                        "intent": exp_intent,
                        "action_type": exp_at,
                        "category": exp_cat,
                        "verb": exp_verb,
                    },
                    "got": {
                        "intent": intent,
                        "action_type": at,
                        "category": cat,
                        "target": target,
                        "verb": verb,
                    },
                    "errors": errs,
                }
            )

    print("\n=== RECORD category (intent_record — không phải action CSV) ===")
    for text, exp_intent, exp_cat in RECORD_CATEGORY_CASES:
        nlu = run_nlu(
            text,
            models["intent"],
            models["cat"],
            models["act"],
            models["rec"],
            models["sent"],
            models["ner"],
            models["slots"],
        )
        intent = nlu.get("intent")
        cat = nlu.get("category")
        errs = []
        if intent != exp_intent:
            errs.append(f"intent={intent}!={exp_intent}")
        if cat != exp_cat:
            errs.append(f"category={cat}!={exp_cat}")
        status = "OK" if not errs else "FAIL"
        print(f"[{status}] {text} -> {cat}")
        if errs:
            record_failed.append({"text": text, "expected_category": exp_cat, "got": cat, "errors": errs})

    print(f"\n=== SUMMARY: action {len(action_failed)}/{len(ACTION_CASES)} failed, "
          f"record category {len(record_failed)}/{len(RECORD_CATEGORY_CASES)} failed ===")

    out = ROOT / "tests" / "keyword_removal_failures.json"
    import json

    out.write_text(
        json.dumps({"action": action_failed, "record_category": record_failed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Failures written: {out}")

    return 1 if action_failed else 0


if __name__ == "__main__":
    sys.exit(main())
