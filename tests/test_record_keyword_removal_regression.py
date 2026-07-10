"""
Hồi quy Record sau gỡ keyword runtime (is_entertainment_cafe, is_clear_income, intent guards…).

Chạy: python tests/test_record_keyword_removal_regression.py
"""
from __future__ import annotations

import json
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

# (text, intent, category|None, record_type|None, tag)
RECORD_CASES: list[tuple] = [
    # --- Cafe Food vs Entertainment (is_entertainment_cafe) ---
    ("Mua cà phê 19k", "Record", "Food", "Expense", "cafe_buy_food"),
    ("Mua cafe sữa đá 25k", "Record", "Food", "Expense", "cafe_buy_food"),
    ("order cf mang về 32k", "Record", "Food", "Expense", "cafe_buy_food"),
    ("Đi cà phê với bạn 19k", "Record", "Entertainment", "Expense", "cafe_social"),
    ("đi cf với bạn bè hết 50k", "Record", "Entertainment", "Expense", "cafe_social"),
    ("hẹn đi cafe sữa đá với bồ 45k", "Record", "Entertainment", "Expense", "cafe_social"),
    ("cf date với crush 80k", "Record", "Entertainment", "Expense", "cafe_social"),
    ("đi uống trà sữa với nhóm bạn 120k", "Record", "Entertainment", "Expense", "cafe_social"),
    # --- Record vs Action ---
    ("Mới tiêu 2tr", "Record", None, "Expense", "record_not_action"),
    ("hôm nay tiêu hết 2tr", "Record", None, "Expense", "record_not_action"),
    ("chi grab 45k", "Record", "Transport", "Expense", "record_expense"),
    ("thanh toán 2tr tiền nhà", "Record", "Housing", "Expense", "record_housing"),
    ("Đặt giới hạn chi tiêu thành 2tr", "Action", None, None, "action_not_record"),
    ("gợi ý chi tiêu tuần này", "Action", None, None, "action_not_record"),
    # --- Income (is_clear_income_phrase / record_type) ---
    ("lương tháng 6 về 12tr", "Record", "Salary", "Income", "income_salary"),
    ("nhận lương 15 củ", "Record", "Salary", "Income", "income_salary"),
    ("thưởng dự án 2tr", "Record", "Bonus", "Income", "income_bonus"),
    ("freelance design 3tr5", "Record", "Business", "Income", "income_business"),
    ("mẹ chuyển khoản 500k", "Record", "Salary", "Income", "income_transfer"),
    ("hoàn tiền shopee 89k", "Record", "Others", "Income", "income_refund"),
    # --- Subscription / digital ---
    ("Netflix tháng 109k", "Record", "Entertainment", "Expense", "sub_entertainment"),
    ("gia hạn Spotify 59k", "Record", "Entertainment", "Expense", "sub_entertainment"),
    # --- Social vs Entertainment ---
    ("mua quà sinh nhật bạn 200k", "Record", "Social", "Expense", "social_gift"),
    ("đi nhậu team building 350k", "Record", "Entertainment", "Expense", "entertainment_drink"),
    # --- Transport ---
    ("Grab đi học 35k", "Record", "Transport", "Expense", "transport"),
    ("đổ xăng 100k", "Record", "Transport", "Expense", "transport"),
    # --- Education / Health ---
    ("photocopy tài liệu 25k", "Record", "Education", "Expense", "education"),
    ("khám bệnh viện 180k", "Record", "Health", "Expense", "health"),
    # --- NOT Record (Chitchat) ---
    ("Cảm ơn bot nhiều nha", "Chitchat", None, None, "chitchat_not_record"),
    ("Bạn khỏe không", "Chitchat", None, None, "chitchat_not_record"),
]

NEGATIVE_ACTION_CASES = [
    ("Đặt hạn mức ăn uống 2tr", "Action"),
    ("xóa giao dịch vừa nhập", "Action"),
    ("tổng chi tháng này", "Action"),
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
    failed: list[dict] = []
    tags_failed: set[str] = set()

    print("=== RECORD keyword-removal regression ===")
    for text, exp_intent, exp_cat, exp_rt, tag in RECORD_CASES:
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
        rt = nlu.get("record_type")
        errs: list[str] = []
        if intent != exp_intent:
            errs.append(f"intent={intent}!={exp_intent}")
        if exp_cat and cat != exp_cat:
            errs.append(f"category={cat}!={exp_cat}")
        if exp_rt and rt != exp_rt:
            errs.append(f"record_type={rt}!={exp_rt}")

        status = "OK" if not errs else "FAIL"
        print(f"[{status}] ({tag}) {text}")
        if errs:
            print(f"       {', '.join(errs)}")
            failed.append(
                {
                    "text": text,
                    "tag": tag,
                    "expected": {"intent": exp_intent, "category": exp_cat, "record_type": exp_rt},
                    "got": {"intent": intent, "category": cat, "record_type": rt},
                    "errors": errs,
                }
            )
            tags_failed.add(tag)

    print(f"\n=== SUMMARY: {len(failed)}/{len(RECORD_CASES)} failed ===")
    if tags_failed:
        print("Tags with failures:", sorted(tags_failed))

    out = ROOT / "tests" / "record_keyword_removal_failures.json"
    out.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
