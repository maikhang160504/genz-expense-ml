"""Chạy nhanh NLU trên câu mẫu cố định (Record / Action / Chitchat) — TASK-15."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

# intent bắt buộc; các field khác tùy chọn (oracle thủ công)
SAMPLES: list[dict] = [
    {"text": "Ăn phở sáng 45k", "intent": "Record", "category": "Food", "record_type": "Expense"},
    {"text": "Lương tháng về 12tr", "intent": "Record", "record_type": "Income"},
    {"text": "me cho 1tr", "intent": "Record", "record_type": "Income", "category": "Bonus"},
    {"text": "mẹ ck 500k", "intent": "Record", "record_type": "Income"},
    {"text": "hoàn tiền 50k", "intent": "Record", "record_type": "Income"},
    {"text": "mua trên shopee 120k", "intent": "Record", "category": "Shopping"},
    {"text": "tiktok shop 89k", "intent": "Record", "category": "Shopping"},
    {"text": "gạo 50k", "intent": "Record", "category": "Essentials"},
    {"text": "mua quà cho mẹ 200k", "intent": "Record", "category": "Essentials", "record_type": "Expense"},
    {"text": "Mua Netflix tháng 109k", "intent": "Record", "category": "Entertainment"},
    {"text": "ăn vặt 30k", "intent": "Record"},
    {"text": "Tháng này tiêu bao nhiêu rồi", "intent": "Action", "action_type": "REPORT_GENERAL"},
    {"text": "tổng chi tháng này", "intent": "Action", "action_type": "REPORT_GENERAL"},
    {"text": "Xóa giao dịch vừa nhập", "intent": "Action", "action_type": "DELETE_RECORD"},
    {"text": "Đặt hạn mức ăn uống 2tr", "intent": "Action", "action_type": "SET_LIMIT"},
    {"text": "Tìm khoản chi trên 500k", "intent": "Action", "action_type": "SEARCH_RECORD"},
    {"text": "Chào bot nha", "intent": "Chitchat"},
    {"text": "okela di", "intent": "Chitchat"},
    {"text": "Bạn là ai vậy", "intent": "Chitchat"},
    {"text": "cho minh hoi app lam gi vay", "intent": "Chitchat"},
]


def _check_field(name: str, got, exp) -> str:
    if exp is None:
        return ""
    if got == exp:
        return f" {name}=OK"
    return f" {name}=FAIL(exp={exp},got={got})"


def main() -> None:
    load_env_file(settings.ENV_PATH)
    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    rec_m = load_record_type_model()
    sent_m = load_chitchat_sentiment_model()
    ner = load_ner_model(settings.NER_MODEL_DIR)

    intent_ok = 0
    extra_ok = 0
    extra_total = 0
    lines_out: list[str] = []

    for sample in SAMPLES:
        text = sample["text"]
        expect_intent = sample["intent"]
        r = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, ner)
        pred = r.get("intent")
        mark = "OK" if pred == expect_intent else "FAIL"
        if pred == expect_intent:
            intent_ok += 1
        extras = ""
        for key, field in (
            ("category", "category"),
            ("record_type", "record_type"),
            ("action_type", "action_type"),
            ("sentiment", "sentiment"),
        ):
            if key in sample:
                extra_total += 1
                chk = _check_field(field, r.get(field), sample[key])
                if "OK" in chk:
                    extra_ok += 1
                extras += chk
        line = f"{mark} intent | {text!r} got={pred}{extras}"
        lines_out.append(line)

    summary = (
        f"\n{intent_ok}/{len(SAMPLES)} intent match"
        f" | extra fields {extra_ok}/{extra_total}"
    )
    out_path = Path(__file__).resolve().parent / "_smoke_intent_result.txt"
    out_path.write_text("\n".join(lines_out) + summary + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
