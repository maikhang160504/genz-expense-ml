"""Audit action_type vs slot extraction for action NLU."""
import json
import sys
from pathlib import Path

import pandas as pd

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


def load_models():
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


def run(text, models):
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
    d = nlu.get("action_details") or {}
    slots = nlu.get("slots") or {}
    return {
        "text": text,
        "intent": nlu.get("intent"),
        "action_type": nlu.get("action_type"),
        "verb": d.get("verb"),
        "target": d.get("target"),
        "value": d.get("value"),
        "time_range": nlu.get("time_range"),
        "slots": {k: v for k, v in slots.items() if v},
    }


def main():
    models = load_models()

    with open(ROOT.parent / "example.json", encoding="utf-8") as f:
        examples = json.load(f)["text"]

    print("=== EXAMPLE.JSON ===")
    for t in examples:
        r = run(t, models)
        ok_at = r["action_type"] in ("SET_LIMIT", "ADD_GOAL", "SET_GOAL")
        ok_verb = r["verb"] == "ADD"
        ok_val = r["value"] is not None
        ok_tgt = r["target"] is not None
        status = "OK" if (ok_at and ok_verb and ok_val and ok_tgt) else "FAIL"
        print(f"[{status}] {t}")
        print(f"  type={r['action_type']} verb={r['verb']} target={r['target']} value={r['value']}")
        print(f"  slots={r['slots']}")

    df = pd.read_csv(ROOT / "text_nlu/datasets/intent_action.csv")
    print("\n=== SAMPLES PER ACTION TYPE (3 each) ===")
    for at in sorted(df["action_type"].unique()):
        subset = df[df["action_type"] == at]
        samples = subset.sample(min(3, len(subset)), random_state=42)["text"].tolist()
        print(f"\n--- {at} ---")
        for t in samples:
            r = run(t, models)
            match = r["action_type"] == at or (at == "SYSTEM_SETTING" and r["action_type"] == "Setting")
            print(f"  pred={r['action_type']} match={match}")
            print(f"    verb={r['verb']} target={r['target']} value={r['value']} slots={r['slots']}")
            print(f"    text: {t[:90]}")

    extra = [
        ("SET_LIMIT", "ADD", "Transport", 1_000_000, "thêm 1tr vào giới hạn di chuyển"),
        ("SET_LIMIT", "SUB", "Food", 200_000, "giảm hạn mức ăn uống 200k"),
        ("SET_LIMIT", "SET", "Food", 2_000_000, "đặt hạn mức ăn uống 2 triệu"),
        ("SET_ALERT", None, "Food", None, "bật cảnh báo vượt hạn mức cho ăn uống"),
        ("SET_ALERT", None, None, None, "tắt thông báo chi tiêu"),
        ("SYSTEM_SETTING", None, None, None, "chuyển sang giao diện tối"),
        ("ADD_GOAL", "ADD", "mua nhà", 2_000_000, "bù 2tr vào mục tiêu mua nhà"),
        ("SET_GOAL", "SET", "laptop", 15_000_000, "tạo mục tiêu mua laptop mới 15 triệu"),
        ("SET_TONE", None, None, None, "đổi sang giọng nói châm chọc nhé"),
        ("SET_USERNAME", None, None, None, "gọi mình là Khang nhé"),
        ("SET_INCOME", None, None, 10_000_000, "thu nhập hàng tháng của mình là 10 triệu"),
        ("SEARCH_RECORD", None, "Shopping", None, "tìm các giao dịch mua sắm"),
        ("DELETE_RECORD", None, None, None, "xóa giao dịch vừa nhập"),
        ("REPORT_GENERAL", None, None, None, "thống kê chi tiêu ăn uống tháng này"),
        ("SUGGEST_BUDGET", None, None, None, "gợi ý ngân sách chi tiêu cho tháng sau"),
        ("UPDATE_RECORD", None, "Food", 50_000, "sửa số tiền giao dịch vừa rồi thành 50k"),
    ]
    print("\n=== MANUAL SLOT TESTS ===")
    failed = 0
    for exp_at, exp_verb, exp_tgt, exp_val, t in extra:
        r = run(t, models)
        at_ok = r["action_type"] in (exp_at, "Setting" if exp_at == "SYSTEM_SETTING" else exp_at)
        verb_ok = exp_verb is None or r["verb"] == exp_verb
        val_ok = exp_val is None or r["value"] == exp_val
        tgt_ok = exp_tgt is None or (r["target"] and exp_tgt.lower() in str(r["target"]).lower())
        ok = at_ok and verb_ok and val_ok and tgt_ok
        if not ok:
            failed += 1
        tag = "OK" if ok else "FAIL"
        print(f"[{tag}] {t}")
        print(f"  exp: type={exp_at} verb={exp_verb} target~={exp_tgt} val={exp_val}")
        print(f"  got: type={r['action_type']} verb={r['verb']} target={r['target']} val={r['value']} slots={r['slots']}")

    print(f"\n=== SUMMARY: {failed} manual slot tests failed ===")


if __name__ == "__main__":
    main()
