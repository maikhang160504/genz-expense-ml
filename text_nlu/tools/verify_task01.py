"""Verify TASK-01 record type samples."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.nlu.models import (
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.pipeline import run_nlu

TESTS = [
    ("me cho 1tr", "Income"),
    ("mẹ ck 500k", "Income"),
    ("hoàn tiền 50k", "Income"),
    ("ăn phở 45k", "Expense"),
    ("mua sua 1113k", "Expense"),
]


def main() -> None:
    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    rec_type_m = load_record_type_model()
    sent_m = load_chitchat_sentiment_model()
    lines = []
    for text, exp in TESTS:
        r = run_nlu(text, intent_m, cat_m, act_m, rec_type_m, sent_m, None)
        got = r.get("record_type")
        status = "OK" if got == exp else "FAIL"
        lines.append(f"{status} | {text} -> {got} (expected {exp})")
    out = Path(__file__).parent / "_task01_verify.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
