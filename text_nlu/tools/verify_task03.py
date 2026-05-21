"""Verify TASK-03 Shopping category samples."""
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
    ("mua trên shopee 120k", "Shopping"),
    ("tiktok shop 89k", "Shopping"),
    ("shopee 200k", "Shopping"),
    ("mua sạc 150k", "Shopping"),
    ("mua giày sneaker 800k", "Shopping"),
    ("gạo 50k", "Essentials"),  # không nhầm Essentials
    ("mua quà cho mẹ 200k", "Essentials"),  # hard negative
]


def main() -> None:
    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    rec_m = load_record_type_model()
    sent_m = load_chitchat_sentiment_model()
    lines = []
    ok = 0
    for text, exp in TESTS:
        r = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, None)
        got = r.get("category")
        status = "OK" if got == exp else "FAIL"
        if status == "OK":
            ok += 1
        lines.append(f"{status} | {text} -> {got} (expected {exp})")
    lines.append(f"\n{ok}/{len(TESTS)} category match")
    out = Path(__file__).parent / "_task03_verify.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
