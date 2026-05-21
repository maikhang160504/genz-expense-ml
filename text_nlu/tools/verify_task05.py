"""Verify TASK-05 REPORT_GENERAL action type (sau khi user train action encoder)."""
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
    ("Tháng này tiêu bao nhiêu rồi", "REPORT_GENERAL"),
    ("tổng chi tháng này", "REPORT_GENERAL"),
    ("thống kê chi tiêu tuần này", "REPORT_GENERAL"),
    ("xem tổng chi hôm nay giúp mình", "REPORT_GENERAL"),
    ("thang nay tieu bao nhieu roi", "REPORT_GENERAL"),
    # Không nhầm Report theo category / SEARCH
    ("Báo cáo ăn uống tháng này", "Report"),
    ("Tìm khoản chi trên 500k", "SEARCH_RECORD"),
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
        got = r.get("action_type")
        status = "OK" if got == exp else "FAIL"
        if status == "OK":
            ok += 1
        lines.append(f"{status} | {text!r} -> {got} (expected {exp})")
    lines.append(f"\n{ok}/{len(TESTS)} action_type match")
    out = Path(__file__).parent / "_task05_verify.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
