"""
Bổ sung Action (REPORT_GENERAL) + Record income rõ — phục vụ smoke / intent encoder.

Chạy: python text_nlu/datasets/boost_action_intent_rows.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ACTION_CSV = ROOT / "intent_action.csv"
RECORD_CSV = ROOT / "intent_record.csv"

SUGGEST_CORE = [
    "gợi ý chi tiêu tuần này",
    "goi y chi tieu thang nay",
    "gợi ý tiết kiệm",
    "xin goi y chi tieu",
    "de xuat goi y chi tieu",
    "gợi ý ngân sách tháng này",
]

REPORT_CORE = [
    "tổng chi tháng này",
    "tong chi thang nay",
    "tổng chi tuần này",
    "tổng chi hôm nay",
    "tổng chi tháng trước",
    "tiêu bao nhiêu tháng này",
    "tháng này tiêu bao nhiêu",
    "tháng này tiêu bao nhiêu rồi",
    "hôm nay đã tiêu mấy",
    "minh da tieu bao nhieu thang nay",
    "thống kê chi tiêu tháng này",
    "xem tổng chi tháng này",
    "bao cao tong chi thang nay",
    "chi hết bao nhiêu tháng này",
    "đã tiêu bao nhiêu tháng này",
]

SUFFIXES = ["", " bot", " nha", " ha", " hả", " vậy", " đi", " ?", " a"]

INCOME_CORE = [
    ("Lương tháng về 12tr", "Salary"),
    ("lương tháng về 12tr", "Salary"),
    ("luong thang ve 12tr", "Salary"),
    ("lg tháng về 14tr", "Salary"),
    ("lg ve 10tr", "Salary"),
    ("nhận lương 15tr", "Salary"),
    ("nhan luong 8tr", "Salary"),
]


def main() -> int:
    act = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    ex_a = set(act["text"].astype(str).str.strip())
    ex_r = set(rec["text"].astype(str).str.strip())

    add_a = []
    for core in REPORT_CORE:
        for suf in SUFFIXES:
            t = (core + suf).strip()
            if t not in ex_a:
                ex_a.add(t)
                add_a.append({"text": t, "intent": "Action", "action_type": "REPORT_GENERAL"})

    for core in SUGGEST_CORE:
        for suf in SUFFIXES:
            t = (core + suf).strip()
            if t not in ex_a:
                ex_a.add(t)
                add_a.append({"text": t, "intent": "Action", "action_type": "SUGGEST_BUDGET"})

    add_r = []
    for text, lab in INCOME_CORE:
        if text not in ex_r:
            ex_r.add(text)
            add_r.append({"text": text, "label": lab, "type": "income", "is_money": 1})

    if add_a:
        act = pd.concat([act, pd.DataFrame(add_a)], ignore_index=True)
        act.to_csv(ACTION_CSV, index=False, encoding="utf-8-sig")
    if add_r:
        rec = pd.concat([rec, pd.DataFrame(add_r)], ignore_index=True)
        rec.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")

    print(f"Action +{len(add_a)} (total {len(act)})")
    print(f"Record income +{len(add_r)} (total {len(rec)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
