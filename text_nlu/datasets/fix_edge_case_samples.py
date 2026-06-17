"""
Bổ sung / sửa mẫu cho 2 edge case:
- đi cf với bạn bè → Entertainment (record)
- Báo cáo ăn uống tháng này → Report (action)

Chạy: python text_nlu/datasets/fix_edge_case_samples.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RECORD_CSV = ROOT / "intent_record.csv"
ACTION_CSV = ROOT / "intent_action.csv"

# Đi cf / cafe kiểu social — Entertainment, không phải mua đồ ăn
ENTERTAINMENT_CF = [
    "đi cf với bạn bè hết 50k",
    "đi cf với bạn thân 50k",
    "đi cf với bọn bạn hết 50k",
    "rủ đi cf với bạn bè 50k",
    "hẹn đi cf với bạn bè 45k",
    "cf với bạn bè hết 50k",
    "đi cf cùng bạn bè 50k",
    "đi cf với nhỏ bạn 50k",
    "đi cf với bạn 50k",
    "đi cf với bạn bè 30k",
    "đi cf với bạn bè 80k",
    "tụ tập đi cf với bạn bè 50k",
    "đi cf hẹn bạn bè 50k",
    "cf với bạn bè tốn 50k",
    "đi cf với đám bạn 50k",
]

# Báo cáo theo danh mục → Report (không phải REPORT_GENERAL tổng)
REPORT_BY_CATEGORY = [
    "Báo cáo ăn uống tháng này",
    "Báo cáo ăn uống tuần này",
    "Báo cáo ăn uống hôm nay",
    "Báo cáo giải trí tháng này",
    "Báo cáo mua sắm tháng này",
    "Báo cáo đi lại tháng này",
    "Báo cáo y tế tháng này",
    "Báo cáo nhà cửa tháng này",
    "bao cao an uong thang nay",
    "bao cao giai tri thang nay",
]

# Sửa nhãn sai: "Báo cáo {category} ..." không có "tổng chi" → Report
FIX_ACTION_LABELS = {
    "Báo cáo giải trí tuần trước": "Report",
    "Báo cáo học phí tuần trước": "Report",
    "Báo cáo mua sắm tuần trước": "Report",
    "Báo cáo nhà cửa tuần trước": "Report",
    "Báo cáo y tế tuần trước": "Report",
    "Báo cáo ăn uống tuần trước": "Report",
    "Báo cáo đi lại tuần trước": "Report",
    "Báo cáo điện nước tuần trước": "Report",
}


def boost_record() -> int:
    df = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    existing = set(df["text"].astype(str).str.strip())
    new_rows = []
    for text in ENTERTAINMENT_CF:
        for _ in range(3):  # oversample mẫu cf social
            if text not in existing:
                existing.add(text)
            new_rows.append({
                "text": text,
                "label": "Entertainment",
                "type": "expense",
                "is_money": 1,
            })
    if not new_rows:
        return 0
    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    return len(new_rows)


def boost_action() -> tuple[int, int]:
    df = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    fixed = 0
    for text, label in FIX_ACTION_LABELS.items():
        mask = df["text"].astype(str).str.strip() == text
        if mask.any():
            old = df.loc[mask, "action_type"].iloc[0]
            if old != label:
                df.loc[mask, "action_type"] = label
                fixed += 1

    existing = set(df["text"].astype(str).str.strip())
    added = 0
    for text in REPORT_BY_CATEGORY:
        if text in existing:
            continue
        df = pd.concat([df, pd.DataFrame([{
            "text": text,
            "intent": "Action",
            "action_type": "Report",
        }])], ignore_index=True)
        existing.add(text)
        added += 1

    df.to_csv(ACTION_CSV, index=False, encoding="utf-8-sig")
    return fixed, added


def main() -> None:
    n_rec = boost_record()
    n_fix, n_add = boost_action()
    print(f"Record: +{n_rec} Entertainment cf-social rows")
    print(f"Action: fixed {n_fix} labels, +{n_add} Report-by-category rows")


if __name__ == "__main__":
    main()
