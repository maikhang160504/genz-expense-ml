"""Thêm mẫu Record category regression (cafe/social → Entertainment) — không thuộc action CSV."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RECORD_CSV = ROOT / "intent_record.csv"

ROWS = [
    ("Đi cà phê với bạn 19k", "Entertainment", "expense", 1),
    ("đi cf với bạn bè hết 50k", "Entertainment", "expense", 1),
    ("hẹn đi cafe sữa đá với bồ 45k", "Entertainment", "expense", 1),
    ("đi cafe hẹn hò với người yêu 80k", "Entertainment", "expense", 1),
    ("cf với bạn thân 35k", "Entertainment", "expense", 1),
    ("đi uống cafe trà sữa với crush 55k", "Entertainment", "expense", 1),
    ("hẹn hò quán cf view đẹp 120k", "Entertainment", "expense", 1),
    ("Mua cà phê 19k", "Food", "expense", 1),
    ("Mua cafe sữa đá 25k", "Food", "expense", 1),
]


def main() -> None:
    df = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    existing = set(df["text"].astype(str).str.strip())
    added = 0
    for text, label, typ, is_money in ROWS:
        if text in existing:
            continue
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{"text": text, "label": label, "type": typ, "is_money": is_money}]
                ),
            ],
            ignore_index=True,
        )
        existing.add(text)
        added += 1
    df.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print(f"Added {added} record regression rows → {RECORD_CSV} (total {len(df)})")


if __name__ == "__main__":
    main()
