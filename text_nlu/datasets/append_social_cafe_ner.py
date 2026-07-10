"""NER + record mẫu cafe/xã giao → Entertainment (thay keyword is_entertainment_cafe)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RECORD_CSV = ROOT / "intent_record.csv"
NER_JSONL = ROOT / "ner_dataset.jsonl"

NER_ROWS = [
    {
        "text": "Đi cà phê với bạn 19k",
        "label": [[3, 8, "CATEGORY"], [9, 16, "COMPANION"], [17, 20, "AMOUNT"]],
    },
    {
        "text": "đi cf với bạn bè hết 50k",
        "label": [[3, 5, "CATEGORY"], [6, 16, "COMPANION"], [21, 24, "AMOUNT"]],
    },
    {
        "text": "hẹn đi cafe sữa đá với bồ 45k",
        "label": [[7, 19, "CATEGORY"], [20, 27, "COMPANION"], [28, 31, "AMOUNT"]],
    },
    {
        "text": "đi cafe hẹn hò với người yêu 80k",
        "label": [[3, 7, "CATEGORY"], [8, 27, "COMPANION"], [28, 31, "AMOUNT"]],
    },
    {
        "text": "cf với bạn thân 35k",
        "label": [[0, 2, "CATEGORY"], [3, 15, "COMPANION"], [16, 19, "AMOUNT"]],
    },
]

RECORD_ROWS = [
    ("Đi cà phê với bạn 19k", "Entertainment"),
    ("đi cf với bạn bè hết 50k", "Entertainment"),
    ("hẹn đi cafe sữa đá với bồ 45k", "Entertainment"),
    ("đi cafe hẹn hò với người yêu 80k", "Entertainment"),
    ("cf với bạn thân 35k", "Entertainment"),
    ("đi uống cafe trà sữa với crush 55k", "Entertainment"),
    ("hẹn hò quán cf view đẹp 120k", "Entertainment"),
    ("Mua cà phê 19k", "Food"),
    ("Mua cafe sữa đá 25k", "Food"),
]


def main() -> None:
    existing_ner: dict[str, dict] = {}
    if NER_JSONL.is_file():
        with NER_JSONL.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_ner[row["text"].strip()] = row

    ner_added = 0
    for row in NER_ROWS:
        t = row["text"]
        if t not in existing_ner:
            ner_added += 1
        existing_ner[t] = row

    with NER_JSONL.open("w", encoding="utf-8") as f:
        for row in existing_ner.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    seen = set(df["text"].astype(str).str.strip())
    rec_added = 0
    for text, label in RECORD_ROWS:
        if text in seen:
            continue
        df = pd.concat(
            [df, pd.DataFrame([{"text": text, "label": label, "type": "expense", "is_money": 1}])],
            ignore_index=True,
        )
        seen.add(text)
        rec_added += 1
    df.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print(f"NER updated: {ner_added} new social-cafe rows")
    print(f"Record updated: {rec_added} new rows (total {len(df)})")


if __name__ == "__main__":
    main()
