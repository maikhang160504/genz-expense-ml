"""
Thay khối intent_record.csv dòng 6237–12507 (1-based, kèm header):
dữ liệu lặp kiểu «mua bánh ngọt Grab hết 12k» / «lg tháng 7508 ngàn».

Chạy: python text_nlu/datasets/fix_record_monotone_block.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from short_record_generator import generate_rows

CSV_PATH = Path(__file__).resolve().parent / "intent_record.csv"
# Dòng file 6237 = index 6235 (sau header); dòng 12507 = index 12505 → cắt [6235:12506]
START_IDX = 6235
END_IDX = 12506


def main() -> int:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Missing header", file=sys.stderr)
            return 1
        all_rows = list(reader)

    block_len = END_IDX - START_IDX
    if block_len <= 0 or START_IDX >= len(all_rows):
        print(f"Invalid range: file has {len(all_rows)} rows", file=sys.stderr)
        return 1

    existing = {r["text"].strip() for r in all_rows if r.get("text")}
    for i in range(START_IDX, min(END_IDX, len(all_rows))):
        existing.discard(all_rows[i]["text"].strip())

    replacement = generate_rows(block_len, existing, seed=START_IDX, allow_bare=True)
    if len(replacement) < block_len:
        print(f"Warning: only generated {len(replacement)}/{block_len}", file=sys.stderr)

    new_block = [
        {"text": t, "label": lab, "type": typ, "is_money": im}
        for t, lab, typ, im in replacement
    ]
    while len(new_block) < block_len:
        extra = generate_rows(block_len - len(new_block), existing, seed=START_IDX + len(new_block))
        for t, lab, typ, im in extra:
            new_block.append({"text": t, "label": lab, "type": typ, "is_money": im})

    head = all_rows[:START_IDX]
    tail = all_rows[END_IDX:]
    merged = head + new_block[:block_len] + tail

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(merged)

    print(f"Replaced rows {START_IDX + 1}–{END_IDX} ({block_len} lines) with diverse short/medium samples.")
    print(f"Total rows: {len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
