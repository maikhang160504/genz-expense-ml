"""
Sửa label Business → Bonus cho income từ người thân (mẹ cho, gia đình…).

Chạy: python text_nlu/datasets/fix_family_bonus_labels.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from family_bonus_labels import is_family_gift_income

CSV_PATH = Path(__file__).resolve().parent / "intent_record.csv"


def main() -> int:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Missing header", file=sys.stderr)
            return 1
        rows = list(reader)

    changed = 0
    for r in rows:
        if str(r.get("type", "")).lower() != "income":
            continue
        if str(r.get("is_money", "1")).strip() not in ("1", "1.0", "True", "true"):
            continue
        if str(r.get("label", "")) != "Business":
            continue
        text = str(r.get("text", "")).strip()
        if is_family_gift_income(text):
            r["label"] = "Bonus"
            changed += 1

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Updated {changed} rows: Business -> Bonus (family gift income)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
