"""
Gán / chuẩn hóa cột slot trong intent_action.csv trước train action_slots.

Chạy thủ công khi thêm câu Action mới:
  python text_nlu/datasets/label_action_slots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TEXT_NLU = ROOT.parent
sys.path.insert(0, str(TEXT_NLU))

from load_slot_schema import import_slot_schema  # noqa: E402

_schema = import_slot_schema()
ALL_COLUMNS = _schema.ALL_COLUMNS
SLOT_COLUMNS = _schema.SLOT_COLUMNS

ACTION_CSV = ROOT / "intent_action.csv"


def _has_label(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip().lower()
    return s not in ("", "nan", "none")


def main() -> None:
    if not ACTION_CSV.is_file():
        raise FileNotFoundError(f"Missing {ACTION_CSV}")

    df = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[ALL_COLUMNS]
    df.to_csv(ACTION_CSV, index=False, encoding="utf-8-sig")

    labeled = int(df[SLOT_COLUMNS].apply(lambda r: any(_has_label(v) for v in r), axis=1).sum())
    print(f"Saved {ACTION_CSV}")
    print(f"  rows: {len(df)}, with any slot label: {labeled}")


if __name__ == "__main__":
    main()
