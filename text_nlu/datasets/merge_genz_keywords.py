"""
Gộp danh_sach_tu_khoa_genz_converted.csv vào intent_record.csv.

1) Loại bỏ mẫu #{số} thừa trong text (vd: "mua sách #1581 122k" -> "mua sách 122k")
2) Dedupe (exact + chỉ khác số tiền)
3) Append vào intent_record.csv

Chạy:
  python text_nlu/datasets/merge_genz_keywords.py
  python text_nlu/datasets/merge_genz_keywords.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
DATASETS = Path(__file__).resolve().parent
GENZ_CSV = PROJECT_ROOT / "danh_sach_tu_khoa_genz_converted.csv"
RECORD_CSV = DATASETS / "intent_record.csv"

VALID_LABELS = {
    "Food", "Transport", "Housing", "Essentials", "Shopping", "Beauty",
    "Health", "Education", "Entertainment", "Social", "Salary", "Bonus",
    "Business", "Investment", "Debt", "Charity", "Savings", "Saving", "Others",
}
VALID_TYPES = {"expense", "income"}

# #{digits} — noise id inserted during conversion
HASH_ID_RE = re.compile(r"\s#\d+(?=\s|$)", re.UNICODE)


def strip_hash_id(text: str) -> str:
    t = HASH_ID_RE.sub("", str(text))
    return re.sub(r"\s+", " ", t).strip()


def normalize_row(row: pd.Series) -> dict | None:
    text = strip_hash_id(row["text"])
    if len(text) < 2:
        return None
    label = str(row["label"]).strip()
    typ = str(row["type"]).strip().lower()
    if label not in VALID_LABELS or typ not in VALID_TYPES:
        return None
    if label == "Saving":
        label = "Savings"
    try:
        is_money = int(row["is_money"])
    except (TypeError, ValueError):
        is_money = 1
    if is_money not in (0, 1):
        is_money = 1
    return {"text": text, "label": label, "type": typ, "is_money": is_money}


def load_and_clean_genz() -> pd.DataFrame:
    if not GENZ_CSV.is_file():
        raise FileNotFoundError(f"Missing {GENZ_CSV}")
    raw = pd.read_csv(GENZ_CSV, encoding="utf-8-sig")
    rows: list[dict] = []
    stripped_count = 0
    for _, r in raw.iterrows():
        orig = str(r["text"])
        cleaned = normalize_row(r)
        if cleaned is None:
            continue
        if "#" in orig and cleaned["text"] != orig.strip():
            stripped_count += 1
        rows.append(cleaned)
    df = pd.DataFrame(rows)
    print(f"Genz: {len(raw)} raw -> {len(df)} valid ({stripped_count} had #id stripped)")
    return df


def dedupe_genz(df: pd.DataFrame) -> pd.DataFrame:
    from improve_datasets import dedupe_record

    before = len(df)
    out = dedupe_record(df)
    print(f"Genz dedupe: {before} -> {len(out)} (removed {before - len(out)})")
    return out


def append_to_record(genz: pd.DataFrame, dry_run: bool = False) -> int:
    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    rec = rec[rec["type"].isin(list(VALID_TYPES))].copy()
    existing = set(rec["text"].astype(str).str.strip())
    to_add = genz[~genz["text"].astype(str).str.strip().isin(existing)]
    print(f"Record before: {len(rec)} | new genz rows to append: {len(to_add)}")
    if dry_run or to_add.empty:
        return len(to_add)
    merged = pd.concat([rec, to_add], ignore_index=True)
    merged.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print(f"Record after: {len(merged)}")
    return len(to_add)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    genz = load_and_clean_genz()
    genz = dedupe_genz(genz)
    n = append_to_record(genz, dry_run=args.dry_run)

    # Sample cleaned texts
    samples = genz["text"].head(5).tolist()
    print("Samples after clean:", samples[:3])
    print(f"Done. {'Would append' if args.dry_run else 'Appended'} {n} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
