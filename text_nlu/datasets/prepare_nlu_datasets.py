"""
Chuẩn bị toàn bộ dataset NLU trước khi retrain.

Thứ tự:
  1) Rút gọn record_to_Need_Fix.csv (Gemini) nếu --clean-need-fix
  2) Gộp record_to_Need_Fix.csv vào intent_record.csv
  3) Dedupe intent_record (trùng text + trùng nghĩa chỉ khác số tiền)
  4) Bổ sung action SUGGEST_BUDGET + cân bằng action types
  5) Gemini augment (cân bằng income Salary/Bonus/Business + expense labels)
  6) improve_datasets.py (dedupe + augment biên)

Chạy:
  python text_nlu/datasets/prepare_nlu_datasets.py
  python text_nlu/datasets/prepare_nlu_datasets.py --skip-gemini
  python text_nlu/datasets/prepare_nlu_datasets.py --clean-need-fix --gemini-batches 20
"""
from __future__ import annotations

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import csv
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(__file__).resolve().parent
RECORD_CSV = DATASETS / "intent_record.csv"
NEED_FIX_CSV = DATASETS / "record_to_Need_Fix.csv"
ACTION_CSV = DATASETS / "intent_action.csv"


def _run_py(script: str, *args: str) -> None:
    path = DATASETS / script
    cmd = [sys.executable, str(path), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def merge_need_fix() -> int:
    if not NEED_FIX_CSV.is_file():
        print("Skip merge: record_to_Need_Fix.csv not found")
        return 0
    need = pd.read_csv(NEED_FIX_CSV, encoding="utf-8-sig", header=None, names=["text", "label", "type", "is_money"])
    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    existing = set(rec["text"].astype(str).str.strip())
    rows = []
    for _, r in need.iterrows():
        t = str(r["text"]).strip()
        if not t or t in existing:
            continue
        existing.add(t)
        rows.append({"text": t, "label": r["label"], "type": r["type"], "is_money": int(r["is_money"])})
    if not rows:
        print("Need_Fix: nothing new to merge")
        return 0
    rec = pd.concat([rec, pd.DataFrame(rows)], ignore_index=True)
    rec.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print(f"Merged {len(rows)} rows from record_to_Need_Fix.csv into intent_record ({len(rec)} total)")
    return len(rows)


def dedupe_record_amount_only() -> int:
    from improve_datasets import dedupe_record

    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    before = len(rec)
    rec = dedupe_record(rec)
    rec.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    removed = before - len(rec)
    print(f"Dedupe amount-only: {before} → {len(rec)} (removed {removed})")
    return removed


def add_suggest_budget_actions() -> int:
    """Đổi / thêm mẫu gợi ý chi tiêu → action_type SUGGEST_BUDGET."""
    act = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    suggest_phrases = [
        "gợi ý chi tiêu",
        "gợi ý chi tiêu cho mình",
        "gợi ý chi tiêu tuần này",
        "gợi ý chi tiêu tháng này",
        "cho mình gợi ý chi tiêu",
        "đề xuất gợi ý chi tiêu",
        "gợi ý tiết kiệm",
        "gợi ý tiết kiệm tháng này",
        "cho xin goi y chi tieu",
        "goi y chi tieu tuan nay",
        "goi y chi tieu thang nay",
        "goi y tiet kiem",
        "de xuat goi y chi tieu",
        "mimo goi y chi tieu giup minh",
        "bot goi y chi tieu tuan nay",
        "xin goi y han muc chi tieu",
        "goi y chi tieu hom nay",
        "gợi ý ngân sách tháng này",
        "gợi ý hạn mức chi tiêu",
        "đề xuất ngân sách tuần này",
    ]
    suffixes = ["", " nha", " nhé", " đi", " bot", " mimo", " ?", " ha"]
    changed = 0
    added = 0
    ex = set(act["text"].astype(str).str.strip())

    # Relabel existing REPORT_GENERAL → SUGGEST_BUDGET when matching suggest phrases
    for i, row in act.iterrows():
        t = str(row["text"]).strip().lower()
        if row["action_type"] == "REPORT_GENERAL" and any(p in t for p in ["goi y", "gợi ý", "de xuat", "đề xuất"]):
            if any(k in t for k in ["chi tieu", "chi tiêu", "tiet kiem", "tiết kiệm", "ngan sach", "ngân sách", "han muc", "hạn mức"]):
                act.at[i, "action_type"] = "SUGGEST_BUDGET"
                changed += 1

    new_rows = []
    for core in suggest_phrases:
        for suf in suffixes:
            text = (core + suf).strip()
            if text in ex:
                continue
            ex.add(text)
            new_rows.append({"text": text, "intent": "Action", "action_type": "SUGGEST_BUDGET"})
            added += 1

    if new_rows:
        act = pd.concat([act, pd.DataFrame(new_rows)], ignore_index=True)
    act.to_csv(ACTION_CSV, index=False, encoding="utf-8-sig")
    print(f"SUGGEST_BUDGET: relabeled {changed}, added {added} (total action {len(act)})")
    return changed + added


def local_balance_income(target_income: int = 3500) -> int:
    """Bổ sung income local (Salary/Bonus/Business) khi Gemini không khả dụng."""
    import random
    from family_bonus_labels import FAMILY_INCOME_TEMPLATES
    from short_record_generator import money_variant

    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    cur_income = int((rec["type"] == "income").sum())
    need = max(0, target_income - cur_income)
    if need == 0:
        print("Local income balance: already sufficient")
        return 0

    existing = set(rec["text"].astype(str).str.strip())
    salary_tpl = [
        ("lg ve {m}", "Salary"), ("luong thang ve {m}", "Salary"), ("nhan luong {m}", "Salary"),
        ("lương về {m}", "Salary"), ("ck luong {m}", "Salary"), ("lg thang {m}", "Salary"),
    ]
    business_tpl = [
        ("ban do cu {m}", "Business"), ("thu tien ban {m}", "Business"), ("fl nhan {m}", "Business"),
        ("hoan tien {m}", "Business"), ("refund {m}", "Business"), ("tip {m}", "Business"),
        ("nhan tien dich vu {m}", "Business"), ("ck ve {m}", "Business"),
    ]
    all_tpl = FAMILY_INCOME_TEMPLATES + salary_tpl + business_tpl
    rows = []
    n = 42
    random.seed(n)
    while len(rows) < need:
        tpl, lab = random.choice(all_tpl)
        m = money_variant(n)
        text = tpl.format(m=m).strip()
        n += 1
        if not text or text in existing:
            continue
        existing.add(text)
        rows.append({"text": text, "label": lab, "type": "income", "is_money": 1})

    rec = pd.concat([rec, pd.DataFrame(rows)], ignore_index=True)
    rec.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print(f"Local income balance: +{len(rows)} (total income {(rec['type']=='income').sum()})")
    return len(rows)


def gemini_balance_income(*, batches: int, rows: int, sleep: float) -> None:
    import gemini_augment_record as gar

    gar.load_env()
    gar.GENERATION_THEMES = gar.GENERATION_THEMES + gar.BALANCE_INCOME_THEMES
    df = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    gar.run_generate(
        df,
        batches=batches,
        rows_per_batch=rows,
        sleep_s=sleep,
        themes_override=gar.BALANCE_INCOME_THEMES,
        ignore_done=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-need-fix", action="store_true", help="Chạy clean_need_fix.py (Gemini)")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--gemini-batches", type=int, default=15)
    parser.add_argument("--gemini-rows", type=int, default=80)
    parser.add_argument("--gemini-sleep", type=float, default=2.0)
    args = parser.parse_args()

    if args.clean_need_fix:
        _run_py("clean_need_fix.py")

    merge_need_fix()
    dedupe_record_amount_only()
    add_suggest_budget_actions()
    _run_py("boost_action_intent_rows.py")

    if not args.skip_gemini:
        try:
            gemini_balance_income(batches=args.gemini_batches, rows=args.gemini_rows, sleep=args.gemini_sleep)
        except Exception as exc:
            print(f"Gemini augment skipped: {exc}", file=sys.stderr)
            local_balance_income()
    else:
        local_balance_income()

    dedupe_record_amount_only()
    _run_py("improve_datasets.py")

    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    act = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    print("\n=== Final stats ===")
    print(f"intent_record: {len(rec)} rows")
    print(f"  type: {rec['type'].value_counts().to_dict()}")
    print(f"  income labels: {rec[rec['type']=='income']['label'].value_counts().head(5).to_dict()}")
    print(f"intent_action: {len(act)} rows")
    print(f"  SUGGEST_BUDGET: {(act['action_type']=='SUGGEST_BUDGET').sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
