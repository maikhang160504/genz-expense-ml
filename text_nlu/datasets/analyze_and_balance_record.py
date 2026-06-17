"""
Phân tích lệch intent_record.csv -> JSON + bổ sung nhãn yếu.

Output: text_nlu/datasets/record_balance_analysis.json

Chạy:
  python text_nlu/datasets/analyze_and_balance_record.py
  python text_nlu/datasets/analyze_and_balance_record.py --skip-supplement
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATASETS = Path(__file__).resolve().parent
RECORD_CSV = DATASETS / "intent_record.csv"
GENZ_CSV = DATASETS.parents[2] / "danh_sach_tu_khoa_genz_converted.csv"
ANALYSIS_JSON = DATASETS / "record_balance_analysis.json"

INCOME_LABELS = {"Salary", "Bonus", "Business", "Investment", "Savings", "Saving"}
EXPENSE_LABELS = {
    "Food", "Transport", "Housing", "Essentials", "Shopping", "Beauty",
    "Health", "Education", "Entertainment", "Social", "Debt", "Charity", "Others",
}

# Mẫu bổ sung local khi genz không đủ
LOCAL_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Others": [
        ("phí dịch vụ {m}", "expense"),
        ("misc fee {m}", "expense"),
        ("chi linh tinh {m}", "expense"),
        ("khoản lạ {m}", "expense"),
        ("phí nhỏ {m}", "expense"),
        ("khoản không rõ {m}", "expense"),
        ("chi phí khác {m}", "expense"),
        ("transaction fee {m}", "expense"),
        ("phí chuyển khoản {m}", "expense"),
        ("phí duy trì tài khoản {m}", "expense"),
        ("chi phí phát sinh {m}", "expense"),
        ("khoản chi khác {m}", "expense"),
        ("phí xử lý giao dịch {m}", "expense"),
        ("chi tiêu linh tinh {m}", "expense"),
        ("khoản không phân loại {m}", "expense"),
        ("phí quản lý {m}", "expense"),
        ("misc expense {m}", "expense"),
        ("other charge {m}", "expense"),
        ("phí hành chính {m}", "expense"),
        ("khoản chi lẻ {m}", "expense"),
        ("tip nhỏ {m}", "expense"),
        ("phí cộng thêm {m}", "expense"),
        ("chi phí ngoài dự kiến {m}", "expense"),
        ("khoản phí lạ {m}", "expense"),
    ],
    "Savings": [
        ("gửi tiết kiệm {m}", "income"),
        ("lãi tiết kiệm {m}", "income"),
        ("rút tiết kiệm {m}", "income"),
    ],
    "Charity": [
        ("quyên góp từ thiện {m}", "expense"),
        ("donate {m}", "expense"),
        ("ủng hộ đồng bào lũ lụt {m}", "expense"),
    ],
    "Debt": [
        ("trả nợ ngân hàng {m}", "expense"),
        ("trả góp {m}", "expense"),
        ("đóng nợ {m}", "expense"),
    ],
}

AMOUNTS = ["25k", "50k", "88k", "120k", "200k", "350k", "500k", "1tr", "1.5tr", "2tr", "2.5tr", "3tr"]
SUFFIXES = ["", " nha", " nhé", " rồi", " đó", " ha"]


def normalize_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].replace({"Saving": "Savings"})
    return out


def analyze(df: pd.DataFrame) -> dict:
    df = normalize_labels_df(df)
    df = df[df["type"].isin(["expense", "income"])].copy()
    label_vc = df["label"].value_counts()
    type_vc = df["type"].value_counts()
    median_label = float(label_vc.median()) if len(label_vc) else 0.0
    target_per_label = int(max(2500, median_label * 0.85))

    income_target = int(type_vc.get("expense", 0) * 0.18)
    current_income = int(type_vc.get("income", 0))

    weak: list[dict] = []
    for lab, cnt in label_vc.items():
        gap = max(0, target_per_label - int(cnt))
        if gap > 0:
            weak.append({
                "label": str(lab),
                "count": int(cnt),
                "target": target_per_label,
                "gap": gap,
                "type": "income" if str(lab) in INCOME_LABELS else "expense",
            })
    weak.sort(key=lambda x: x["gap"], reverse=True)

    expense_cnt = int(type_vc.get("expense", 0))
    income_gap = max(0, income_target - current_income)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(RECORD_CSV),
        "total_rows": len(df),
        "type_distribution": {str(k): int(v) for k, v in type_vc.items()},
        "label_distribution": {str(k): int(v) for k, v in label_vc.items()},
        "expense_label_distribution": {
            str(k): int(v) for k, v in df[df["type"] == "expense"]["label"].value_counts().items()
        },
        "income_label_distribution": {
            str(k): int(v) for k, v in df[df["type"] == "income"]["label"].value_counts().items()
        },
        "stats": {
            "median_label_count": median_label,
            "target_per_label": target_per_label,
            "expense_income_ratio": round(expense_cnt / max(current_income, 1), 2),
            "income_target": income_target,
            "income_gap": income_gap,
        },
        "weak_labels": weak[:20],
        "recommendations": [
            f"Bổ sung ~{income_gap} mẫu income (hiện {current_income}, mục tiêu ~{income_target})",
            f"Ưu tiên nhãn: {', '.join(w['label'] for w in weak[:5])}",
        ],
    }


def _gen_local(label: str, typ: str, n: int, existing: set[str]) -> list[dict]:
    tpls = LOCAL_TEMPLATES.get(label, [])
    if not tpls:
        return []
    random.seed(hash(label) % 99991)
    rows: list[dict] = []
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        tpl, t = random.choice(tpls)
        text = (tpl.format(m=random.choice(AMOUNTS)) + random.choice(SUFFIXES)).strip()
        if text in existing:
            continue
        existing.add(text)
        rows.append({"text": text, "label": label, "type": t or typ, "is_money": 1})
    return rows


def _gen_income_boost(n: int, existing: set[str]) -> list[dict]:
    from family_bonus_labels import FAMILY_INCOME_TEMPLATES
    from short_record_generator import money_variant

    salary_tpl = [
        ("lg ve {m}", "Salary"), ("luong thang ve {m}", "Salary"), ("nhan luong {m}", "Salary"),
    ]
    business_tpl = [
        ("fl nhan {m}", "Business"), ("hoan tien {m}", "Business"), ("tip {m}", "Business"),
    ]
    all_tpl = list(FAMILY_INCOME_TEMPLATES) + salary_tpl + business_tpl
    rows: list[dict] = []
    seed = 77
    while len(rows) < n:
        tpl, lab = all_tpl[seed % len(all_tpl)]
        text = tpl.format(m=money_variant(seed)).strip()
        seed += 1
        if text in existing:
            continue
        existing.add(text)
        rows.append({"text": text, "label": lab, "type": "income", "is_money": 1})
    return rows


def supplement_from_analysis(
    df: pd.DataFrame,
    analysis: dict,
    genz: pd.DataFrame | None,
) -> pd.DataFrame:
    from merge_genz_keywords import strip_hash_id

    existing = set(df["text"].astype(str).str.strip())
    blocked = set(existing)
    new_rows: list[dict] = []

    for item in analysis["weak_labels"]:
        label, gap = item["label"], min(item["gap"], 3200 if item["label"] == "Others" else 1200)
        if gap <= 0:
            continue
        typ = item["type"]
        added = 0

        if genz is not None and len(genz):
            pool = genz[genz["label"].isin([label, "Saving"] if label == "Savings" else [label])]
            pool = pool[~pool["text"].astype(str).str.strip().isin(blocked)]
            for _, r in pool.iterrows():
                if added >= gap:
                    break
                t = strip_hash_id(r["text"])
                if t not in blocked:
                    blocked.add(t)
                    new_rows.append({
                        "text": t,
                        "label": "Savings" if label == "Savings" else label,
                        "type": r["type"],
                        "is_money": int(r["is_money"]),
                    })
                    added += 1

        remain = gap - added
        if remain > 0:
            new_rows.extend(_gen_local(label, typ, remain, blocked))

    income_gap = analysis["stats"].get("income_gap", 0)
    if income_gap > 0:
        boost_n = min(income_gap, 1500)
        new_rows.extend(_gen_income_boost(boost_n, blocked))

    if not new_rows:
        print("Supplement: nothing to add")
        return df

    from improve_datasets import dedupe_record

    add_df = dedupe_record(pd.DataFrame(new_rows))
    add_df = add_df[~add_df["text"].astype(str).str.strip().isin(existing)]
    print(f"Supplement: +{len(add_df)} rows for weak labels")
    return pd.concat([df, add_df], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-supplement", action="store_true")
    args = parser.parse_args()

    df = normalize_labels_df(pd.read_csv(RECORD_CSV, encoding="utf-8-sig"))
    analysis = analyze(df)
    ANALYSIS_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analysis saved: {ANALYSIS_JSON}")
    print(f"  total={analysis['total_rows']} weak_labels={len(analysis['weak_labels'])}")
    print(f"  income_gap={analysis['stats']['income_gap']}")

    if args.skip_supplement:
        return 0

    genz = None
    if GENZ_CSV.is_file():
        from merge_genz_keywords import load_and_clean_genz, dedupe_genz as dedupe_g

        genz = dedupe_g(load_and_clean_genz())

    df = df[df["type"].isin(["expense", "income"])].copy()
    merged = supplement_from_analysis(df, analysis, genz)
    merged.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print(f"Record saved: {len(merged)} rows")

    # Refresh analysis after supplement
    analysis2 = analyze(merged)
    analysis2["supplement_applied"] = True
    ANALYSIS_JSON.write_text(json.dumps(analysis2, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
