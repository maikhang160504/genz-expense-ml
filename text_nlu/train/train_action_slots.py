"""
Huấn luyện slot classifier đa đầu ra từ intent_action.csv (TF-IDF + LogisticRegression).

Mỗi cột slot (verb, category_code, value, ...) có model riêng, chỉ train trên hàng có nhãn.
Ghi metrics → text_nlu/models/action_slots_metrics.json (retrain_all gộp vào retrain_all_metrics.json).

Chạy:
  python text_nlu/datasets/label_action_slots.py   # thủ công sau khi gộp CSV
  python text_nlu/train/train_action_slots.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))

from load_slot_schema import import_slot_schema  # noqa: E402

_schema = import_slot_schema()
SLOT_COLUMNS = _schema.SLOT_COLUMNS
SLOTS_BY_ACTION = _schema.SLOTS_BY_ACTION

DATA_PATH = ROOT / "datasets" / "intent_action.csv"
OUT_PATH = Path(
    os.environ.get(
        "ACTION_SLOTS_MODEL_OUT",
        str(ROOT / "models" / "action_slots_model.joblib"),
    )
)
METRICS_PATH = ROOT / "models" / "action_slots_metrics.json"

MIN_SAMPLES = int(os.environ.get("ACTION_SLOTS_MIN_SAMPLES", "2"))


def _has_label(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "nan"


def _eval_classifier(y_test: list[str], y_pred: list[str]) -> dict:
    if not y_test:
        return {}
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "weighted_f1": round(
            float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4
        ),
        "test_samples": len(y_test),
    }


def _train_classifier(texts: list[str], y: list[str]) -> tuple[dict | None, dict]:
    """Return (model payload, metrics dict)."""
    metrics: dict = {"type": "classifier", "train_samples": len(texts), "classes": sorted(set(y))}
    if len(texts) < MIN_SAMPLES:
        return None, metrics

    payload: dict = {"exact": {}}
    if len(set(y)) < 2:
        payload["exact"] = dict(zip(texts, y))
        payload["type"] = "exact"
        metrics["type"] = "exact"
        metrics["exact_rules"] = len(payload["exact"])
        return payload, metrics

    strat = y if pd.Series(y).value_counts().min() >= 2 else None
    if len(texts) >= 10:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, y, test_size=0.2, random_state=42, stratify=strat
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, y, test_size=0.2, random_state=42
            )
    else:
        X_train, y_train, X_test, y_test = texts, y, [], []

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000),
            ),
            (
                "clf",
                LogisticRegression(max_iter=3000, class_weight="balanced"),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    if len(X_test) > 0:
        y_pred = pipe.predict(X_test).tolist()
        print(classification_report(y_test, y_pred, zero_division=0))
        metrics.update(_eval_classifier(list(y_test), y_pred))

    # Exact map for inference shortcut; skip huge dicts (Kaggle RAM / joblib size).
    exact_map = dict(zip(texts, y)) if len(texts) <= 2000 else {}
    payload = {"type": "classifier", "model": pipe, "exact": exact_map}
    return payload, metrics


def _train_value_regressor(texts: list[str], y: list[float]) -> Pipeline | None:
    if len(texts) < MIN_SAMPLES:
        return None
    y_log = np.log1p(np.asarray(y, dtype=float))
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(texts, y_log)
    return pipe


def _parse_numeric(val) -> float | None:
    if not _has_label(val):
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except ValueError:
        return None


def _summary_from_fields(fields: dict[str, dict]) -> dict:
    f1s = [
        m["weighted_f1"]
        for m in fields.values()
        if m.get("weighted_f1") is not None
    ]
    accs = [
        m["accuracy"]
        for m in fields.values()
        if m.get("accuracy") is not None
    ]
    summary = {
        "trained_fields": len(fields),
        "classifier_fields": sum(1 for m in fields.values() if m.get("type") in {"classifier", "exact"}),
        "regressor_fields": sum(1 for m in fields.values() if m.get("type") == "regressor"),
    }
    if f1s:
        summary["avg_weighted_f1"] = round(sum(f1s) / len(f1s), 4)
    if accs:
        summary["avg_accuracy"] = round(sum(accs) / len(accs), 4)
    return summary


def _load_dataframe() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Copy intent_action.csv into text_nlu/datasets/ "
            "(Kaggle: re-version mimo-nlu-dataset)."
        )
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    missing = [c for c in SLOT_COLUMNS if c not in df.columns]
    if missing:
        print(f"[WARN] intent_action.csv missing slot columns: {missing}")
        for col in missing:
            df[col] = np.nan
    return df


def _count_labeled_cells(df: pd.DataFrame) -> int:
    cols = [c for c in SLOT_COLUMNS if c in df.columns]
    if not cols:
        return 0
    return int(sum(df[c].apply(_has_label).sum() for c in cols))


def main() -> None:
    df = _load_dataframe()
    labeled_cells = _count_labeled_cells(df)
    if labeled_cells == 0:
        raise RuntimeError(
            "intent_action.csv has no slot labels. Run manually before Kaggle train:\n"
            "  python text_nlu/datasets/label_action_slots.py\n"
            "Then re-version mimo-nlu-dataset with the updated CSV."
        )
    print(f"Loaded {len(df)} rows, {labeled_cells} labeled slot cells")

    bundle: dict = {"fields": {}, "slots_by_action": SLOTS_BY_ACTION}
    field_metrics: dict[str, dict] = {}

    for field in SLOT_COLUMNS:
        if field == "value":
            rows = df[df["value"].apply(_has_label)].copy()
            if rows.empty:
                continue
            numeric_rows = rows[rows["value"].apply(lambda v: _parse_numeric(v) is not None)]
            text_rows = rows[rows["value"].apply(lambda v: _parse_numeric(v) is None)]
            if not numeric_rows.empty:
                texts = numeric_rows["text"].astype(str).tolist()
                vals = [_parse_numeric(v) for v in numeric_rows["value"]]
                exact_val = {t: int(v) for t, v in zip(texts, vals) if v is not None}
                reg = _train_value_regressor(texts, vals)
                if reg:
                    bundle["fields"]["value"] = {
                        "type": "regressor",
                        "model": reg,
                        "exact": exact_val,
                    }
                    field_metrics["value"] = {
                        "type": "regressor",
                        "train_samples": len(texts),
                    }
                    print(f"value regressor: {len(texts)} samples")
            if not text_rows.empty:
                texts = text_rows["text"].astype(str).tolist()
                labels = text_rows["value"].astype(str).str.strip().tolist()
                clf, fm = _train_classifier(texts, labels)
                if clf:
                    bundle["fields"]["value_text"] = clf
                    field_metrics["value_text"] = fm
                    print(f"value_text classifier: {len(texts)} samples")
            continue

        rows = df[df[field].apply(_has_label)].copy()
        if rows.empty:
            continue
        texts = rows["text"].astype(str).tolist()
        labels = rows[field].astype(str).str.strip().tolist()
        clf, fm = _train_classifier(texts, labels)
        if clf:
            bundle["fields"][field] = clf
            field_metrics[field] = fm
            print(f"{field} classifier: {len(texts)} samples, classes={sorted(set(labels))}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved {OUT_PATH} ({len(bundle['fields'])} field models)")

    labeled_rows = int(sum(df[c].apply(_has_label).sum() for c in SLOT_COLUMNS if c in df.columns))
    metrics_doc = {
        "fields": field_metrics,
        "summary": {
            **_summary_from_fields(field_metrics),
            "dataset_rows": len(df),
            "labeled_slot_cells": labeled_rows,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved slot metrics → {METRICS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"train_action_slots FAILED: {exc}", file=sys.stderr)
        raise
