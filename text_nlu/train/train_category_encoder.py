"""
PhoBERT embedding + logistic category — TASK-16 (thử thay TF-IDF).

Đầu ra: text_nlu/models/category_encoder.joblib

Lưu ý: pipeline ưu tiên file này nếu tồn tại (xem settings.CATEGORY_ENCODER_PATH).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT.parent
sys.path.insert(0, str(TRAIN_ROOT))

from src.nlu.encoder_runtime import embed_texts  # noqa: E402

DATA_DIR = ROOT / "datasets"
OUT_PATH = Path(
    os.environ.get(
        "CATEGORY_ENCODER_OUT",
        str(ROOT / "models" / "category_encoder.joblib"),
    )
)
MAX_SAMPLES = int(os.environ.get("CATEGORY_ENCODER_MAX_SAMPLES", "12000"))


def main() -> None:
    model_name = os.environ.get("ENCODER_MODEL_NAME", "vinai/phobert-base")
    df = pd.read_csv(DATA_DIR / "intent_record.csv", encoding="utf-8-sig")
    df = df[["text", "label"]].dropna()
    if MAX_SAMPLES > 0 and len(df) > MAX_SAMPLES:
        n_labels = df["label"].nunique()
        per = max(1, MAX_SAMPLES // n_labels)
        parts = [
            g.sample(n=min(len(g), per), random_state=42)
            for _, g in df.groupby("label", group_keys=False)
        ]
        df = pd.concat(parts, ignore_index=True)
        if len(df) > MAX_SAMPLES:
            df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    texts = df["text"].astype(str).tolist()
    y = np.asarray(df["label"].astype(str).tolist(), dtype=str)
    print(f"Training on {len(df)} rows, {len(set(y))} labels")
    X = embed_texts(texts, model_name, max_length=96, batch_size=8)
    strat = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    base = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    cal = CalibratedClassifierCV(base, cv=2, method="sigmoid")
    cal.fit(X_train, y_train)
    print(classification_report(y_test, cal.predict(X_test), zero_division=0))
    bundle = {"encoder_model_name": model_name, "classifier": cal, "task": "category"}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
