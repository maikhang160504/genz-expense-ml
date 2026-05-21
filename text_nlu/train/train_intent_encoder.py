"""
Huấn luyện intent bằng PhoBERT (mặc định) hoặc XLM-R + Logistic + hiệu chỉnh sigmoid (CalibratedClassifierCV).

Đầu ra: text_nlu/models/intent_encoder.joblib
  { encoder_model_name, classifier, task: "intent" }

Biến môi trường:
  ENCODER_MODEL_NAME  (mặc định vinai/phobert-base)
  INTENT_ENCODER_OUT  (đường dẫn joblib đầu ra)
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
OUT_PATH = Path(os.environ.get("INTENT_ENCODER_OUT", str(ROOT / "models" / "intent_encoder.joblib")))


def load_intent_rows() -> pd.DataFrame:
    rec = pd.read_csv(DATA_DIR / "intent_record.csv", encoding="utf-8-sig")[["text"]].copy()
    rec["intent"] = "Record"
    act = pd.read_csv(DATA_DIR / "intent_action.csv", encoding="utf-8-sig")[["text", "intent"]]
    chat = pd.read_csv(DATA_DIR / "intent_chitchat.csv", encoding="utf-8-sig")[["text", "intent"]]
    df = pd.concat([rec, act, chat], ignore_index=True)
    df.dropna(subset=["text", "intent"], inplace=True)
    return df


def main() -> None:
    model_name = os.environ.get("ENCODER_MODEL_NAME", "vinai/phobert-base")
    df = load_intent_rows()
    n_record = (df["intent"] == "Record").sum()
    n_chat = (df["intent"] == "Chitchat").sum()
    n_action = (df["intent"] == "Action").sum()

    # Cân Action — Record ~10x nhiều hơn → model hay gán Record cho «tổng chi…»
    action_target = max(n_action, int(n_record * 0.18), 8000)
    action_df = df[df["intent"] == "Action"]
    if len(action_df) < action_target:
        extra = action_df.sample(n=action_target - len(action_df), replace=True, random_state=42)
        df = pd.concat([df, extra], ignore_index=True)

    # Chitchat
    chat_target = min(len(df[df["intent"] == "Action"]), max(n_chat * 2, 2000))
    chat_df = df[df["intent"] == "Chitchat"]
    if len(chat_df) < chat_target:
        extra = chat_df.sample(n=chat_target - len(chat_df), replace=True, random_state=42)
        df = pd.concat([df, extra], ignore_index=True)

    print(f"Train rows: Record={n_record} Action={n_action}->{action_target} Chitchat={n_chat} total={len(df)}")

    max_samples = int(os.environ.get("INTENT_ENCODER_MAX_SAMPLES", "0"))
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    texts = df["text"].astype(str).tolist()
    # ndarray str — tránh ArrowExtensionArray (pandas 2+) gây lỗi với train_test_split/stratify
    y = np.asarray(df["intent"].astype(str).tolist(), dtype=str)
    X_emb = embed_texts(texts, model_name, max_length=128, batch_size=8)

    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y, test_size=0.2, random_state=42, stratify=y
    )
    base = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
    )
    cal = CalibratedClassifierCV(base, cv=2, method="sigmoid")
    cal.fit(X_train, y_train)
    pred = cal.predict(X_test)
    print(classification_report(y_test, pred))

    bundle = {
        "encoder_model_name": model_name,
        "classifier": cal,
        "task": "intent",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
