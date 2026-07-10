"""
Huấn luyện intent bằng PhoBERT + Logistic + CalibratedClassifierCV.

Đầu ra: text_nlu/models/intent_encoder.joblib
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT.parent
sys.path.insert(0, str(TRAIN_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder_data import load_intent_rows, xy_from_df  # noqa: E402
from encoder_embed import embed_texts  # noqa: E402
from encoder_metrics import compute_sklearn_metrics, save_task_metrics  # noqa: E402

OUT_PATH = Path(os.environ.get("INTENT_ENCODER_OUT", str(ROOT / "models" / "intent_encoder.joblib")))


def main() -> None:
    model_name = os.environ.get("ENCODER_MODEL_NAME", "vinai/phobert-base")
    df = load_intent_rows()
    print(f"Intent train rows: {len(df)}  classes={sorted(df['intent'].unique())}")

    texts, y = xy_from_df(df, "intent")
    X_emb = embed_texts(texts, model_name, max_length=128, batch_size=8)

    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y, test_size=0.2, random_state=42, stratify=y
    )
    base = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
    cal = CalibratedClassifierCV(base, cv=2, method="sigmoid")
    cal.fit(X_train, y_train)
    y_pred = cal.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    save_task_metrics("intent", compute_sklearn_metrics(y_test, y_pred))

    bundle = {"encoder_model_name": model_name, "classifier": cal, "task": "intent"}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
