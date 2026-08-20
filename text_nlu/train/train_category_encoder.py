"""
PhoBERT embedding + logistic category.

Đầu ra: text_nlu/models/category_encoder.joblib
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

from encoder_data import load_category_rows, xy_from_df  # noqa: E402
from encoder_embed import embed_texts  # noqa: E402
from encoder_metrics import compute_sklearn_metrics, save_task_metrics  # noqa: E402

OUT_PATH = Path(
    os.environ.get("CATEGORY_ENCODER_OUT", str(ROOT / "models" / "category_encoder.joblib"))
)


def main() -> None:
    model_name = os.environ.get("ENCODER_MODEL_NAME", "vinai/phobert-base")
    df = load_category_rows()
    print(f"Category rows: {len(df)}  labels={df['label'].nunique()}")

    texts, y = xy_from_df(df, "label")
    X = embed_texts(texts, model_name, max_length=96, batch_size=8)
    strat = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    base = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    cal = CalibratedClassifierCV(base, cv=2, method="sigmoid")
    cal.fit(X_train, y_train)
    y_pred = cal.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    save_task_metrics("category", compute_sklearn_metrics(y_test, y_pred))

    bundle = {"encoder_model_name": model_name, "classifier": cal, "task": "category"}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved {OUT_PATH}")

    # Đồng bộ trực tiếp vào /storage vĩnh viễn nếu môi trường hỗ trợ
    if Path("/storage").is_dir():
        for storage_dir in [Path("/storage/nlu_models_candidate"), Path("/storage/nlu_models")]:
            try:
                storage_dir.mkdir(parents=True, exist_ok=True)
                dest = storage_dir / "category_encoder.joblib"
                joblib.dump(bundle, dest)
                print(f"✅ Synced category encoder to persistent storage: {dest}")
            except Exception as e:
                print(f"⚠️ Failed to sync category encoder to {storage_dir}: {e}")


if __name__ == "__main__":
    main()
