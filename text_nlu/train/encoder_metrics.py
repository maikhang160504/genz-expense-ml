"""Collect encoder training metrics in the same shape as retrain_all_metrics.json (subset)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = Path(os.environ.get("ENCODER_METRICS_OUT", str(ROOT / "models" / "encoder_metrics.json")))


def compute_sklearn_metrics(y_true, y_pred) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Per-class classification report (dict format)
    per_class = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(acc, 4),
        "test_set": int(len(y_true)),
        "weighted_precision": round(float(wp), 4),
        "weighted_recall": round(float(wr), 4),
        "weighted_f1": round(float(wf1), 4),
        "macro_precision": round(float(mp), 4),
        "macro_recall": round(float(mr), 4),
        "macro_f1": round(float(mf1), 4),
        "per_class_report": {k: v for k, v in per_class.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }


def reset_metrics() -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text("{}", encoding="utf-8")


def load_metrics() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_task_metrics(task_key: str, block: dict) -> None:
    data = load_metrics()
    data[task_key] = block
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_metrics(*, model_name: str | None = None) -> Path:
    data = load_metrics()
    data["train_type"] = "encoder"
    data["encoder_model"] = model_name or os.environ.get("ENCODER_MODEL_NAME", "vinai/phobert-base")
    content = json.dumps(data, ensure_ascii=False, indent=2)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(content, encoding="utf-8")
    
    if Path("/storage").is_dir():
        for storage_dir in [Path("/storage/nlu_models_candidate"), Path("/storage/nlu_models")]:
            try:
                storage_dir.mkdir(parents=True, exist_ok=True)
                (storage_dir / "encoder_metrics.json").write_text(content, encoding="utf-8")
                print(f"✅ Synced encoder metrics to {storage_dir}/encoder_metrics.json")
            except Exception:
                pass

    return METRICS_PATH
