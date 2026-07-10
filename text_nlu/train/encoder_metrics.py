"""Collect encoder training metrics in the same shape as retrain_all_metrics.json (subset)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    return {
        "accuracy": round(acc, 4),
        "test_set": int(len(y_true)),
        "weighted_precision": round(float(wp), 4),
        "weighted_recall": round(float(wr), 4),
        "weighted_f1": round(float(wf1), 4),
        "macro_precision": round(float(mp), 4),
        "macro_recall": round(float(mr), 4),
        "macro_f1": round(float(mf1), 4),
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
    METRICS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return METRICS_PATH
