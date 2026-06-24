"""
Huấn luyện lại toàn bộ mô hình TF-IDF (.joblib) + chuẩn bị & train NER spaCy (thư mục ``text_nlu``).

Thứ tự:
1) train_intent_model
2) train_record_type_model
3) train_category_model
4) train_action_type_model
5) ner_prepare (jsonl -> .spacy)
7) spacy train -> ghi đè models/ner_model/model-best

Biến môi trường:
- NER_MAX_STEPS: mặc định 6000 (giảm nếu cần train nhanh).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "retrain_all_metrics.json"

# ---------------------------------------------------------------------------
# Helpers – parse sklearn / spaCy metrics from stdout
# ---------------------------------------------------------------------------

_ACCURACY_RE = re.compile(r"accuracy\s+([\d.]+)\s+(\d+)")
_MACRO_AVG_RE = re.compile(
    r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
)
_WEIGHTED_AVG_RE = re.compile(
    r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
)
# spaCy NER final line like: " 25    2600  145.88  94.50  99.98  99.96  100.00  1.00"
_NER_LINE_RE = re.compile(
    r"\s*\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def _parse_sklearn_report(stdout: str) -> dict | None:
    """Extract accuracy, macro-avg, weighted-avg from classification_report text."""
    acc_m = _ACCURACY_RE.search(stdout)
    macro_m = _MACRO_AVG_RE.search(stdout)
    weighted_m = _WEIGHTED_AVG_RE.search(stdout)
    if not acc_m:
        return None
    result: dict = {
        "accuracy": float(acc_m.group(1)),
        "test_set": int(acc_m.group(2)),
    }
    if macro_m:
        result["macro_precision"] = float(macro_m.group(1))
        result["macro_recall"] = float(macro_m.group(2))
        result["macro_f1"] = float(macro_m.group(3))
    if weighted_m:
        result["weighted_precision"] = float(weighted_m.group(1))
        result["weighted_recall"] = float(weighted_m.group(2))
        result["weighted_f1"] = float(weighted_m.group(3))
    return result


def _parse_ner_metrics(stdout: str) -> dict | None:
    """Extract last NER score line from spaCy training output."""
    last = None
    for m in _NER_LINE_RE.finditer(stdout):
        last = m
    if not last:
        return None
    return {
        "ents_f": float(last.group(1)),
        "ents_p": float(last.group(2)),
        "ents_r": float(last.group(3)),
        "score": float(last.group(4)),
    }


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def _run(script: str) -> str:
    """Run a training script and return its captured stdout."""
    path = TRAIN_DIR / script
    print(f"\n>>> python {path.name}")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(TRAIN_DIR),
        check=True,
        capture_output=True,
        text=True,
    )
    # Still print stdout/stderr so log file has it
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout or ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(TRAIN_DIR)
    metrics: dict = {}

    # 1) Intent model
    out = _run("train_intent_model.py")
    parsed = _parse_sklearn_report(out)
    if parsed:
        metrics["intent"] = parsed

    # 2) Record type model
    out = _run("train_record_type_model.py")
    parsed = _parse_sklearn_report(out)
    if parsed:
        metrics["record_type"] = parsed

    # 3) Category model
    out = _run("train_category_model.py")
    parsed = _parse_sklearn_report(out)
    if parsed:
        metrics["category"] = parsed

    # 4) Action type model
    out = _run("train_action_type_model.py")
    parsed = _parse_sklearn_report(out)
    if parsed:
        metrics["action_type"] = parsed

    # 5) NER prepare
    _run("ner_prepare.py")

    # 6) NER train
    out = _run("train_ner_only.py")
    parsed = _parse_ner_metrics(out)
    if parsed:
        metrics["ner"] = parsed

    # Save metrics to JSON for nlu.py to read
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    print("All training steps finished.")


if __name__ == "__main__":
    main()
