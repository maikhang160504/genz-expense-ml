"""
Huấn luyện lại toàn bộ mô hình TF-IDF (.joblib) + chuẩn bị & train NER spaCy (thư mục ``text_nlu``).

Thứ tự:
1) train_intent_model
2) train_record_type_model
3) train_category_model
4) train_action_type_model
5) train_action_slots  (+ action_slots_metrics.json)
6) ner_prepare (jsonl -> .spacy)
7) train_ner_only -> ghi đè models/ner_model/model-best

Dataset intent_action.csv: gộp / label thủ công trước khi train (không tự merge trong pipeline).
c
Biến môi trường:
- NER_MAX_STEPS: mặc định 6000 (giảm nếu cần train nhanh).
"""
from __future__ import annotations

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
METRICS_PATH = ROOT / "models" / "retrain_all_metrics.json"
SLOTS_METRICS_PATH = ROOT / "models" / "action_slots_metrics.json"

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
    path = (TRAIN_DIR / script).resolve()
    if not path.is_file():
        path = (ROOT / script).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training script not found: {script}")
    print(f"\n>>> python {path}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(path.parent),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout or ""


def _ensure_action_slot_labels() -> None:
    """Auto-label intent_action.csv when slot columns are missing (Kaggle / fresh CSV)."""
    import pandas as pd

    from load_slot_schema import import_slot_schema

    SLOT_COLUMNS = import_slot_schema().SLOT_COLUMNS

    csv_path = ROOT / "datasets" / "intent_action.csv"
    label_script = ROOT / "datasets" / "label_action_slots.py"
    if not csv_path.is_file():
        return
    if not label_script.is_file():
        print("[WARN] label_action_slots.py missing — skip auto-label (CSV must have slot columns)")
        return

    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing_cols = [c for c in SLOT_COLUMNS if c not in header]
    if not missing_cols:
        usecols = [c for c in SLOT_COLUMNS if c in header]
        sample = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=usecols)
        if sample.apply(
            lambda r: any(str(v).strip().lower() not in ("", "nan") for v in r),
            axis=1,
        ).any():
            return

    print("\n>>> Slot columns missing or empty — running label_action_slots.py")
    _run("../datasets/label_action_slots.py")


def _load_action_slots_metrics() -> dict | None:
    if not SLOTS_METRICS_PATH.is_file():
        return None
    try:
        return json.loads(SLOTS_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(TRAIN_DIR)
    metrics: dict = {}
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

    # 5) Action slot models
    _ensure_action_slot_labels()
    _run("train_action_slots.py")
    slots_metrics = _load_action_slots_metrics()
    if slots_metrics:
        metrics["action_slots"] = slots_metrics

    # 6) NER prepare
    _run("ner_prepare.py")

    # 7) NER train
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
