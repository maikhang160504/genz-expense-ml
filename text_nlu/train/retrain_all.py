"""
Huấn luyện lại mô hình TF-IDF 2 tầng (Intent, Category).

Thứ tự:
1) export_dataset (Kéo dữ liệu từ PostgreSQL, lọc is_intent_wrong)
2) train_intent_model (Tầng 1)
3) train_category_model (Tầng 2)
"""
from __future__ import annotations

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
METRICS_PATH = ROOT / "models_new" / "retrain_all_metrics.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACCURACY_RE = re.compile(r"accuracy\s+([\d.]+)\s+(\d+)")
_MACRO_AVG_RE = re.compile(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)")
_WEIGHTED_AVG_RE = re.compile(r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)")

def _parse_sklearn_report(stdout: str) -> dict | None:
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

def _run(script: str) -> str:
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(TRAIN_DIR)
    metrics: dict = {}
    
    # 1) Lọc và xuất dữ liệu từ DB (PostgreSQL)
    try:
        _run("export_dataset.py")
    except Exception as e:
        print(f"Lỗi xuất dataset: {e}")
        # Không throw, cứ tiếp tục dùng dataset cũ nếu xuất lỗi
    
    # 2) Intent model (Stage 1)
    out = _run("train_intent_model.py")
    parsed = _parse_sklearn_report(out)
    if parsed:
        metrics["intent"] = parsed

    # 3) Category model (Stage 2)
    out = _run("train_category_model.py")
    parsed = _parse_sklearn_report(out)
    if parsed:
        metrics["category"] = parsed

    # Save metrics to models_new
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")
    
    # Đồng bộ metrics trực tiếp vào /storage vĩnh viễn
    if Path("/storage").is_dir():
        for storage_dir in [Path("/storage/nlu_models_candidate"), Path("/storage/nlu_models")]:
            try:
                storage_dir.mkdir(parents=True, exist_ok=True)
                dest_m = storage_dir / "retrain_all_metrics.json"
                with open(dest_m, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                print(f"✅ Synced metrics to persistent storage: {dest_m}")
            except Exception as e:
                print(f"⚠️ Failed to sync metrics to {storage_dir}: {e}")
                
    print("All training steps finished.")

if __name__ == "__main__":
    main()
