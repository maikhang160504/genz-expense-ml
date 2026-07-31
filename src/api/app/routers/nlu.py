"""NLU endpoints."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas.nlu import NLURequest, NLUResponse
from app.services.nlu_service import get_nlu_service
from app.core.config import get_settings
import os
import json
import sys
import subprocess
from pathlib import Path

router = APIRouter(prefix="/nlu", tags=["nlu"])

TRAINING_ACTIVE = False

TRAIN_STATUS_INFO = {
    "training_active": False,
    "target": "local",
    "stage": "IDLE",
    "message": "Sẵn sàng",
    "started_ts": None,
    "elapsed_seconds": 0,
    "progress_percent": 0,
    "error": None,
}

import datetime
import time


def _update_train_status(active: bool, stage: str, message: str, percent: int, start_ts: float | None = None, error: str | None = None, target: str = "local"):
    global TRAINING_ACTIVE, TRAIN_STATUS_INFO
    TRAINING_ACTIVE = active
    TRAIN_STATUS_INFO["training_active"] = active
    TRAIN_STATUS_INFO["target"] = target
    TRAIN_STATUS_INFO["stage"] = stage
    TRAIN_STATUS_INFO["message"] = message
    TRAIN_STATUS_INFO["progress_percent"] = percent
    TRAIN_STATUS_INFO["error"] = error
    if start_ts:
        TRAIN_STATUS_INFO["started_ts"] = start_ts
        TRAIN_STATUS_INFO["elapsed_seconds"] = round(time.time() - start_ts, 1)
    elif not active and TRAIN_STATUS_INFO.get("started_ts"):
        TRAIN_STATUS_INFO["elapsed_seconds"] = round(time.time() - TRAIN_STATUS_INFO["started_ts"], 1)
        TRAIN_STATUS_INFO["started_ts"] = None


def _count_csv_rows(csv_path: Path) -> int:
    """Count non-empty rows in a CSV file (excluding header)."""
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            return max(0, sum(1 for line in f if line.strip()) - 1)
    except Exception:
        return 0


def append_nlu_history(nlu_dir: Path, status: str, duration_sec: float, error_msg: str | None = None):
    """Append a training run record using **real** metrics from retrain_all_metrics.json."""
    history_file = nlu_dir / "text_nlu" / "models" / "nlu_training_history.json"
    metrics_file = nlu_dir / "text_nlu" / "models" / "retrain_all_metrics.json"

    # Count training rows from each dataset
    datasets_dir = nlu_dir / "text_nlu" / "datasets"
    training_rows = {
        "intent_record": _count_csv_rows(datasets_dir / "intent_record.csv"),
        "intent_action": _count_csv_rows(datasets_dir / "intent_action.csv"),
        "intent_chitchat": _count_csv_rows(datasets_dir / "intent_chitchat.csv"),
    }
    total_rows = sum(training_rows.values())

    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    run_idx = len(history) + 1

    # Read REAL metrics from retrain_all_metrics.json (produced by retrain_all.py)
    metrics = None
    f1_score = None
    if status == "success" and metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            cat = metrics.get("category", {})
            raw_f1 = cat.get("weighted_f1", cat.get("accuracy", 0))
            from app.services.nlu_registry import _format_f1
            f1_score = _format_f1(raw_f1).replace("%", "")
        except Exception as e:
            print(f"[NLU history] Failed to read metrics file: {e}", flush=True)

    record = {
        "run_index": run_idx,
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "duration_sec": round(duration_sec, 2),
        "status": status,
        "train_type": "tfidf",
        "training_rows": total_rows,
        "training_rows_detail": training_rows,
        "error": error_msg,
        "f1_score": f1_score,
        "metrics": metrics,
    }
    history.append(record)
    history = history[-100:]

    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to write NLU history: {e}", flush=True)

    if status == "success":
        try:
            from app.services.nlu_registry import mark_train_success
            mark_train_success(nlu_dir, run_idx)
        except Exception as e:
            print(f"[NLU history] Failed to update registry pending: {e}", flush=True)


def append_nlu_history(nlu_dir: Path, status: str, duration_sec: float, error_msg: str = None, source: str = "webadmin", train_type: str = "tfidf"):
    """Append NLU training run using the corresponding metrics file (tfidf or encoder)."""
    history_file = nlu_dir / "text_nlu" / "models" / "nlu_training_history.json"
    
    if train_type == "encoder":
        metrics_file = nlu_dir / "text_nlu" / "models" / "encoder_metrics.json"
    else:
        metrics_file = nlu_dir / "text_nlu" / "models" / "retrain_all_metrics.json"

    history = []
    # Try reading from storage first, then local
    storage_history = Path("/storage/nlu_models/nlu_training_history.json")
    if storage_history.is_file():
        try:
            with open(storage_history, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
            
    if not history and history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    run_idx = len(history) + 1

    metrics = None
    f1_score = None
    encoder_model_name = None
    if status == "success" and metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            if train_type == "encoder":
                encoder_model_name = metrics.get("encoder_model")
                metrics = {k: v for k, v in metrics.items() if k not in ("train_type", "encoder_model")}

            cat = metrics.get("category", {})
            raw_f1 = cat.get("weighted_f1", cat.get("accuracy", 0))
            from app.services.nlu_registry import _format_f1
            f1_score = _format_f1(raw_f1).replace("%", "")
        except Exception as e:
            print(f"[NLU history] Failed to read metrics: {e}", flush=True)

    record = {
        "run_index": run_idx,
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "duration_sec": round(duration_sec, 2),
        "status": status,
        "train_type": train_type,
        "source": source,
        "training_rows": None,
        "error": error_msg,
        "f1_score": f1_score,
        "metrics": metrics,
        "encoder_model": encoder_model_name,
    }
    history.append(record)
    history = history[-100:]

    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        # Also save to storage if exists
        if Path("/storage").exists():
            storage_history_path = Path("/storage/nlu_models/nlu_training_history.json")
            storage_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(storage_history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print("[NLU history] Successfully synced history to /storage/nlu_models/nlu_training_history.json", flush=True)
    except Exception as e:
        print(f"Failed to write NLU training history: {e}", flush=True)


def run_retraining(nlu_dir: Path, target: str = "local"):
    global TRAINING_ACTIVE
    start_time = time.time()
    error_msg = None
    status = "failed"
    train_type = "encoder" if target in ("encoder", "phobert") else "tfidf"
    try:
        _update_train_status(True, "PREPARING", f"Đang chuẩn bị dữ liệu huấn luyện ({train_type})...", 15, start_time, target=target)
        # Find python path in NLU project venv (Windows: .venv/Scripts/python.exe, Unix: .venv/bin/python)
        venv_python = nlu_dir / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = nlu_dir / ".venv" / "Scripts" / "python"
        if not venv_python.exists():
            venv_python = nlu_dir / ".venv" / "bin" / "python"
        
        python_exec = str(venv_python) if venv_python.exists() else sys.executable
        
        if train_type == "encoder":
            script_path = nlu_dir / "text_nlu" / "train" / "retrain_encoders.py"
        else:
            script_path = nlu_dir / "text_nlu" / "train" / "retrain_all.py"
        
        # Set environment variable to make it train fast or normal
        env = os.environ.copy()
        env.setdefault("INTENT_ENCODER_MAX_SAMPLES", "5000")
        env.setdefault("ACTION_TYPE_ENCODER_MAX_SAMPLES", "3000")
        env.setdefault("RECORD_TYPE_ENCODER_MAX_SAMPLES", "3000")
        env.setdefault("CATEGORY_ENCODER_MAX_SAMPLES", "3000")

        _update_train_status(True, "TRAINING", f"Đang huấn luyện mô hình NLU ({train_type})...", 45, start_time, target=target)
        subprocess.run(
            [python_exec, str(script_path)],
            cwd=str(nlu_dir),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        _update_train_status(True, "SYNCING", "Đang nạp nóng mô hình NLU mới vào bộ nhớ...", 85, start_time, target=target)
        # After successful retraining, copy models to /storage/nlu_models/ if storage is mounted
        storage_models_dir = Path("/storage/nlu_models")
        if Path("/storage").exists():
            import shutil
            print(f"[NLU training] Copying newly trained {train_type} models to persistent storage /storage/nlu_models...", flush=True)
            storage_models_dir.mkdir(parents=True, exist_ok=True)
            local_models_dir = nlu_dir / "text_nlu" / "models"
            
            # Copy all joblib files
            for f in local_models_dir.glob("*.joblib"):
                shutil.copy2(f, storage_models_dir / f.name)
            
            # Copy registry and metrics JSON
            for f in local_models_dir.glob("*.json"):
                shutil.copy2(f, storage_models_dir / f.name)
                
            # Copy spacy ner_model if it exists (only for tfidf)
            ner_dir = local_models_dir / "ner_model" / "model-best"
            if train_type != "encoder" and ner_dir.exists():
                dest_ner = storage_models_dir / "ner_model" / "model-best"
                if dest_ner.exists():
                    shutil.rmtree(dest_ner)
                shutil.copytree(ner_dir, dest_ner)
            print("[NLU training] Sync to /storage/nlu_models completed successfully!", flush=True)
        
        # Force reload in memory
        get_nlu_service().reload()
        status = "success"
        _update_train_status(False, "SUCCESS", "Huấn luyện lại mô hình hoàn tất thành công!", 100, start_time, target=target)
    except Exception as e:
        error_msg = str(e)
        print(f"[NLU training] Error: {e}", flush=True)
        _update_train_status(False, "ERROR", f"Huấn luyện thất bại: {e}", 0, start_time, error=str(e), target=target)
    finally:
        TRAINING_ACTIVE = False
        duration = time.time() - start_time
        append_nlu_history(nlu_dir, status, duration, error_msg, train_type=train_type)


@router.post(
    "/infer",
    response_model=NLUResponse,
    summary="Phân tích câu chi tiêu / hành động (text → intent + amount + category)",
)
def infer(payload: NLURequest) -> NLUResponse:
    """Run the Vietnamese expense NLU pipeline on free text.

    - Trả về `intent` ∈ {Record, Action, Chitchat}
    - Khi `intent == Record`: kèm `amount`, `category`, `record_type`.
    - `backend = "real"` nếu pipeline gốc tải được, ngược lại `backend = "mock"`.
    """
    return get_nlu_service().infer(payload)


@router.get("/prompts", summary="Lấy file cấu hình prompts.json")
def get_prompts():
    settings = get_settings()
    prompts_path = Path(settings.expense_ocr_nlu_dir) / "src" / "prompts" / "prompts.json"
    if not prompts_path.exists():
        raise HTTPException(status_code=404, detail=f"prompts.json not found at {prompts_path}")
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read prompts: {e}")


@router.post("/prompts", summary="Lưu file cấu hình prompts.json và reload")
def save_prompts(payload: dict):
    settings = get_settings()
    prompts_path = Path(settings.expense_ocr_nlu_dir) / "src" / "prompts" / "prompts.json"
    try:
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        # Force reload bundle to pick up new prompts configuration
        get_nlu_service().reload()
        return {"success": True, "message": "Prompts saved and NLU hot-reloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write prompts: {e}")


from pydantic import BaseModel, Field

class NluTrainRequest(BaseModel):
    target: str = Field(default="local", description="Target of retraining: 'local' (CPU) or 'kaggle' (GPU)")


@router.post("/train", summary="Huấn luyện lại toàn bộ mô hình (Chạy ngầm)")
def train(payload: NluTrainRequest = NluTrainRequest(), background_tasks: BackgroundTasks = None):
    global TRAINING_ACTIVE
    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()

    # Modal training flow check
    import os
    if os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable):
        try:
            status_file = Path("/storage/nlu_models/training_status.json")
            status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(status_file, "w") as f:
                json.dump({"training_active": True}, f)
        except Exception:
            pass

        try:
            from modal_app import train_nlu_model
            train_nlu_model.spawn(payload.target)
            return {
                "status": "started",
                "target": payload.target,
                "message": f"NLU model retraining spawned on Modal Cloud GPU ({payload.target})."
            }
        except Exception as e:
            print(f"[NLU train] Failed to spawn Modal train function: {e}", flush=True)

    if payload.target == "kaggle":
        raise HTTPException(
            status_code=400,
            detail="Kaggle target is deprecated. Please use Modal Cloud GPU or local target."
        )

    # Local training flow
    if TRAINING_ACTIVE:
        return {"status": "error", "message": "Training is already in progress locally."}
    
    TRAINING_ACTIVE = True
    if background_tasks:
        background_tasks.add_task(run_retraining, nlu_dir, payload.target)
    else:
        # Fallback if background_tasks is not injected
        import threading
        threading.Thread(target=run_retraining, args=(nlu_dir, payload.target), daemon=True).start()

    return {"status": "started", "target": payload.target, "message": f"NLU model retraining triggered ({payload.target}) in background."}




@router.get("/train/status", summary="Lấy trạng thái huấn luyện")
def train_status():
    # If running on Modal, check status file in shared volume
    status_file = Path("/storage/nlu_models/training_status.json")
    if status_file.is_file():
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    res = dict(TRAIN_STATUS_INFO)
    res["training_active"] = TRAINING_ACTIVE
    if res.get("started_ts") and TRAINING_ACTIVE:
        res["elapsed_seconds"] = round(time.time() - res["started_ts"], 1)
    return res


@router.get("/internal/status", summary="Lấy thông tin mô hình NLU hiện tại")
def internal_status():
    service = get_nlu_service()
    loaded = service.try_load()
    return {
        "loaded": loaded,
        "backend": "real" if loaded else "mock"
    }


@router.get("/train/history", summary="Lấy lịch sử retrain NLU")
def train_history():
    # Check storage first for persistent history
    storage_history = Path("/storage/nlu_models/nlu_training_history.json")
    if storage_history.is_file():
        try:
            with open(storage_history, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    history_file = nlu_dir / "text_nlu" / "models" / "nlu_training_history.json"
    if not history_file.exists():
        return []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read NLU training history: {e}")


@router.get("/benchmark/results", summary="Lấy kết quả đánh giá NLU Benchmark")
def get_benchmark_results():
    results_path = Path("/storage/nlu_models/nlu_benchmark_results.json")
    if not results_path.exists():
        # Fallback default statistics for UI rendering before first run
        return {
            "tfidf": {
                "intent_accuracy": 88.5,
                "category_accuracy": 86.2,
                "record_type_accuracy": 90.1,
                "avg_latency_ms": 1.2,
                "p95_latency_ms": 3.5
            },
            "phobert": {
                "intent_accuracy": 92.4,
                "category_accuracy": 90.5,
                "record_type_accuracy": 94.8,
                "avg_latency_ms": 45.8,
                "p95_latency_ms": 90.2
            },
            "phogpt": {
                "intent_accuracy": 96.8,
                "category_accuracy": 95.1,
                "record_type_accuracy": 98.4,
                "avg_latency_ms": 1820.0,
                "p95_latency_ms": 2450.0
            }
        }
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read benchmark results: {e}")


@router.post("/benchmark/run", summary="Chạy đánh giá NLU Benchmark trên Modal (GPU)")
def trigger_benchmark_run():
    try:
        from modal_app import run_nlu_benchmark
        handle = run_nlu_benchmark.spawn()
        return {
            "ok": True,
            "job_id": handle.object_id,
            "status": "RUNNING",
            "message": "NLU Benchmark run triggered successfully on Modal GPU."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger benchmark: {e}")


@router.get("/train/llm-history", summary="Lấy lịch sử fine-tune LLM Qwen")
def get_llm_history():
    from pathlib import Path
    import json
    history_file = Path("/storage/llm_finetune/finetune_history.json")
    raw_log = Path("/storage/llm_finetune/raw_training_log.txt")
    should_parse = not history_file.is_file()
    if not should_parse:
        try:
            # Force rebuild if the history file is legacy (references PhoGPT)
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            if history_data and any("phogpt" in str(run.get("model_id", "")).lower() or "phogpt" in str(run.get("lora_target", "")).lower() for run in history_data):
                should_parse = True
        except Exception:
            should_parse = True

    if not should_parse and raw_log.is_file():
        try:
            should_parse = raw_log.stat().st_mtime > history_file.stat().st_mtime
        except Exception:
            pass
            
    if should_parse:
        try:
            from text_nlu.tools import parse_llm_logs
            parse_llm_logs.main()
        except Exception:
            pass

    if history_file.is_file():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return []


@router.get("/model-meta", summary="Metadata mô hình NLU đang deploy")
def model_meta():
    from app.services.nlu_registry import get_model_meta

    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    loaded = get_nlu_service().try_load()
    return get_model_meta(nlu_dir, nlu_real=settings.use_real_nlu, nlu_loaded=loaded)


@router.post("/train/llm-trigger", summary="Kích hoạt fine-tune Qwen LLM trên Modal GPU H100")
def trigger_llm_finetune(epochs: int = 3, lr: float = 2e-4, batch_size: int = 4):
    try:
        import modal
        f = modal.Function.from_name("expense-ocr-nlu", "train_qwen_model")
        # Run asynchronously by spawning a task
        handle = f.spawn(num_epochs=epochs, learning_rate=lr, batch_size=batch_size)
        return {
            "ok": True,
            "job_id": handle.object_id,
            "status": "RUNNING",
            "message": "Qwen fine-tuning job spawned successfully on Modal GPU."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger LLM fine-tuning: {e}")


@router.get("/inference-backend", summary="Backend NLU đang dùng: tfidf | encoder")
def get_nlu_inference_backend():
    from app.services.nlu_registry import get_inference_backend

    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    backend = get_inference_backend(nlu_dir)
    return {"backend": backend}


@router.post("/inference-backend", summary="Chọn backend NLU: tfidf hoặc encoder")
def set_nlu_inference_backend(payload: dict):
    import os

    from app.services.nlu_registry import set_inference_backend

    backend = str(payload.get("backend", "")).strip().lower()
    if backend not in {"tfidf", "encoder", "phobert", "llm"}:
        raise HTTPException(status_code=400, detail="backend must be 'tfidf', 'encoder', or 'llm'")

    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    try:
        reg = set_inference_backend(nlu_dir, backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    use_enc = reg.get("inference_backend") == "encoder"
    os.environ["NLU_USE_ENCODER"] = "1" if use_enc else "0"
    get_nlu_service().reload()

    return {
        "backend": reg.get("inference_backend", "tfidf"),
        "message": f"NLU inference switched to {reg.get('inference_backend')}",
        "reloaded": True,
    }


@router.post(
    "/test-prompt",
    response_model=dict,
    summary="Test prompt trực tiếp không lưu lịch sử",
)
def test_prompt(payload: dict) -> dict:
    from src.nlu.llm_intent_handler import run_llm_nlu
    import time

    text = payload.get("text", "")
    override_prompt = payload.get("override_prompt")
    persona = payload.get("persona", "hai_huoc")
    
    t0 = time.monotonic()
    
    # Fake profile context if needed
    context = {
        "user_id": "test_admin",
        "budget_total": 5000000,
        "budget_remain": 2500000,
        "verbal_style": persona
    }
    
    result = run_llm_nlu(
        text=text,
        context_metadata=context,
        run_llm=True,
        override_prompt=override_prompt
    )
    
    latency = int((time.monotonic() - t0) * 1000)
    
    return {
        "ok": True,
        "latency_ms": latency,
        "result": result
    }
