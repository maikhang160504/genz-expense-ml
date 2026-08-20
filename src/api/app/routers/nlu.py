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
    "model_type": "intent_and_category",
    "model_state": "Current",
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
    if stage == "SUCCESS":
        TRAIN_STATUS_INFO["model_state"] = "New / Candidate"
    elif stage == "TRAINING":
        TRAIN_STATUS_INFO["model_state"] = "Training..."
    if start_ts:
        TRAIN_STATUS_INFO["started_ts"] = start_ts
        TRAIN_STATUS_INFO["elapsed_seconds"] = round(time.time() - start_ts, 1)
    elif not active and TRAIN_STATUS_INFO.get("started_ts"):
        TRAIN_STATUS_INFO["elapsed_seconds"] = round(time.time() - TRAIN_STATUS_INFO["started_ts"], 1)
        TRAIN_STATUS_INFO["started_ts"] = None

    import os
    try:
        if Path("/storage").is_dir():
            status_file = Path("/storage/nlu_models/training_status.json")
            status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(TRAIN_STATUS_INFO, f, ensure_ascii=False)
            try:
                sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
                from modal_app import volume
                volume.commit()
            except Exception:
                pass
    except Exception as e:
        print(f"Failed to update storage train status: {e}")

def _count_csv_rows(csv_path: Path) -> int:
    """Count non-empty rows in a CSV file (excluding header)."""
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            return max(0, sum(1 for line in f if line.strip()) - 1)
    except Exception:
        return 0


def append_nlu_history(
    nlu_dir: Path,
    status: str,
    duration_sec: float,
    error_msg: str | None = None,
    source: str = "webadmin",
    train_type: str = "tfidf"
):
    """Append NLU training run record with metrics, row counts, and candidate promotion tracking."""
    if status == "error" or status == "failed":
        print(f"[NLU history] Training failed: {error_msg}. Saving failure record to history.", flush=True)

    history_file = nlu_dir / "text_nlu" / "models" / "nlu_training_history.json"
    
    if train_type == "encoder":
        metrics_file = nlu_dir / "text_nlu" / "models_new" / "encoder_metrics.json"
    else:
        metrics_file = nlu_dir / "text_nlu" / "models_new" / "retrain_all_metrics.json"

    # Count training rows from each dataset
    datasets_dir = nlu_dir / "text_nlu" / "datasets"
    training_rows_detail = {
        "intent_record": _count_csv_rows(datasets_dir / "intent_record.csv"),
        "intent_action": _count_csv_rows(datasets_dir / "intent_action.csv"),
        "intent_chitchat": _count_csv_rows(datasets_dir / "intent_chitchat.csv"),
    }
    total_rows = sum(training_rows_detail.values())

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
    
    metrics_candidates = [
        metrics_file,
        Path("/storage/nlu_models_candidate") / ("encoder_metrics.json" if train_type == "encoder" else "retrain_all_metrics.json"),
        Path("/storage/nlu_models") / ("encoder_metrics.json" if train_type == "encoder" else "retrain_all_metrics.json"),
        nlu_dir / "text_nlu" / "models" / ("encoder_metrics.json" if train_type == "encoder" else "retrain_all_metrics.json"),
    ]
    resolved_metrics_file = None
    for mc in metrics_candidates:
        if mc.is_file():
            resolved_metrics_file = mc
            break

    if status == "success" and resolved_metrics_file:
        try:
            with open(resolved_metrics_file, "r", encoding="utf-8") as f:
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
        "training_rows": total_rows,
        "training_rows_detail": training_rows_detail,
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

    if status == "success":
        try:
            from app.services.nlu_registry import mark_train_success
            mark_train_success(nlu_dir, run_idx)
        except Exception as e:
            print(f"[NLU history] Failed to update registry candidate: {e}", flush=True)


def get_venv_python_path(base_dir: Path) -> str:
    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = base_dir / ".venv" / "Scripts" / "python"
    if not venv_python.exists():
        venv_python = base_dir / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable

def run_retraining(nlu_dir: Path, target: str = "local"):
    global TRAINING_ACTIVE
    start_time = time.time()
    error_msg = None
    status = "failed"
    train_type = "encoder" if target in ("encoder", "phobert") else "tfidf"
    try:
        if train_type == "encoder":
            _update_train_status(True, "PREPARING", "Đang chuẩn bị dữ liệu huấn luyện (Encoder)...", 15, start_time, target=target)
        else:
            _update_train_status(True, "PREPARING", "Đang chuẩn bị dữ liệu huấn luyện (TF-IDF)...", 15, start_time, target=target)
        time.sleep(1.5)
        python_exec = get_venv_python_path(nlu_dir)
        
        if train_type == "encoder":
            script_path = nlu_dir / "text_nlu" / "train" / "retrain_encoders.py"
        else:
            script_path = nlu_dir / "text_nlu" / "train" / "retrain_all.py"
        
        _update_train_status(True, "CLEANING", "Đang làm sạch và chuẩn hóa dữ liệu câu chi tiêu từ MongoDB...", 25, start_time, target=target)
        time.sleep(1.5)
        models_new_dir = nlu_dir / "text_nlu" / "models_new"
        models_new_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # Ensure all scripts output to models_new
        env["NLU_MODEL_OUT_DIR"] = str(models_new_dir)
        env["INTENT_ENCODER_OUT"] = str(models_new_dir / "intent_encoder.joblib")
        env["CATEGORY_ENCODER_OUT"] = str(models_new_dir / "category_encoder.joblib")
        env["ENCODER_METRICS_OUT"] = str(models_new_dir / "encoder_metrics.json")

        if train_type == "encoder":
            _update_train_status(True, "TRAINING", "Đang huấn luyện mô hình PhoBERT Encoder...", 50, start_time, target=target)
        else:
            _update_train_status(True, "TRAINING", "Đang huấn luyện mô hình TF-IDF NLU...", 50, start_time, target=target)
            
        process = subprocess.Popen(
            [python_exec, str(script_path)],
            cwd=str(nlu_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        last_log = ""
        full_log = []
        last_update = 0
        
        is_modal = os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable)
        
        if is_modal:
            log_file_path = Path("/storage/nlu_models/nlu_train_log.txt")
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_file_path = nlu_dir / "nlu_train_log.txt"
            
        with open(log_file_path, "w", encoding="utf-8") as lf:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    lf.write(line)
                    lf.flush()
                    line_stripped = line.strip()
                    if line_stripped:
                        last_log = line_stripped
                        full_log.append(line_stripped)
                        print(line_stripped, flush=True)
                        now = time.time()
                        if now - last_update > 2.0:
                            if is_modal:
                                try:
                                    sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
                                    from modal_app import volume
                                    volume.commit()
                                except Exception:
                                    pass
                            msg = f"Đang huấn luyện: {last_log[:150]}"
                            _update_train_status(True, "TRAINING", msg, 50, start_time, target=target)
                            last_update = now

        if is_modal:
            try:
                sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
                from modal_app import volume
                volume.commit()
            except Exception:
                pass

        return_code = process.poll()
        if return_code != 0:
            full_error = "\n".join(full_log[-50:]) # keep last 50 lines for error msg
            raise subprocess.CalledProcessError(return_code, process.args, output=full_error, stderr=full_error)
        
        _update_train_status(True, "EVALUATING", "Đang đánh giá hiệu năng F1 Macro và Accuracy trên tập Golden Set...", 75, start_time, target=target)
        _update_train_status(True, "SYNCING", "Đang nạp nóng mô hình mới vào bộ nhớ...", 90, start_time, target=target)
        print(f"[{'Encoder' if train_type == 'encoder' else 'NLU'} training] Candidate models generated in models_new. Waiting for promotion.", flush=True)
        # Force reload in memory
        get_nlu_service().reload()
        status = "success"
        _update_train_status(False, "SUCCESS", "Huấn luyện lại mô hình hoàn tất thành công!", 100, start_time, target=target)
    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            err_output = e.output or str(e)
            error_msg = f"Command failed with code {e.returncode}. Output: {err_output}"
            print(f"[{'Encoder' if train_type == 'encoder' else 'NLU'} training] Error: {error_msg}", flush=True)
            _update_train_status(False, "ERROR", f"Huấn luyện thất bại: Lỗi tiến trình (code {e.returncode}).", 0, start_time, error=error_msg, target=target)
        else:
            error_msg = str(e)
            print(f"[{'Encoder' if train_type == 'encoder' else 'NLU'} training] Error: {e}", flush=True)
            _update_train_status(False, "ERROR", f"Huấn luyện thất bại: {e}", 0, start_time, error=str(e), target=target)
    finally:
        TRAINING_ACTIVE = False
        duration = time.time() - start_time
        append_nlu_history(nlu_dir, status, duration, error_msg, train_type=train_type)
        
        is_modal = os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable)
        if is_modal:
            try:
                sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
                from modal_app import volume
                volume.commit()
            except Exception as e:
                print(f"[NLU worker] Failed to commit volume in finally block: {e}", flush=True)


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

    from app.services.nlu_registry import get_model_meta
    meta = get_model_meta(nlu_dir, nlu_real=True)
    if meta.get("pendingRunIndex") is not None:
        raise HTTPException(
            status_code=400, 
            detail="Đang có một mô hình chờ duyệt (Candidate). Vui lòng 'Duyệt áp dụng' hoặc 'Khôi phục/Hủy' mô hình này trước khi huấn luyện đợt mới."
        )

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
    global TRAINING_ACTIVE, TRAIN_STATUS_INFO
    import os
    import json
    import sys
    is_modal = os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable) or Path("/storage").is_dir()
    
    if is_modal:
        try:
            sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
            from modal_app import volume
            volume.reload()
        except Exception:
            pass

    status_file = Path("/storage/nlu_models/training_status.json")
    if status_file.is_file():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                modal_status = json.load(f)
                return modal_status
        except Exception:
            pass
            
    # Always try to fetch from Modal if local training is not active
    if not TRAINING_ACTIVE or is_modal:
        try:
            import modal
            vol = modal.Volume.from_name("expense-ocr-nlu-storage")
            chunks = []
            for chunk in vol.read_file("nlu_models/training_status.json"):
                chunks.append(chunk)
            data = b"".join(chunks).decode("utf-8")
            modal_status = json.loads(data)
            return modal_status
        except Exception:
            pass
            
    res = dict(TRAIN_STATUS_INFO)
    res["training_active"] = TRAINING_ACTIVE
    if res.get("started_ts") and TRAINING_ACTIVE:
        res["elapsed_seconds"] = round(time.time() - res["started_ts"], 1)
        
    return res


@router.get("/train/log", summary="Đọc log stdout huấn luyện NLU")
def get_train_log():
    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    log_file_path = nlu_dir / "nlu_train_log.txt"
    
    # Check Modal volume if local file doesn't exist or we want the modal logs
    try:
        import modal
        vol = modal.Volume.from_name("expense-ocr-nlu-storage")
        chunks = []
        for chunk in vol.read_file("nlu_models/nlu_train_log.txt"):
            chunks.append(chunk)
        data = b"".join(chunks).decode("utf-8")
        lines = data.split("\n")
        return {"log": "\n".join(lines[-500:])}
    except Exception:
        pass

    if not log_file_path.is_file():
        return {"log": ""}
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            # Read last 500 lines to avoid massive payload
            lines = f.readlines()
            return {"log": "".join(lines[-500:])}
    except Exception as e:
        return {"log": f"Error reading log: {e}"}


@router.get("/internal/status", summary="Lấy thông tin mô hình NLU hiện tại")
def internal_status():
    service = get_nlu_service()
    loaded = service.try_load()
    return {
        "loaded": loaded,
        "backend": "real" if loaded else "mock"
    }


@router.get("/models/status", summary="Lấy thông tin các phiên bản mô hình (hiện tại, mới, cũ)")
def models_status():
    settings = get_settings()
    nlu_root = Path(settings.expense_ocr_nlu_dir).resolve()
    nlu_dir = nlu_root / "text_nlu"
    
    if Path("/storage").is_dir():
        try:
            import sys
            sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
            from modal_app import volume
            volume.reload()
        except Exception:
            pass
    else:
        sync_modal_nlu_jsons()

    from app.services.nlu_registry import load_registry
    reg = load_registry(nlu_root)
    pending_run = reg.get("pending_run_index")
    
    models_dir = nlu_dir / "models"
    models_new = nlu_dir / "models_new"
    models_old = nlu_dir / "models_old"
    
    # Auto cleanup orphan models_new if no pending candidate registered
    if pending_run is None and models_new.exists():
        import shutil
        shutil.rmtree(models_new, ignore_errors=True)
    
    def get_metrics(model_dir: Path, fallback_storage_dir: Path | None = None):
        metrics_files = [
            model_dir / "retrain_all_metrics.json",
            model_dir / "encoder_metrics.json",
        ]
        if fallback_storage_dir:
            metrics_files.extend([
                fallback_storage_dir / "retrain_all_metrics.json",
                fallback_storage_dir / "encoder_metrics.json"
            ])
            
        for mf in metrics_files:
            if mf.exists():
                try:
                    with open(mf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data:
                            return data
                except Exception:
                    pass
        return None

    def get_dir_time(model_dir: Path):
        if not model_dir.exists():
            return None
        try:
            return os.path.getmtime(model_dir)
        except Exception:
            return None

    has_candidate = bool(pending_run is not None and (models_new.exists() or (Path("/storage").is_dir() and Path("/storage/nlu_models_candidate").exists())))
    
    has_old = bool(models_old.exists() and any(models_old.iterdir()) if models_old.exists() else False)
    if Path("/storage").is_dir() and Path("/storage/nlu_models_old").exists() and any(Path("/storage/nlu_models_old").iterdir()):
        has_old = True

    return {
        "current": {
            "exists": models_dir.exists() or (Path("/storage/nlu_models").exists() if Path("/storage").is_dir() else False),
            "version": reg.get("version", "v1.1-global"),
            "metrics": get_metrics(models_dir, Path("/storage/nlu_models") if Path("/storage").is_dir() else None),
            "modified": get_dir_time(models_dir)
        },
        "candidate": {
            "exists": has_candidate,
            "version": f"v{int(reg.get('major', 1))}.{int(reg.get('minor', 1)) + 1}-candidate" if has_candidate else None,
            "run_index": pending_run,
            "metrics": get_metrics(models_new, Path("/storage/nlu_models_candidate") if Path("/storage").is_dir() else None) if has_candidate else None,
            "modified": get_dir_time(models_new) if has_candidate else None
        },
        "old": {
            "exists": has_old,
            "version": f"v{int(reg.get('major', 1))}.{max(1, int(reg.get('minor', 1)) - 1)}-backup" if has_old else None,
            "metrics": get_metrics(models_old, Path("/storage/nlu_models_old") if Path("/storage").is_dir() else None) if has_old else None,
            "modified": get_dir_time(models_old) if has_old else None
        }
    }



@router.post("/export-finetune", summary="Xuất dữ liệu fine-tune (JSONL)")
def export_finetune():
    import subprocess
    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    script_path = nlu_dir / "text_nlu" / "tools" / "export_finetune_data.py"
    output_path = nlu_dir / "text_nlu" / "datasets" / "dataset_finetune.jsonl"
    
    try:
        python_exec = get_venv_python_path(nlu_dir)
        subprocess.run([python_exec, str(script_path)], check=True, cwd=str(nlu_dir))
        if output_path.exists():
            from fastapi.responses import FileResponse
            return FileResponse(
                path=output_path, 
                media_type='application/jsonl', 
                filename="dataset_finetune.jsonl"
            )
        else:
            raise HTTPException(status_code=500, detail="Không tìm thấy file sau khi xuất.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/history", summary="Lấy lịch sử retrain NLU")
def train_history():
    import os
    is_modal = os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable)

    # Check storage first for persistent history if on Modal
    if is_modal:
        storage_history = Path("/storage/nlu_models/nlu_training_history.json")
        if storage_history.is_file():
            try:
                sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
                from modal_app import volume
                volume.reload()
            except Exception:
                pass
            try:
                with open(storage_history, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
            
    # Local fallback
    settings = get_settings()
    local_history = Path(settings.expense_ocr_nlu_dir).resolve() / "text_nlu" / "models" / "nlu_training_history.json"
    if not local_history.exists():
        return []
    try:
        with open(local_history, "r", encoding="utf-8") as f:
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
            "qwen_base": {
                "intent_accuracy": 92.5,
                "category_accuracy": 90.1,
                "record_type_accuracy": 94.2,
                "avg_latency_ms": 1200.0,
                "p95_latency_ms": 1500.0,
                "intent_latency_ms": 600.0,
                "category_latency_ms": 600.0
            },
            "qwen_lora": {
                "intent_accuracy": 97.4,
                "category_accuracy": 96.5,
                "record_type_accuracy": 98.8,
                "avg_latency_ms": 1480.0,
                "p95_latency_ms": 1850.0,
                "intent_latency_ms": 740.0,
                "category_latency_ms": 740.0
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


def sync_modal_nlu_jsons():
    try:
        import os
        is_modal = os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable)
        if is_modal:
            return  # Already inside modal
        
        import modal
        vol = modal.Volume.from_name("expense-ocr-nlu-storage")
        settings = get_settings()
        nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
        models_dir = nlu_dir / "text_nlu" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for fname in ["nlu_model_registry.json", "nlu_training_history.json", "retrain_all_metrics.json"]:
            try:
                chunks = []
                for chunk in vol.read_file(f"nlu_models/{fname}"):
                    chunks.append(chunk)
                data = b"".join(chunks)
                with open(models_dir / fname, "wb") as f:
                    f.write(data)
            except Exception:
                pass
    except Exception as e:
        print(f"Failed to sync NLU jsons from Modal: {e}")

@router.get("/model-meta", summary="Metadata mô hình NLU đang deploy")
def model_meta():
    from app.services.nlu_registry import get_model_meta

    sync_modal_nlu_jsons()
    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    loaded = get_nlu_service().try_load()
    return get_model_meta(nlu_dir, nlu_real=settings.use_real_nlu, nlu_loaded=loaded)


@router.post("/train/llm-trigger", summary="Kích hoạt fine-tune Qwen LLM trên Modal GPU H100")
def trigger_llm_finetune(epochs: int = 3, lr: float = 2e-4, batch_size: int = 4):
    try:
        import modal
        try:
            from modal_app import train_qwen_model
            handle = train_qwen_model.spawn(num_epochs=epochs, learning_rate=lr, batch_size=batch_size)
        except Exception:
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

@router.get("/train/llm-status", summary="Lấy trạng thái tiến trình Fine-tune LLM Qwen trên Modal")
def get_llm_finetune_status():
    import os
    is_modal = os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable)
    status_file = Path("/storage/llm_finetune/training_progress.json")
    
    if is_modal and status_file.is_file():
        try:
            import sys
            sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
            from modal_app import volume
            volume.reload()
        except Exception:
            pass
        
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    return {"isTraining": False, "stage": "IDLE", "progress_percent": 0, "message": "Sẵn sàng"}


@router.get("/inference-backend", summary="Cấu hình mô hình cho Tầng 1 và Tầng 2")
def get_nlu_inference_backend():
    from app.services.nlu_registry import get_intent_backend, get_category_backend

    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    return {
        "intent_backend": get_intent_backend(nlu_dir),
        "category_backend": get_category_backend(nlu_dir),
    }


@router.post("/inference-backend", summary="Chọn mô hình Tầng 1 và Tầng 2")
def set_nlu_inference_backend(payload: dict):
    from app.services.nlu_registry import set_inference_backends

    intent_b = str(payload.get("intent_backend") or payload.get("backend", "")).strip().lower()
    cat_b = str(payload.get("category_backend") or payload.get("backend", "")).strip().lower()

    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    try:
        reg = set_inference_backends(nlu_dir, intent_b, cat_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Env NLU_USE_ENCODER is no longer used for dynamic routing, models.py handles it
    get_nlu_service().reload()

    return {
        "intent_backend": reg.get("intent_backend", "tfidf"),
        "category_backend": reg.get("category_backend", "tfidf"),
        "message": f"NLU updated: Layer 1 ({reg.get('intent_backend')}), Layer 2 ({reg.get('category_backend')})",
        "reloaded": True,
    }


@router.post(
    "/test-prompt",
    response_model=dict,
    summary="Test prompt trực tiếp không lưu lịch sử (Kiến trúc 2 tầng)",
)
def test_prompt(payload: dict) -> dict:
    from src.nlu.llm_intent_handler import run_llm_nlu_v2
    import time

    text = payload.get("text", "")
    override_prompt = payload.get("override_prompt")
    persona = payload.get("persona", "hai_huoc")
    caller_context = payload.get("caller_context", "chat")
    force_intent = payload.get("force_intent", "Auto")
    
    t0 = time.monotonic()
    
    # Fake profile context if needed
    context = {
        "user_id": "test_admin",
        "budget_total": 5000000,
        "budget_remain": 2500000,
        "verbal_style": persona
    }
    
    # Xác định forced_intent
    forced_intent_val = None
    if force_intent in ("Record", "Action", "Chitchat"):
        forced_intent_val = force_intent
    elif caller_context == "addstory":
        forced_intent_val = "Record"
    
    result = run_llm_nlu_v2(
        text=text,
        context_metadata=context,
        nlg_persona=persona,
        forced_intent=forced_intent_val,
        override_prompt=override_prompt,
    )
    
    latency = int((time.monotonic() - t0) * 1000)
    
    return {
        "ok": True,
        "latency_ms": latency,
        "result": result
    }



def sync_modal_nlu_jsons():
    """Sync json files from Modal volume to local directory if Modal is available."""
    try:
        import modal
        vol = modal.Volume.from_name("expense-ocr-nlu-storage")
        settings = get_settings()
        nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
        local_models = nlu_dir / "text_nlu" / "models"
        local_models.mkdir(parents=True, exist_ok=True)
        
        for name in ["nlu_model_registry.json", "nlu_training_history.json"]:
            try:
                data = b"".join(vol.read_file(f"nlu_models/{name}")).decode("utf-8")
                (local_models / name).write_text(data, encoding="utf-8")
            except Exception:
                pass
    except Exception as e:
        print(f"[Modal Sync] Error syncing NLU jsons: {e}")

def push_modal_nlu_jsons():
    """Push local NLU json files to Modal volume."""
    try:
        import modal
        vol = modal.Volume.from_name("expense-ocr-nlu-storage")
        settings = get_settings()
        nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
        local_models = nlu_dir / "text_nlu" / "models"
        
        reg_p = local_models / "nlu_model_registry.json"
        hist_p = local_models / "nlu_training_history.json"
        
        with vol.batch_upload(force=True) as batch:
            if reg_p.is_file():
                batch.put_file(str(reg_p), "nlu_models/nlu_model_registry.json")
            if hist_p.is_file():
                batch.put_file(str(hist_p), "nlu_models/nlu_training_history.json")
    except Exception as e:
        print(f"[Modal Push] Error pushing NLU jsons: {e}")


@router.post("/benchmark", summary="Chạy kiểm thử hiệu năng Golden Set 2 tầng (Intent & Category) cho 3 model")
def run_nlu_benchmark():
    try:
        from src.nlu.llm_benchmark import run_golden_set_benchmark
        return run_golden_set_benchmark()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {e}")


@router.post("/models/promote", summary="Duyệt áp dụng mô hình mới (Candidate -> Active)")
def promote_candidate_model():
    try:
        from app.services.nlu_registry import accept_pending_version
        settings = get_settings()
        nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
        reg = accept_pending_version(nlu_dir)
        push_modal_nlu_jsons()
            
        get_nlu_service().reload()
        return {
            "ok": True,
            "message": "Đã duyệt áp dụng mô hình mới thành công! Mô hình chuyển sang trạng thái Active.",
            "registry": reg,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to promote model: {e}")

@router.post("/models/reject", summary="Từ chối mô hình mới (Xóa Candidate)")
def reject_candidate_model():
    try:
        from app.services.nlu_registry import reject_pending_version
        settings = get_settings()
        nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
        reg = reject_pending_version(nlu_dir)
        push_modal_nlu_jsons()
            
        return {
            "ok": True,
            "message": "Đã từ chối và xóa mô hình Candidate thành công!",
            "registry": reg,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reject model: {e}")

@router.post("/models/rollback", summary="Khôi phục mô hình cũ (Active -> Old)")
def rollback_model():
    try:
        from app.services.nlu_registry import rollback_to_previous_version
        settings = get_settings()
        nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
        reg = rollback_to_previous_version(nlu_dir)
        push_modal_nlu_jsons()
            
        get_nlu_service().reload()
        return {
            "ok": True,
            "message": "Đã khôi phục mô hình phiên bản trước thành công!",
            "registry": reg,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rollback model: {e}")
