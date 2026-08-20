"""WebAdmin bill retrain API endpoints.

Kaggle OCR retrain endpoints restored for PICK KIE flow.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.bill_retrain_service import (
    deploy_artifact,
    export_verified,
    get_kaggle_job,
    kaggle_retrain_plan,
    kaggle_webhook,
    list_kaggle_jobs,
    prelabel_image,
    run_golden_eval,
    sync_kaggle_kernel,
    trigger_kaggle_retrain,
    kaggle_username,
    trigger_modal_layoutlmv3_train,
)


router = APIRouter(prefix="/bill-retrain", tags=["bill-retrain"])

MAX_BYTES = 8 * 1024 * 1024


class VerifiedSample(BaseModel):
    id: str
    admin_labels: list[dict[str, Any]] = Field(default_factory=list)
    image_url: str | None = None
    image_path: str | None = None
    image_ext: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    samples: list[VerifiedSample]
    trigger_kaggle: bool = False
    kaggle_job_type: str = Field(default="layoutlmv3", pattern="^(pick_retrain|pick_train|layoutlmv3|layoutlmv3_train|layoutlmv3_retrain)$")
    webhook_url: str | None = None


class KagglePlanRequest(BaseModel):
    job_type: str = Field(default="layoutlmv3", pattern="^(pick_retrain|pick_train|layoutlmv3|layoutlmv3_train|layoutlmv3_retrain)$")


class KaggleTriggerRequest(BaseModel):
    job_type: str = Field(default="layoutlmv3", pattern="^(pick_retrain|pick_train|layoutlmv3|layoutlmv3_train|layoutlmv3_retrain)$")
    webhook_url: str | None = None
    cloud_fallback_url: str | None = None


class KaggleDeployRequest(BaseModel):
    source: str = Field(description="Local path, zip, or https URL to trained artifact")
    job_type: str = Field(default="layoutlmv3", pattern="^(pick_retrain|pick_train|layoutlmv3|layoutlmv3_train|layoutlmv3_retrain)$")
    batch_id: str | None = None


class KaggleSyncRequest(BaseModel):
    job_type: str = Field(default="layoutlmv3", pattern="^(pick_retrain|pick_train|layoutlmv3|layoutlmv3_train|layoutlmv3_retrain)$")
    skip_download: bool = False
    download_dir: str | None = None


@router.post("/prelabel")
async def bill_prelabel(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        from app.core.exceptions import InvalidInputError
        raise InvalidInputError("Image is larger than 8 MB.")
    return prelabel_image(raw, file.filename)


@router.post("/export-verified")
def bill_export_verified(body: ExportRequest) -> dict[str, Any]:
    samples = [s.model_dump() for s in body.samples]
    result = export_verified(samples, body.webhook_url)
    result["kaggle_username"] = kaggle_username()
    if body.trigger_kaggle:
        job = trigger_kaggle_retrain(
            body.kaggle_job_type,
            webhook_url=body.webhook_url,
            auto_after_export=True,
        )
        result["kaggle_job"] = job
    return result


@router.post("/kaggle/plan")
def bill_kaggle_plan(body: KagglePlanRequest) -> dict[str, Any]:
    return kaggle_retrain_plan(body.job_type)


@router.post("/kaggle/trigger")
def bill_kaggle_trigger(body: KaggleTriggerRequest) -> dict[str, Any]:
    return trigger_kaggle_retrain(body.job_type, body.webhook_url, body.cloud_fallback_url)


@router.get("/kaggle/jobs")
def bill_kaggle_jobs(limit: int = 20) -> list[dict[str, Any]]:
    return list_kaggle_jobs(limit)


@router.get("/kaggle/jobs/{job_id}")
def bill_kaggle_job_status(job_id: str) -> dict[str, Any]:
    job = get_kaggle_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.post("/kaggle/deploy")
def bill_kaggle_deploy(body: KaggleDeployRequest) -> dict[str, Any]:
    return deploy_artifact(body.source, body.job_type, body.batch_id)


@router.post("/kaggle/sync")
def bill_kaggle_sync(body: KaggleSyncRequest | None = None) -> dict[str, Any]:
    payload = body or KaggleSyncRequest()
    return sync_kaggle_kernel(
        payload.job_type,
        skip_download=payload.skip_download,
        download_dir=payload.download_dir,
    )


@router.post("/kaggle/webhook")
def bill_kaggle_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    return kaggle_webhook(payload)


@router.get("/golden-eval")
def bill_golden_eval() -> dict[str, Any]:
    return run_golden_eval()


class ModalTriggerRequest(BaseModel):
    num_epochs: int = Field(default=30, ge=1, le=100)
    learning_rate: float = Field(default=2e-5, gt=0.0)


@router.post("/modal/trigger")
def bill_modal_trigger(body: ModalTriggerRequest | None = None) -> dict[str, Any]:
    payload = body or ModalTriggerRequest()
    return trigger_modal_layoutlmv3_train(payload.num_epochs, payload.learning_rate)


@router.get("/ocr-history")
def get_ocr_history() -> list[dict[str, Any]]:
    data = read_modal_json("/storage/layoutlmv3/ocr_training_history.json")
    if data:
        return data
    return []

def read_modal_json(path_str: str) -> dict | None:
    from pathlib import Path
    import os
    import json
    import sys

    # Reload volume if running on Modal container so writes from worker containers are seen
    is_modal = os.environ.get("IS_MODAL") == "true" or os.environ.get("MODAL_PROJECT_NAME") or "modal" in str(sys.executable) or Path("/storage").is_dir()
    if is_modal:
        try:
            sys.path.insert(0, str(Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace"))))
            from modal_app import volume
            volume.reload()
        except Exception:
            pass

    p = Path(path_str)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # Check local workspace relative paths if running locally on Windows/Host
    base_dir = Path(__file__).resolve().parents[4]
    local_candidates = [
        base_dir / p.name,
        base_dir / "storage" / path_str.replace("/storage/", ""),
        Path(os.environ.get("EXPENSE_OCR_NLU_DIR", ".")) / p.name,
        Path(os.environ.get("EXPENSE_OCR_NLU_DIR", ".")) / "storage" / path_str.replace("/storage/", "")
    ]
    for cand in local_candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    
    # Fallback to Modal Volume SDK if running locally
    try:
        import modal
        vol = modal.Volume.from_name("expense-ocr-nlu-storage")
        vol_path = path_str.replace("/storage/", "")
        chunks = []
        for chunk in vol.read_file(vol_path):
            chunks.append(chunk)
        data = b"".join(chunks).decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print(f"Error reading from modal volume {path_str}: {e}")
        return None

@router.get("/modal/progress")
def get_modal_progress() -> dict[str, Any]:
    data = read_modal_json("/storage/layoutlmv3/training_progress.json")
    if data:
        return data
    return {"isTraining": False}


@router.get("/modal/log")
def get_modal_log() -> dict[str, Any]:
    # Read the text log from Modal volume
    path_str = "/storage/layoutlmv3_train_log.txt"
    from pathlib import Path
    p = Path(path_str)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return {"log": "".join(lines[-500:])}
        except Exception:
            return {"log": ""}
            
    # Fallback to Modal SDK
    try:
        import modal
        vol = modal.Volume.from_name("expense-ocr-nlu-storage")
        vol_path = path_str.replace("/storage/", "")
        chunks = []
        for chunk in vol.read_file(vol_path):
            chunks.append(chunk)
        data = b"".join(chunks).decode("utf-8")
        lines = data.split("\n")
        return {"log": "\n".join(lines[-500:])}
    except Exception as e:
        return {"log": f"Log not found or error reading log: {e}"}



@router.get("/model/candidate")
def get_model_candidate() -> dict[str, Any]:
    history = read_modal_json("/storage/layoutlmv3/ocr_training_history.json")
    if not history:
        return {"candidate": None, "current": None}
        
    current = None
    candidate = None
    for run in reversed(history):
        if run.get("is_candidate"):
            if not candidate:
                candidate = run
        else:
            if not current and run.get("status") == "success":
                current = run
                
    return {"candidate": candidate, "current": current}


@router.post("/model/promote")
def promote_model() -> dict[str, Any]:
    try:
        from pathlib import Path
        import shutil
        import json
        
        # 1. Direct local volume manipulation if /storage is mounted
        candidate_path = Path("/storage/layoutlmv3/candidate_model.pth")
        if candidate_path.parent.exists():
            best_path = Path("/storage/layoutlmv3/model_best.pth")
            previous_path = Path("/storage/layoutlmv3/model_previous.pth")
            if not candidate_path.is_file():
                return {"ok": False, "error": "No candidate model found."}
            if best_path.is_file():
                shutil.copy2(best_path, previous_path)
            shutil.copy2(candidate_path, best_path)
            
            history_file = Path("/storage/layoutlmv3/ocr_training_history.json")
            if history_file.is_file():
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                    for run in reversed(history):
                        if run.get("is_candidate"):
                            run["is_candidate"] = False
                            run["status"] = "success"
                            break
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump(history, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error updating history: {e}")
            
            try:
                from modal_app import volume
                volume.commit()
            except Exception:
                pass
            return {"ok": True, "message": "Đã duyệt áp dụng mô hình LayoutLMv3 mới thành công."}

        # 2. Remote Modal execution fallback (synchronous remote call)
        import modal
        f = modal.Function.from_name("expense-ocr-nlu", "promote_layoutlmv3_model")
        res = f.remote()
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.post("/model/reject")
def reject_model() -> dict[str, Any]:
    try:
        from pathlib import Path
        import json
        
        # 1. Direct local volume manipulation if /storage is mounted
        candidate_path = Path("/storage/layoutlmv3/candidate_model.pth")
        if candidate_path.parent.exists():
            if candidate_path.is_file():
                candidate_path.unlink()
            history_file = Path("/storage/layoutlmv3/ocr_training_history.json")
            if history_file.is_file():
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                    history = [run for run in history if not run.get("is_candidate")]
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump(history, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error updating history: {e}")
            try:
                from modal_app import volume
                volume.commit()
            except Exception:
                pass
            return {"ok": True, "message": "Đã từ chối và xóa mô hình candidate thành công."}

        # 2. Remote Modal execution fallback (synchronous remote call)
        import modal
        try:
            from modal_app import reject_layoutlmv3_model
            res = reject_layoutlmv3_model.remote()
        except Exception:
            f = modal.Function.from_name("expense-ocr-nlu", "reject_layoutlmv3_model")
            res = f.remote()
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/model/rollback")
def rollback_model() -> dict[str, Any]:
    try:
        from pathlib import Path
        import shutil
        import json
        
        # 1. Direct local volume manipulation if /storage is mounted
        previous_path = Path("/storage/layoutlmv3/model_previous.pth")
        if previous_path.parent.exists():
            best_path = Path("/storage/layoutlmv3/model_best.pth")
            if not previous_path.is_file():
                return {"ok": False, "error": "Không tìm thấy bản sao lưu mô hình trước đó."}
            shutil.copy2(previous_path, best_path)
            try:
                from modal_app import volume
                volume.commit()
            except Exception:
                pass
            return {"ok": True, "message": "Đã khôi phục mô hình phiên bản trước thành công."}

        # 2. Remote Modal execution fallback (synchronous remote call)
        import modal
        f = modal.Function.from_name("expense-ocr-nlu", "rollback_layoutlmv3_model")
        res = f.remote()
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/model/sync-workspace")
def sync_workspace_model() -> dict[str, Any]:
    import os
    from pathlib import Path
    try:
        import modal
        vol = modal.Volume.lookup("expense-ocr-nlu-storage")
        
        # Ensure local dir exists
        local_dir = Path(os.environ.get("EXPENSE_OCR_NLU_DIR", "/workspace")) / "bill_ocr" / "models" / "layoutlmv3"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "model_best.pth"
        
        # Download file
        chunks = []
        for chunk in vol.read_file("layoutlmv3/model_best.pth"):
            chunks.append(chunk)
            
        with open(local_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
                
        return {"ok": True, "message": f"Synchronized model_best.pth successfully to {local_path}."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

