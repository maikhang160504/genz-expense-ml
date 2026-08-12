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
    kaggle_job_type: str = Field(default="pick_retrain", pattern="^(pick_retrain|pick_train)$")
    webhook_url: str | None = None


class KagglePlanRequest(BaseModel):
    job_type: str = Field(default="pick_retrain", pattern="^(pick_retrain|pick_train)$")


class KaggleTriggerRequest(BaseModel):
    job_type: str = Field(default="pick_retrain", pattern="^(pick_retrain|pick_train)$")
    webhook_url: str | None = None
    cloud_fallback_url: str | None = None


class KaggleDeployRequest(BaseModel):
    source: str = Field(description="Local path, zip, or https URL to trained artifact")
    job_type: str = Field(default="pick_retrain", pattern="^(pick_retrain|pick_train)$")
    batch_id: str | None = None


class KaggleSyncRequest(BaseModel):
    job_type: str = Field(default="pick_retrain", pattern="^(pick_retrain|pick_train)$")
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

def read_modal_json(path_str: str) -> dict:
    from pathlib import Path
    import json
    p = Path(path_str)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    
    # Fallback to Modal Volume SDK if running locally
    try:
        import modal
        vol = modal.Volume.lookup("expense-ocr-nlu-storage")
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
        import modal
        f = modal.Function.from_name("expense-ocr-nlu", "promote_layoutlmv3_model")
        res = f.spawn()
        return {"ok": True, "message": "Triggered model promotion successfully. (May require container restart)", "job_id": res.object_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/model/rollback")
def rollback_model() -> dict[str, Any]:
    try:
        import modal
        f = modal.Function.from_name("expense-ocr-nlu", "rollback_layoutlmv3_model")
        res = f.spawn()
        return {"ok": True, "message": "Triggered model rollback successfully. (May require container restart)", "job_id": res.object_id}
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

