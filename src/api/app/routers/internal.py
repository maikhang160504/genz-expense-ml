"""Internal ops: reload models without restarting uvicorn."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.adapters import expense_ocr_nlu as adapter
from app.services.nlu_service import get_nlu_service
from app.services.ocr_service import get_ocr_service


router = APIRouter(prefix="/internal", tags=["internal"])


class ReloadModelsRequest(BaseModel):
    scope: Literal["all", "ocr", "nlu"] = Field(default="all")


@router.post("/reload-models", summary="Hot-reload NLU/OCR models from disk")
def reload_models(body: ReloadModelsRequest | None = None) -> dict:
    scope = (body.scope if body else "all")
    result: dict = {"scope": scope, "ok": True}

    if scope in ("all", "nlu"):
        from app.core.config import get_settings
        from app.services.nlu_registry import accept_pending_version
        from pathlib import Path

        nlu_root = Path(get_settings().expense_ocr_nlu_dir).resolve()
        reg = accept_pending_version(nlu_root)
        nlu_ok = get_nlu_service().reload()
        result["nlu_loaded"] = nlu_ok
        result["nlu_error"] = adapter.get_nlu_error()
        result["nlu_version"] = reg.get("version")

    if scope in ("all", "ocr"):
        import receipt_ocr.pick_kie as kie_mod
        kie_mod.reset_kie_engine()
        ocr_ok = get_ocr_service().reload()
        result["ocr_loaded"] = ocr_ok
        result["ocr_error"] = adapter.get_ocr_error()
        if ocr_ok:
            pipeline = adapter._OCR_PIPELINE
            pick_path = getattr(pipeline, "pick_kie_model", None)
            result["pick_kie_status"] = kie_mod.pick_kie_weights_status(pick_path)
            result["kie_backend"] = getattr(getattr(pipeline, "_kie", None), "backend", None)
            if result["kie_backend"] is None and pipeline:
                result["kie_backend"] = pipeline._get_kie().backend

    if scope == "all":
        result["ok"] = bool(result.get("nlu_loaded")) and bool(result.get("ocr_loaded"))
    elif scope == "ocr":
        result["ok"] = bool(result.get("ocr_loaded"))
    else:
        result["ok"] = bool(result.get("nlu_loaded"))

    return result
