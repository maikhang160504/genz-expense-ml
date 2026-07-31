"""Bill OCR retrain: pre-label, export PICK, Kaggle plan, golden eval."""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from app.adapters import expense_ocr_nlu as adapter
from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
_LOCK = threading.Lock()


def _ocr_paths() -> Path:
    settings = get_settings()
    root = Path(settings.expense_ocr_nlu_dir).resolve()
    for sub in (root, root / "bill_ocr"):
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
    return root


def _verified_dir() -> Path:
    settings = get_settings()
    return Path(
        settings.verified_ocr_labels_dir
        or Path(settings.expense_ocr_nlu_dir) / "bill_ocr" / "exported"
    )


def ensure_ocr_loaded_for_prelabel() -> tuple[bool, str | None]:
    """Bill retrain always attempts real OCR (ignores USE_REAL_OCR gate on OCRService)."""
    if adapter.is_ocr_loaded():
        return True, None
    if not adapter.load_real_ocr_safe():
        err = adapter.get_ocr_error() or "Real OCR pipeline not loaded"
        hint = (
            "Set USE_REAL_OCR=true in ai-service/.env, verify OCR_WEIGHTS_PATH, "
            "then restart ai-service."
        )
        return False, f"{err}. {hint}"
    return True, None


def prelabel_image(image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
    """Auto-label from hybrid pipeline for WebAdmin review."""
    ok, err = ensure_ocr_loaded_for_prelabel()
    if not ok:
        return {
            "boxes": [],
            "kie_fields": {},
            "amount": None,
            "category": "Others",
            "kie_backend": "mock",
            "error": err,
            "ocr_loaded": False,
        }
    pipeline = adapter._OCR_PIPELINE
    suffix = Path(filename or "bill.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        image_rgb = pipeline._read_rgb(tmp_path)
        payload = pipeline.prelabel_for_admin(image_rgb)
        payload["backend"] = payload.get("auto_label_engine", "real-hybrid")
        payload["ocr_loaded"] = True
        kie_mod = importlib.import_module("receipt_ocr.pick_kie")
        layoutlmv3_path = getattr(pipeline, "layoutlmv3_model", None)
        payload["layoutlmv3_status"] = kie_mod.pick_kie_weights_status(layoutlmv3_path)
        if payload.get("kie_backend") != "layoutlmv3":
            payload.setdefault("warnings", []).append(
                "Auto-label đang dùng heuristic — deploy model_best.pth (LayoutLMv3) rồi bấm Tải lại model OCR."
            )
        return payload
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def export_verified(samples: list[dict[str, Any]]) -> dict[str, Any]:
    _ocr_paths()
    import shutil
    import json
    from datetime import datetime, timezone
    import urllib.request as urllib_request
    from urllib.parse import urljoin
    
    pick_export = importlib.import_module("receipt_ocr.pick_export")
    out_dir = _verified_dir()
    incremental = out_dir / "incremental"
    if incremental.exists():
        shutil.rmtree(incremental)
    incremental.mkdir(parents=True, exist_ok=True)

    export_result = pick_export.export_verified_samples(samples, incremental)

    images_dir = incremental / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, str]] = []
    copied = 0

    for sample in samples:
        sid = str(sample.get("id", "")).strip()
        src = sample.get("image_path") or sample.get("imagePath")
        src_path = Path(src) if src else None
        ext = sample.get("image_ext") or sample.get("imageExt") or (src_path.suffix if src_path else None) or ".jpg"
        dest_name = f"{sid}{ext if str(ext).startswith('.') else '.' + ext}"
        dest = images_dir / dest_name

        if src_path and src_path.is_file():
            shutil.copy2(src_path, dest)
            image_rows.append({"image": f"images/{dest_name}", "sample_id": sid, "split": "train"})
            copied += 1
        else:
            url = sample.get("image_url") or sample.get("imageUrl")
            if url:
                try:
                    if url.startswith("/"):
                        base_url = os.environ.get("BILL_RETRAIN_WEBHOOK_URL") or "http://127.0.0.1:4000"
                        url = urljoin(base_url, url)
                    urllib_request.urlretrieve(url, str(dest))
                    image_rows.append({"image": f"images/{dest_name}", "sample_id": sid, "split": "train"})
                    copied += 1
                except Exception as e:
                    logger.warning(f"Failed to download image for sample {sid} from {url}: {e}")

    image_list_path = incremental / "image_list.csv"
    lines = ["image,sample_id,split"]
    for row in image_rows:
        lines.append(f"{row['image']},{row['sample_id']},{row['split']}")
    image_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pack_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "incremental_samples": export_result["manifest"]["count"],
        "incremental_images": copied,
        "notes": "Exported verified samples locally for LayoutLMv3 training.",
    }
    pack_path = incremental / "training_pack.json"
    pack_path.write_text(json.dumps(pack_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        **export_result,
        "incremental_dir": str(incremental),
        "training_pack": pack_meta,
        "training_pack_path": str(pack_path),
        "images_copied": copied,
        "image_list_path": str(image_list_path),
    }


def kaggle_retrain_plan(job_type: str) -> dict[str, Any]:
    return {"ok": False, "error": "Kaggle integration is deprecated. Real-time GPU training is now done on Modal Cloud."}


def kaggle_username() -> str | None:
    return None


def trigger_kaggle_retrain(
    job_type: str,
    webhook_url: str | None = None,
    cloud_fallback_url: str | None = None,
    auto_after_export: bool = False,
) -> dict[str, Any]:
    return {"ok": False, "error": "Kaggle training has been moved to Modal Cloud. Please use the Modal trigger instead."}


def get_kaggle_job(job_id: str) -> dict[str, Any] | None:
    return None


def list_kaggle_jobs(limit: int = 20) -> list[dict[str, Any]]:
    return []


def deploy_artifact(source: str, job_type: str, batch_id: str | None = None) -> dict[str, Any]:
    _ocr_paths()
    deploy_mod = importlib.import_module("receipt_ocr.artifact_deploy")
    return deploy_mod.deploy_from_source(source, job_type, batch_id)


def sync_kaggle_kernel(
    job_type: str = "pick_retrain",
    *,
    skip_download: bool = False,
    download_dir: str | None = None,
) -> dict[str, Any]:
    return {"ok": False, "error": "Kaggle integration is deprecated."}


def kaggle_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """Record external webhook (e.g. manual Kaggle completion notify)."""
    logger.info("Bill retrain webhook (disabled): %s", payload)
    return {"ok": True, "received": payload}


def run_golden_eval() -> dict[str, Any]:
    _ocr_paths()
    golden_mod = importlib.import_module("receipt_ocr.golden_eval")
    nlu_mod = importlib.import_module("receipt_ocr.receipt_nlu")
    kie_mod = importlib.import_module("receipt_ocr.pick_kie")
    import pandas as pd

    fixtures_path = Path(get_settings().expense_ocr_nlu_dir) / "bill_ocr" / "receipt_ocr" / "tests" / "golden" / "fixtures.jsonl"
    fixtures = golden_mod.load_golden_fixtures(fixtures_path)
    results = []
    for fx in fixtures:
        lines = pd.DataFrame([{"line_text": ln, "bbox": [0, 0, 1, 1]} for ln in fx["lines"]])
        boxes = pd.DataFrame(fx["boxes"])
        kie = kie_mod.extract_kie_fields(fx["boxes"])
        pred = nlu_mod.extract_receipt_summary(lines, df_boxes=boxes, kie_fields=kie, split_mode=False)
        metrics = golden_mod.eval_summary_against_golden(pred, fx["expected"])
        results.append({"fixture_id": fx["fixture_id"], "metrics": metrics, "predicted": pred})
    return golden_mod.run_golden_eval(results)


def trigger_modal_layoutlmv3_train(num_epochs: int = 30, learning_rate: float = 2e-5) -> dict[str, Any]:
    """Trigger serverless LayoutLMv3 training on Modal Cloud."""
    try:
        import modal
        f = modal.Function.from_name("expense-ocr-nlu", "train_layoutlmv3_model")
        handle = f.spawn(num_epochs=num_epochs, learning_rate=learning_rate)
        return {
            "ok": True,
            "job_id": handle.object_id,
            "status": "RUNNING",
            "message": "LayoutLMv3 training triggered successfully on Modal Cloud."
        }
    except Exception as e:
        logger.error(f"Failed to trigger Modal training: {e}")
        return {"ok": False, "error": str(e)}
