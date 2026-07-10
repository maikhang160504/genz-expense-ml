"""Adapter that wraps the original `expense-ocr-nlu` repo as a lazy-loaded backend.

We don't want the FastAPI service to crash on startup if those big artifacts
(PaddleOCR, VietOCR, PhoBERT, joblib bundles) are missing or fail to import.
Hence the adapter:

* keeps the heavy import inside `try/except`,
* exposes booleans `is_nlu_loaded()` / `is_ocr_loaded()`,
* falls back to the lightweight mock when the real pipeline cannot be loaded.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
_LOCK = threading.Lock()

_NLU_BUNDLE: dict[str, Any] | None = None
_NLU_ERROR: str | None = None

_OCR_PIPELINE: Any = None
_OCR_ERROR: str | None = None
_OCR_RELOADING = False


def _ensure_paths_on_sys_path() -> Path:
    """Add expense-ocr-nlu sub-paths to sys.path so its modules import correctly."""
    settings = get_settings()
    root = Path(settings.expense_ocr_nlu_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"expense-ocr-nlu directory not found: {root}")
    candidates = [root, root / "bill_ocr", root / "text_nlu"]
    for path in candidates:
        spath = str(path)
        if spath not in sys.path:
            sys.path.insert(0, spath)
    return root


def _load_nlu_bundle_unlocked() -> dict[str, Any]:
    """Heavy import; only call inside lock."""
    global _NLU_BUNDLE, _NLU_ERROR

    if _NLU_BUNDLE is not None:
        return _NLU_BUNDLE

    _ensure_paths_on_sys_path()

    try:
        env_module = importlib.import_module("src.config.env")
        settings_module = importlib.import_module("src.config.settings")
        models_module = importlib.import_module("src.nlu.models")
        ner_module = importlib.import_module("src.nlu.ner")
        pipeline_module = importlib.import_module("src.nlu.pipeline")
        llm_runner = importlib.import_module("src.nlg.llm_runner")
        context_meta = importlib.import_module("src.nlg.context_meta")
        json_sanitize = importlib.import_module("src.nlu.json_sanitize")
        action_executor = importlib.import_module("src.nlu.action_executor")
    except Exception as exc:  # pragma: no cover - depends on local env
        _NLU_ERROR = f"NLU import failed: {exc}"
        logger.warning(_NLU_ERROR)
        raise

    env_module.load_env_file(settings_module.ENV_PATH)

    bundle = {
        "intent": models_module.load_intent_model(),
        "category": models_module.load_category_model(),
        "action_type": models_module.load_action_type_model(),
        "action_slots": models_module.load_action_slots_model(),
        "record_type": models_module.load_record_type_model(),
        "sentiment": models_module.load_chitchat_sentiment_model(),
        "ner": ner_module.load_ner_model(settings_module.NER_MODEL_DIR),
        "prompts": llm_runner.load_prompts(settings_module.PROMPTS_PATH),
        "request_template": llm_runner.load_request_template(settings_module.REQUEST_TEMPLATE_PATH),
        "pipeline_module": pipeline_module,
        "llm_runner": llm_runner,
        "context_meta": context_meta,
        "json_sanitize": json_sanitize,
        "action_executor": action_executor,
    }
    _NLU_BUNDLE = bundle
    logger.info("Loaded real NLU bundle from expense-ocr-nlu.")

    # Auto-warmup/pre-load Hugging Face encoder models on startup to avoid lazy-loading on first request
    try:
        encoder_runtime = importlib.import_module("src.nlu.encoder_runtime")
        for key in ("intent", "category", "action_type", "record_type"):
            model_info = bundle.get(key)
            if isinstance(model_info, dict) and model_info.get("backend") == "encoder":
                inner_bundle = model_info.get("bundle")
                if isinstance(inner_bundle, dict) and "encoder_model_name" in inner_bundle:
                    model_name = inner_bundle["encoder_model_name"]
                    logger.info(f"Pre-loading/Warming up Hugging Face encoder model '{model_name}' for {key}...")
                    encoder_runtime._get_hf(model_name)
    except Exception as exc:
        logger.warning(f"Failed to pre-load Hugging Face encoder models during startup: {exc}")

    # Pre-load Qwen local LLM if active backend is LLM and local loading is enabled (disabled in Modal to use RPC GPU container)
    try:
        if models_module.get_inference_backend() == "llm" and os.environ.get("USE_LOCAL_PHOGPT") == "1" and os.environ.get("IS_MODAL") != "true":
            logger.info("NLU inference backend is LLM. Pre-loading local Qwen model into GPU...")
            local_llm = importlib.import_module("src.nlu.local_llm")
            local_llm.load_local_phogpt()
    except Exception as exc:
        logger.warning(f"Failed to pre-load Qwen LLM during startup: {exc}")

    return bundle


def load_real_nlu_safe() -> bool:
    """Try to load NLU once; return True on success."""
    global _NLU_ERROR
    if _NLU_BUNDLE is not None:
        return True
    with _LOCK:
        if _NLU_BUNDLE is not None:
            return True
        try:
            _load_nlu_bundle_unlocked()
        except Exception as exc:  # noqa: BLE001
            _NLU_ERROR = str(exc)
            return False
    return True


def reload_nlu() -> bool:
    """Reload the NLU model registry configuration and models."""
    global _NLU_BUNDLE, _NLU_ERROR
    try:
        import os
        import importlib
        from app.core.config import get_settings
        
        settings = get_settings()
        _ensure_paths_on_sys_path()
        
        models_module = importlib.import_module("src.nlu.models")
        ner_module = importlib.import_module("src.nlu.ner")
        pipeline_module = importlib.import_module("src.nlu.pipeline")
        llm_runner = importlib.import_module("src.nlg.llm_runner")
        context_meta = importlib.import_module("src.nlg.context_meta")
        json_sanitize = importlib.import_module("src.nlu.json_sanitize")
        action_executor = importlib.import_module("src.nlu.action_executor")
        
        # Reload models from disk/volume
        bundle = {
            "intent": models_module.load_intent_model(),
            "category": models_module.load_category_model(),
            "action_type": models_module.load_action_type_model(),
            "action_slots": models_module.load_action_slots_model(),
            "record_type": models_module.load_record_type_model(),
            "sentiment": models_module.load_chitchat_sentiment_model(),
            "ner": ner_module.load_ner_model(settings.expense_ocr_nlu_dir / "text_nlu" / "models" / "ner_model"),
            "prompts": llm_runner.load_prompts(settings.expense_ocr_nlu_dir / "src" / "prompts" / "prompts.json"),
            "request_template": llm_runner.load_request_template(settings.expense_ocr_nlu_dir / "src" / "prompts" / "request_template.json"),
            "pipeline_module": pipeline_module,
            "llm_runner": llm_runner,
            "context_meta": context_meta,
            "json_sanitize": json_sanitize,
            "action_executor": action_executor,
        }
        
        # Warmup encoder models on reload
        try:
            encoder_runtime = importlib.import_module("src.nlu.encoder_runtime")
            for key in ("intent", "category", "action_type", "record_type"):
                model_info = bundle.get(key)
                if isinstance(model_info, dict) and model_info.get("backend") == "encoder":
                    inner_bundle = model_info.get("bundle")
                    if isinstance(inner_bundle, dict) and "encoder_model_name" in inner_bundle:
                        model_name = inner_bundle["encoder_model_name"]
                        logger.info(f"Pre-loading/Warming up Hugging Face encoder model '{model_name}' for {key}...")
                        encoder_runtime._get_hf(model_name)
        except Exception as exc:
            logger.warning(f"Failed to pre-load Hugging Face encoder models during reload: {exc}")

        # Pre-load Qwen local LLM if active backend is LLM and local loading is enabled (disabled in Modal)
        try:
            if models_module.get_inference_backend() == "llm" and os.environ.get("USE_LOCAL_PHOGPT") == "1" and os.environ.get("IS_MODAL") != "true":
                logger.info("NLU inference backend is LLM. Pre-loading local Qwen model into GPU...")
                local_llm = importlib.import_module("src.nlu.local_llm")
                local_llm.load_local_phogpt()
        except Exception as exc:
            logger.warning(f"Failed to pre-load Qwen LLM during reload: {exc}")

        with _LOCK:
            _NLU_BUNDLE = bundle
            _NLU_ERROR = None
        return True
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _NLU_ERROR = str(exc)
        return False


def is_nlu_loaded() -> bool:
    return _NLU_BUNDLE is not None


def get_nlu_error() -> str | None:
    return _NLU_ERROR


def run_real_nlu(
    text: str,
    profile: dict[str, Any] | None = None,
    run_llm: bool = False,
    nlg_persona: str | None = None,
    emotion: str | None = None,
    user_id: str | None = None,
    user_corrections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call the real NLU pipeline + optional Gemini NLG layer."""
    bundle = _NLU_BUNDLE
    if bundle is None:
        raise RuntimeError("real NLU bundle not loaded")

    pipeline_module = bundle["pipeline_module"]
    llm_runner = bundle["llm_runner"]
    context_meta = bundle["context_meta"]
    json_sanitize_mod = bundle["json_sanitize"]
    action_executor = bundle["action_executor"]

    result = pipeline_module.run_nlu(
        text,
        bundle["intent"],
        bundle["category"],
        bundle["action_type"],
        bundle["record_type"],
        bundle["sentiment"],
        bundle["ner"],
        bundle.get("action_slots"),
        run_llm=run_llm,
        user_id=user_id,
        user_corrections=user_corrections,
        profile=profile,
    )

    if result.get("intent") == "Action":
        try:
            result["demo_execution_lines"] = action_executor.describe_action_execution(result)
        except Exception:  # noqa: BLE001
            result["demo_execution_lines"] = []

    record_type = result.get("record_type") if result.get("intent") == "Record" else None
    nlu_for_meta = {
        "intent": result.get("intent"),
        "text": text,
        "item": result.get("item"),
        "category": result.get("category"),
        "amount": (result.get("amount_spent") or result.get("amount"))
        if result.get("intent") == "Record"
        else result.get("action_param"),
        "record_type": record_type,
        "is_expense": record_type == "Expense" if record_type else None,
        "income_type": result.get("income_type") if result.get("intent") == "Record" else None,
        "action_type": result.get("action_type"),
        "value": result.get("action_param"),
    }
    context_metadata = context_meta.build_context_metadata(nlu_for_meta, profile or {})

    # Skip LLM call in the first pass for Action intents, since the backend will
    # enrich it with action_facts and make a second call to generate the final response.
    is_action_first_pass = (result.get("intent") == "Action") and (not profile or "action_facts" not in profile)
    
    # Skip LLM call if already processed in a single pass by Qwen LLM or active backend is LLM
    from src.nlu.models import get_inference_backend
    is_already_unified = (result.get("backend") == "llm_unified") or (get_inference_backend() == "llm")

    if (run_llm or result.get("intent") == "Chitchat") and not is_action_first_pass and not is_already_unified:
        try:
            llm_runner.attach_nlg_and_llm(
                result,
                user_text=text,
                nlu_result=nlu_for_meta,
                context_metadata=context_metadata,
                prompts_config=bundle["prompts"],
                request_template=bundle["request_template"],
                nlg_persona=(nlg_persona or emotion or "hai_huoc"),
                run_llm=run_llm,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM enrichment failed: %s", exc)
            result["llm_error"] = str(exc)

    result["debug_info"] = {
        "is_already_unified": is_already_unified,
        "backend": result.get("backend"),
        "raw_keys": list(result.keys()),
        "run_llm": run_llm,
        "is_action_first_pass": is_action_first_pass,
    }

    return json_sanitize_mod.json_sanitize(result)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _sync_layoutlmv3_weights_from_storage():
    import shutil
    from pathlib import Path
    volume_layoutlmv3 = Path("/storage/layoutlmv3/model_best.pth")
    if volume_layoutlmv3.is_file():
        dest_layoutlmv3 = Path("/workspace/bill_ocr/models/layoutlmv3/model_best.pth")
        dest_layoutlmv3.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dest_layoutlmv3.is_file() or volume_layoutlmv3.stat().st_mtime > dest_layoutlmv3.stat().st_mtime:
                shutil.copy2(volume_layoutlmv3, dest_layoutlmv3)
                logger.info("[ADAPTER] Successfully synchronized newest LayoutLMv3 weights from /storage to /workspace.")
        except Exception as e:
            logger.warning("[ADAPTER] Failed to copy LayoutLMv3 weights: %s", e)


def _load_ocr_pipeline_unlocked() -> Any:
    global _OCR_PIPELINE, _OCR_ERROR
    if _OCR_PIPELINE is not None:
        return _OCR_PIPELINE

    root = _ensure_paths_on_sys_path()
    settings = get_settings()
    
    # Sync weights from storage first
    _sync_layoutlmv3_weights_from_storage()
    
    from receipt_ocr.model_paths import LAYOUTLMV3_MODEL_PATH, resolve_vietocr_weights_path

    weights_path = resolve_vietocr_weights_path(settings.ocr_weights_path or None)
    if not weights_path.is_file():
        _OCR_ERROR = (
            f"VietOCR weights missing at {weights_path}. "
            f"Expected: bill_ocr/models/vietocr/vgg_transformer.pth — "
            f"fix OCR_WEIGHTS_PATH in ai-service/.env and restart."
        )
        raise FileNotFoundError(_OCR_ERROR)

    try:
        pipeline_module = importlib.import_module("receipt_ocr.hybrid_pipeline")
    except Exception as exc:
        _OCR_ERROR = f"MC-OCR pipeline import failed: {exc}"
        raise

    import receipt_ocr.pick_kie as kie_mod
    kie_mod.reset_kie_engine()

    layoutlmv3_model = getattr(settings, "layoutlmv3_model_path", None) or LAYOUTLMV3_MODEL_PATH
    rot_path = getattr(settings, "rotation_model_path", None) or None
    use_rotation = getattr(settings, "use_rotation_corrector", True)
    _OCR_PIPELINE = pipeline_module.HybridReceiptOCRPipeline(
        vietocr_weights=weights_path,
        device=settings.device,
        layoutlmv3_model=layoutlmv3_model,
        rotation_weights=rot_path or None,
        use_rotation=use_rotation,
        paddle_use_gpu=(settings.device == "cuda"),
    ).load()
    rot_mod = importlib.import_module("receipt_ocr.rotation_corrector")
    rot_status = rot_mod.rotation_weights_status(rot_path)
    logger.info(
        "Loaded MC-OCR pipeline (Paddle+Rotation+VietOCR+LayoutLMv3). "
        "LayoutLMv3 ready=%s rotation ready=%s",
        (_OCR_PIPELINE._get_kie().backend == "layoutlmv3"),
        rot_status.get("ready"),
    )
    return _OCR_PIPELINE


def load_real_ocr_safe() -> bool:
    global _OCR_ERROR
    if _OCR_PIPELINE is not None:
        return True
    with _LOCK:
        if _OCR_PIPELINE is not None:
            return True
        try:
            _load_ocr_pipeline_unlocked()
        except Exception as exc:  # noqa: BLE001
            _OCR_ERROR = str(exc)
            return False
    return True


def reload_ocr() -> bool:
    """Force reload MC-OCR pipeline (VietOCR + PICK KIE) from disk with zero downtime."""
    global _OCR_PIPELINE, _OCR_ERROR, _OCR_RELOADING
    with _LOCK:
        if _OCR_RELOADING:
            logger.info("OCR reload already in progress — skipping duplicate request.")
            return _OCR_PIPELINE is not None
        _OCR_RELOADING = True
    try:
        root = _ensure_paths_on_sys_path()
        settings = get_settings()
        
        # Sync weights from storage first
        _sync_layoutlmv3_weights_from_storage()
        
        from receipt_ocr.model_paths import LAYOUTLMV3_MODEL_PATH, resolve_vietocr_weights_path

        weights_path = resolve_vietocr_weights_path(settings.ocr_weights_path or None)
        if not weights_path.is_file():
            _OCR_ERROR = (
                f"VietOCR weights missing at {weights_path}. "
                f"Expected: bill_ocr/models/vietocr/vgg_transformer.pth — "
                f"fix OCR_WEIGHTS_PATH in ai-service/.env and restart."
            )
            raise FileNotFoundError(_OCR_ERROR)

        pipeline_module = importlib.import_module("receipt_ocr.hybrid_pipeline")
        import receipt_ocr.pick_kie as kie_mod
        kie_mod.reset_kie_engine()

        layoutlmv3_model = getattr(settings, "layoutlmv3_model_path", None) or LAYOUTLMV3_MODEL_PATH
        rot_path = getattr(settings, "rotation_model_path", None) or None
        use_rotation = getattr(settings, "use_rotation_corrector", True)

        new_pipeline = pipeline_module.HybridReceiptOCRPipeline(
            vietocr_weights=weights_path,
            device=settings.device,
            layoutlmv3_model=layoutlmv3_model,
            rotation_weights=rot_path or None,
            use_rotation=use_rotation,
            paddle_use_gpu=(settings.device == "cuda"),
        ).load()

        with _LOCK:
            _OCR_PIPELINE = new_pipeline
            _OCR_ERROR = None
        return True
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _OCR_ERROR = str(exc)
        return False
    finally:
        with _LOCK:
            _OCR_RELOADING = False


def is_ocr_loaded() -> bool:
    return _OCR_PIPELINE is not None


def get_ocr_error() -> str | None:
    return _OCR_ERROR


def run_real_ocr(image_bytes: bytes, filename_hint: str | None = None) -> dict[str, Any]:
    pipeline = _OCR_PIPELINE
    if pipeline is None:
        raise RuntimeError("real OCR pipeline not loaded")

    from receipt_ocr.pipeline import ReceiptOCRPipeline

    image_rgb = ReceiptOCRPipeline.decode_rgb_bytes(image_bytes)
    result = pipeline.process_image_rgb(image_rgb, split_mode=False)
    summary = result
    df_lines_data = result.get("lines", [])
    labeled_boxes = result.get("boxes", [])

    lines_data = []
    raw_text_list = []
    for item in df_lines_data:
        t = str(item.get("text", "")).strip()
        if t:
            raw_text_list.append(t)
            lines_data.append({
                "text": t,
                "bbox": item.get("bbox"),
                "confidence": 0.90,
            })

    joined_text = "\n".join(raw_text_list)

    product_names: list[str] = []
    personalization_keyword: str | None = None
    try:
        from receipt_ocr.receipt_nlu import extract_product_names_for_category

        product_names = extract_product_names_for_category(raw_text_list)
        if product_names:
            personalization_keyword = product_names[0]
    except Exception:
        pass

    if not personalization_keyword:
        seller = (summary.get("kie_fields") or {}).get("SELLER")
        if seller:
            personalization_keyword = str(seller).strip()

    return {
        "text": joined_text,
        "lines": lines_data,
        "boxes": labeled_boxes,
        "kie_fields": summary.get("kie_fields", {}),
        "kie_backend": summary.get("kie_backend", "heuristic"),
        "suggestion": {
            "amount": summary.get("amount"),
            "category": summary.get("category"),
            "confidence": 0.85 if summary.get("amount") else 0.3,
            "currency": summary.get("currency", "VND"),
        },
        "requires_confirmation": True,
        "backend": "real-hybrid",
        "warnings": summary.get("warnings", []),
        "items_count": summary.get("items_count", 0),
        "product_names": product_names,
        "personalization_keyword": personalization_keyword,
    }


def pre_import_real_backends() -> None:
    """Import heavy modules in the main thread to avoid thread/import lock deadlocks."""
    settings = get_settings()
    if settings.use_real_nlu:
        try:
            _ensure_paths_on_sys_path()
            importlib.import_module("src.config.env")
            importlib.import_module("src.config.settings")
            importlib.import_module("src.nlu.models")
            importlib.import_module("src.nlu.ner")
            importlib.import_module("src.nlu.pipeline")
            importlib.import_module("src.nlg.llm_runner")
            importlib.import_module("src.nlg.context_meta")
            importlib.import_module("src.nlu.json_sanitize")
            importlib.import_module("src.nlu.action_executor")
            logger.info("Main thread pre-import of real NLU modules completed.")
        except Exception as exc:
            logger.warning("Real NLU pre-import failed (will retry at load): %s", exc)

    if settings.use_real_ocr:
        try:
            _ensure_paths_on_sys_path()
            importlib.import_module("receipt_ocr.pipeline")
            logger.info("Main thread pre-import of real OCR modules completed.")
        except Exception as exc:
            logger.warning("Real OCR pre-import failed (will retry at load): %s", exc)

