"""Load NLU classifiers for inference.

Production (default): TF-IDF joblib only — Kaggle/local retrain pipeline.
Encoder (PhoBERT): experimental comparison track; never auto-selected by file mtime.
Enable encoder inference only with env ``NLU_USE_ENCODER=1`` (benchmark / A-B).
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import joblib

from src.config import settings

_PATCHED_TOKENIZER = False


def get_intent_backend() -> str:
    from src.api.app.services.nlu_registry import get_intent_backend as _get
    from src.config import settings
    return _get(settings.TEXT_NLU_DIR.parent)

def get_category_backend() -> str:
    from src.api.app.services.nlu_registry import get_category_backend as _get
    from src.config import settings
    return _get(settings.TEXT_NLU_DIR.parent)

def use_intent_encoder_runtime() -> bool:
    env = os.environ.get("NLU_USE_ENCODER", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return get_intent_backend() == "encoder"

def use_category_encoder_runtime() -> bool:
    env = os.environ.get("NLU_USE_ENCODER", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return get_category_backend() == "encoder"


def _ensure_pickled_tokenizer() -> None:
    """Joblib TF-IDF tham chiếu ``text_preprocessing.vi_tokenize`` — tránh trùng PyPI."""
    global _PATCHED_TOKENIZER
    if _PATCHED_TOKENIZER:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*align should be passed as Python or NumPy boolean.*",
    )

    root = settings.TEXT_NLU_DIR
    pipeline_dir = root / "pipeline"
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import text_preprocessing as tp_pkg

    from pipeline.text_preprocessing import vi_tokenize

    tp_pkg.vi_tokenize = vi_tokenize
    _PATCHED_TOKENIZER = True


def _load_tfidf(path) -> dict:
    _ensure_pickled_tokenizer()
    payload = joblib.load(path)
    return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}


def _load_encoder(path) -> dict:
    return {"backend": "encoder", "bundle": joblib.load(path)}


def _get_path(default_path: Path) -> Path:
    from pathlib import Path
    storage_path = Path("/storage/nlu_models") / default_path.name
    if storage_path.exists() and storage_path.is_file():
        return storage_path
    return default_path


def load_intent_model():
    """Production: TF-IDF only. Encoder only if ``NLU_USE_ENCODER=1``. LLM backend skips local models."""
    if get_intent_backend() == "llm":
        return {"backend": "llm"}
    intent_enc = _get_path(settings.INTENT_ENCODER_PATH)
    if use_intent_encoder_runtime() and intent_enc.is_file():
        return _load_encoder(intent_enc)
    intent_model = _get_path(settings.MODEL_PATH)
    if intent_model.is_file():
        return _load_tfidf(intent_model)
    return {"backend": "missing"}


def load_category_model():
    if get_category_backend() == "llm":
        return {"backend": "llm"}
    cat_enc = _get_path(settings.CATEGORY_ENCODER_PATH)
    if use_category_encoder_runtime() and cat_enc.is_file():
        return _load_encoder(cat_enc)
    cat_model = _get_path(settings.CATEGORY_MODEL_PATH)
    if cat_model.is_file():
        return _load_tfidf(cat_model)
    return {"backend": "missing"}


def load_action_type_model():
    if get_intent_backend() == "llm":
        return {"backend": "llm"}
    act_enc = _get_path(settings.ACTION_TYPE_ENCODER_PATH)
    if use_intent_encoder_runtime() and act_enc.is_file():
        return _load_encoder(act_enc)
    act_model = _get_path(settings.ACTION_TYPE_MODEL_PATH)
    if act_model.is_file():
        return _load_tfidf(act_model)
    return {"backend": "missing"}


def load_action_slots_model():
    if get_intent_backend() == "llm":
        return {"backend": "llm"}
    from src.nlu.action_slots import load_action_slots_model as _load

    slots_model = _get_path(settings.ACTION_SLOTS_MODEL_PATH)
    bundle = _load(slots_model)
    if bundle is None:
        return {"backend": "missing"}
    return {"backend": "slots", "bundle": bundle}


def load_record_type_model():
    if get_intent_backend() == "llm":
        return {"backend": "llm"}
    rec_enc = _get_path(settings.RECORD_TYPE_ENCODER_PATH)
    if use_intent_encoder_runtime() and rec_enc.is_file():
        return _load_encoder(rec_enc)
    rec_model = _get_path(settings.RECORD_TYPE_MODEL_PATH)
    if rec_model.is_file():
        return _load_tfidf(rec_model)
    return {"backend": "missing"}


def load_encoder_intent_model() -> dict:
    """Comparison track — intent encoder only."""
    if not settings.INTENT_ENCODER_PATH.is_file():
        raise FileNotFoundError(settings.INTENT_ENCODER_PATH)
    return _load_encoder(settings.INTENT_ENCODER_PATH)


def load_encoder_category_model() -> dict:
    if not settings.CATEGORY_ENCODER_PATH.is_file():
        raise FileNotFoundError(settings.CATEGORY_ENCODER_PATH)
    return _load_encoder(settings.CATEGORY_ENCODER_PATH)


def load_encoder_action_type_model() -> dict:
    if not settings.ACTION_TYPE_ENCODER_PATH.is_file():
        raise FileNotFoundError(settings.ACTION_TYPE_ENCODER_PATH)
    return _load_encoder(settings.ACTION_TYPE_ENCODER_PATH)


def load_encoder_record_type_model() -> dict:
    if not settings.RECORD_TYPE_ENCODER_PATH.is_file():
        raise FileNotFoundError(settings.RECORD_TYPE_ENCODER_PATH)
    return _load_encoder(settings.RECORD_TYPE_ENCODER_PATH)


def load_chitchat_sentiment_model():
    """Chitchat dùng LLM — không load PhoBERT sentiment."""
    return {"backend": "llm"}


def predict_category_from_text(text: str | None, category_model: dict) -> str | None:
    """Dự đoán danh mục của một đoạn văn bản ngắn (ví dụ: item từ NER hoặc BILL text)."""
    if not text or not text.strip():
        return None
    from pipeline.text_preprocessing import clean_category_text
    cleaned_input = clean_category_text(text)
    if not cleaned_input:
        cleaned_input = text.strip()
    if category_model.get("backend") == "encoder" and category_model.get("bundle"):
        from src.nlu.encoder_runtime import predict_category_encoder

        return predict_category_encoder(category_model["bundle"], cleaned_input)
    if category_model.get("backend") == "tfidf" and category_model.get("vectorizer"):
        cat_vec = category_model["vectorizer"].transform([cleaned_input])
        return str(category_model["model"].predict(cat_vec)[0])
    return None
