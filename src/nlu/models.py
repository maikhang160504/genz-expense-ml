import sys
import warnings

import joblib

from src.config import settings

_PATCHED_TOKENIZER = False


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
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import text_preprocessing as tp_pkg

    from pipeline.text_preprocessing import vi_tokenize

    tp_pkg.vi_tokenize = vi_tokenize
    _PATCHED_TOKENIZER = True


def load_intent_model():
    """Trả dict: ``{backend: encoder|tfidf, ...}`` để pipeline phân nhánh."""
    if settings.INTENT_ENCODER_PATH.exists():
        return {"backend": "encoder", "bundle": joblib.load(settings.INTENT_ENCODER_PATH)}
    _ensure_pickled_tokenizer()
    payload = joblib.load(settings.MODEL_PATH)
    return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}


def load_category_model():
    if settings.CATEGORY_ENCODER_PATH.exists():
        return {"backend": "encoder", "bundle": joblib.load(settings.CATEGORY_ENCODER_PATH)}
    if settings.CATEGORY_MODEL_PATH.exists():
        _ensure_pickled_tokenizer()
        payload = joblib.load(settings.CATEGORY_MODEL_PATH)
        return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    return {"backend": "missing"}


def load_action_type_model():
    if settings.ACTION_TYPE_ENCODER_PATH.exists():
        return {"backend": "encoder", "bundle": joblib.load(settings.ACTION_TYPE_ENCODER_PATH)}
    if settings.ACTION_TYPE_MODEL_PATH.exists():
        _ensure_pickled_tokenizer()
        payload = joblib.load(settings.ACTION_TYPE_MODEL_PATH)
        return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    return {"backend": "missing"}


def load_record_type_model():
    if settings.RECORD_TYPE_ENCODER_PATH.exists():
        return {"backend": "encoder", "bundle": joblib.load(settings.RECORD_TYPE_ENCODER_PATH)}
    if settings.RECORD_TYPE_MODEL_PATH.exists():
        _ensure_pickled_tokenizer()
        payload = joblib.load(settings.RECORD_TYPE_MODEL_PATH)
        return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    return {"backend": "missing"}


def load_chitchat_sentiment_model():
    """
    Chitchat dùng LLM cho tone + trả lời — không load PhoBERT sentiment.
    Giữ hàm để signature run_nlu() không đổi.
    """
    return {"backend": "llm"}
