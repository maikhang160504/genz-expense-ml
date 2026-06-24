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
    encoder_exists = settings.INTENT_ENCODER_PATH.exists()
    tfidf_exists = settings.MODEL_PATH.exists()

    if encoder_exists and tfidf_exists:
        enc_mtime = settings.INTENT_ENCODER_PATH.stat().st_mtime
        tfidf_mtime = settings.MODEL_PATH.stat().st_mtime
        if enc_mtime >= tfidf_mtime:
            return {"backend": "encoder", "bundle": joblib.load(settings.INTENT_ENCODER_PATH)}
        else:
            _ensure_pickled_tokenizer()
            payload = joblib.load(settings.MODEL_PATH)
            return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    elif encoder_exists:
        return {"backend": "encoder", "bundle": joblib.load(settings.INTENT_ENCODER_PATH)}
    elif tfidf_exists:
        _ensure_pickled_tokenizer()
        payload = joblib.load(settings.MODEL_PATH)
        return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    else:
        raise FileNotFoundError(f"Neither intent encoder model ({settings.INTENT_ENCODER_PATH}) nor TF-IDF model ({settings.MODEL_PATH}) exists.")


def load_category_model():
    encoder_exists = settings.CATEGORY_ENCODER_PATH.exists()
    tfidf_exists = settings.CATEGORY_MODEL_PATH.exists()

    if encoder_exists and tfidf_exists:
        enc_mtime = settings.CATEGORY_ENCODER_PATH.stat().st_mtime
        tfidf_mtime = settings.CATEGORY_MODEL_PATH.stat().st_mtime
        if enc_mtime >= tfidf_mtime:
            return {"backend": "encoder", "bundle": joblib.load(settings.CATEGORY_ENCODER_PATH)}
        else:
            _ensure_pickled_tokenizer()
            payload = joblib.load(settings.CATEGORY_MODEL_PATH)
            return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    elif encoder_exists:
        return {"backend": "encoder", "bundle": joblib.load(settings.CATEGORY_ENCODER_PATH)}
    elif tfidf_exists:
        _ensure_pickled_tokenizer()
        payload = joblib.load(settings.CATEGORY_MODEL_PATH)
        return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    return {"backend": "missing"}


def load_action_type_model():
    encoder_exists = settings.ACTION_TYPE_ENCODER_PATH.exists()
    tfidf_exists = settings.ACTION_TYPE_MODEL_PATH.exists()

    if encoder_exists and tfidf_exists:
        enc_mtime = settings.ACTION_TYPE_ENCODER_PATH.stat().st_mtime
        tfidf_mtime = settings.ACTION_TYPE_MODEL_PATH.stat().st_mtime
        if enc_mtime >= tfidf_mtime:
            return {"backend": "encoder", "bundle": joblib.load(settings.ACTION_TYPE_ENCODER_PATH)}
        else:
            _ensure_pickled_tokenizer()
            payload = joblib.load(settings.ACTION_TYPE_MODEL_PATH)
            return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    elif encoder_exists:
        return {"backend": "encoder", "bundle": joblib.load(settings.ACTION_TYPE_ENCODER_PATH)}
    elif tfidf_exists:
        _ensure_pickled_tokenizer()
        payload = joblib.load(settings.ACTION_TYPE_MODEL_PATH)
        return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    return {"backend": "missing"}


def load_record_type_model():
    encoder_exists = settings.RECORD_TYPE_ENCODER_PATH.exists()
    tfidf_exists = settings.RECORD_TYPE_MODEL_PATH.exists()

    if encoder_exists and tfidf_exists:
        enc_mtime = settings.RECORD_TYPE_ENCODER_PATH.stat().st_mtime
        tfidf_mtime = settings.RECORD_TYPE_MODEL_PATH.stat().st_mtime
        if enc_mtime >= tfidf_mtime:
            return {"backend": "encoder", "bundle": joblib.load(settings.RECORD_TYPE_ENCODER_PATH)}
        else:
            _ensure_pickled_tokenizer()
            payload = joblib.load(settings.RECORD_TYPE_MODEL_PATH)
            return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    elif encoder_exists:
        return {"backend": "encoder", "bundle": joblib.load(settings.RECORD_TYPE_ENCODER_PATH)}
    elif tfidf_exists:
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


def predict_category_from_text(text: str | None, category_model: dict) -> str | None:
    """Dự đoán danh mục của một đoạn văn bản ngắn (ví dụ: item từ NER) bằng category_model."""
    if not text or not text.strip():
        return None
    if category_model.get("backend") == "encoder" and category_model.get("bundle"):
        from src.nlu.encoder_runtime import predict_category_encoder
        return predict_category_encoder(category_model["bundle"], text)
    elif category_model.get("backend") == "tfidf" and category_model.get("vectorizer"):
        from pipeline.text_preprocessing import clean_category_text
        cat_input = clean_category_text(text)
        cat_vec = category_model["vectorizer"].transform([cat_input])
        return str(category_model["model"].predict(cat_vec)[0])
    return None
