from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
# Thư mục dữ liệu + mô hình text (đổi tên từ TF-IDF → text_nlu; encoder + joblib cũ cùng nơi)
TEXT_NLU_DIR = ROOT_DIR / "text_nlu"
TFIDF_DIR = TEXT_NLU_DIR  # alias cũ, tránh gãy import

ENV_PATH = ROOT_DIR / ".env"
REQUEST_TEMPLATE_PATH = ROOT_DIR / "format_request_reponse.md"
PROMPTS_PATH = ROOT_DIR / "src" / "prompts" / "prompts.json"

MODEL_PATH = TEXT_NLU_DIR / "models" / "intent_model.joblib"
CATEGORY_MODEL_PATH = TEXT_NLU_DIR / "models" / "category_model.joblib"
ACTION_TYPE_MODEL_PATH = TEXT_NLU_DIR / "models" / "action_type_model.joblib"
ACTION_SLOTS_MODEL_PATH = TEXT_NLU_DIR / "models" / "action_slots_model.joblib"
RECORD_TYPE_MODEL_PATH = TEXT_NLU_DIR / "models" / "record_type_model.joblib"
CHITCHAT_SENTIMENT_MODEL_PATH = TEXT_NLU_DIR / "models" / "chitchat_sentiment_model.joblib"
NER_MODEL_DIR = TEXT_NLU_DIR / "models" / "ner_model" / "model-best"

# Encoder (PhoBERT) — experimental / A-B compare only; production uses TF-IDF (see NLU_USE_ENCODER)
INTENT_ENCODER_PATH = TEXT_NLU_DIR / "models" / "intent_encoder.joblib"
ACTION_TYPE_ENCODER_PATH = TEXT_NLU_DIR / "models" / "action_type_encoder.joblib"
CHITCHAT_ENCODER_PATH = TEXT_NLU_DIR / "models" / "chitchat_encoder.joblib"
RECORD_TYPE_ENCODER_PATH = TEXT_NLU_DIR / "models" / "record_type_encoder.joblib"
CATEGORY_ENCODER_PATH = TEXT_NLU_DIR / "models" / "category_encoder.joblib"
CHITCHAT_PHOBERT_DIR = TEXT_NLU_DIR / "models" / "chitchat_phobert_sentiment"
