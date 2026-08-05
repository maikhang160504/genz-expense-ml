"""Settings loaded from environment / .env file."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


HERE = Path(__file__).resolve()
SERVICE_ROOT = HERE.parents[2]
for p in (HERE.parents[2], HERE.parents[3], HERE.parents[4]):
    if (p / ".env").is_file() or (p / ".git").is_dir() or p.name in ("ai-service", "expense-ocr-nlu"):
        SERVICE_ROOT = p
        break

PROJECT_ROOT = SERVICE_ROOT.parent
if SERVICE_ROOT.name == "expense-ocr-nlu":
    EXPENSE_OCR_NLU_DIR = SERVICE_ROOT
else:
    EXPENSE_OCR_NLU_DIR = PROJECT_ROOT / "expense-ocr-nlu"

_ENV_PATH = SERVICE_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)

_NLU_ENV_PATH = EXPENSE_OCR_NLU_DIR / ".env"
if _NLU_ENV_PATH.exists():
    load_dotenv(_NLU_ENV_PATH, override=False)

# Force CPU mode early if DEVICE is set to "cpu" or defaults to it.
# This avoids CUDA context initialization hangs on environments with broken/unconfigured drivers.
_device_env = (os.getenv("DEVICE") or os.getenv("device") or "cpu").lower().strip().replace('"', '').replace("'", "")
if _device_env == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["USE_CUDA"] = "0"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = ""


class Settings(BaseSettings):
    """Runtime configuration for the AI microservice."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH) if _ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Expense AI Service")
    environment: str = Field(default="local")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    device: str = Field(default="cpu", description="Target device for ML models ('cpu' or 'cuda').")

    use_real_nlu: bool = Field(default=False, description="Load full NLU pipeline (joblib + PhoBERT).")
    use_real_ocr: bool = Field(default=False, description="Load PaddleOCR + VietOCR weights.")
    lazy_load_models: bool = Field(
        default=True,
        description="If true, skip NLU weight load at startup; OCR may still preload when preload_ocr=true.",
    )
    preload_ocr: bool = Field(
        default=True,
        description="Eager-load OCR (Paddle+VietOCR+PICK) at startup when use_real_ocr=true.",
    )
    expense_ocr_nlu_dir: str = Field(default=str(EXPENSE_OCR_NLU_DIR))
    nlu_models_dir: str = Field(default="")
    ocr_weights_path: str = Field(default="")
    pick_kie_model_path: str = Field(default="")
    rotation_model_path: str = Field(default="")
    use_rotation_corrector: bool = Field(
        default=True,
        description="Apply MC_OCR MobileNetV3 page rotation before OCR (pretrained weights).",
    )
    kaggle_config_dir: str = Field(default="")
    verified_ocr_labels_dir: str = Field(default="")

    qwen_api_url: str | None = Field(default=None, alias="QWEN_API_URL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash")
    run_llm: bool = Field(default=False)

    api_key: str | None = Field(
        default=None,
        description="Optional shared secret. When set, requests must send X-API-Key header.",
    )
    cors_origins: str = Field(default="*")

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> Settings:
        # Resolve expense_ocr_nlu_dir relative to SERVICE_ROOT if it's relative
        p_nlu = Path(self.expense_ocr_nlu_dir)
        if not p_nlu.is_absolute():
            self.expense_ocr_nlu_dir = str((SERVICE_ROOT / p_nlu).resolve())

        # Resolve ocr_weights_path relative to SERVICE_ROOT if it's relative
        if self.ocr_weights_path:
            p_ocr = Path(self.ocr_weights_path)
            if not p_ocr.is_absolute():
                self.ocr_weights_path = str((SERVICE_ROOT / p_ocr).resolve())
            # Auto-correct legacy paths → models/vietocr/vgg_transformer.pth
            if not Path(self.ocr_weights_path).is_file():
                try:
                    from receipt_ocr.model_paths import resolve_vietocr_weights_path

                    resolved = resolve_vietocr_weights_path(self.ocr_weights_path)
                    if resolved.is_file():
                        self.ocr_weights_path = str(resolved)
                except Exception:
                    pass
        elif EXPENSE_OCR_NLU_DIR:
            try:
                ocr_root = Path(self.expense_ocr_nlu_dir) / "bill_ocr"
                if str(ocr_root) not in sys.path:
                    sys.path.insert(0, str(ocr_root))
                from receipt_ocr.model_paths import resolve_vietocr_weights_path

                resolved = resolve_vietocr_weights_path(None)
                if resolved.is_file():
                    self.ocr_weights_path = str(resolved)
            except Exception:
                pass
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.cors_origins or self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_expense_ocr_nlu_path() -> Path:
    return Path(get_settings().expense_ocr_nlu_dir).resolve()
