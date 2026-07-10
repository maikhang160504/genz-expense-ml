"""Canonical paths for OCR production weights and training artifacts."""
from __future__ import annotations

import os
from pathlib import Path

OCR_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = OCR_ROOT / "models"
ARTIFACTS_DIR = OCR_ROOT / "artifacts"
MANIFESTS_DIR = OCR_ROOT / "manifests"

VIETOCR_DIR = MODELS_DIR / "vietocr"
VIETOCR_WEIGHTS = VIETOCR_DIR / "vgg_transformer.pth"
VIETOCR_PRETRAIN_URL = "https://vocr.vn/data/vietocr/vgg_transformer.pth"
VIETOCR_LEGACY_WEIGHTS = VIETOCR_DIR / "vietocr_receipt.pth"
VIETOCR_CONFIG = VIETOCR_DIR / "config.yml"
VIETOCR_META = VIETOCR_DIR / "meta.json"

PICK_KIE_DIR = MODELS_DIR / "pick_kie"
PICK_KIE_MODEL_PATH = PICK_KIE_DIR / "model_best.pth"
PICK_KIE_CONFIG = PICK_KIE_DIR / "config.json"
PICK_KIE_META = PICK_KIE_DIR / "meta.json"

LAYOUTLMV3_DIR = MODELS_DIR / "layoutlmv3"
LAYOUTLMV3_MODEL_PATH = LAYOUTLMV3_DIR / "model_best.pth"

ROTATION_DIR = MODELS_DIR / "rotation_corrector"
ROTATION_WEIGHTS = ROTATION_DIR / "mobilenetv3-Epoch-487-Loss-0.03-Acc-0.99.pth"

VIETOCR_ARTIFACTS_DIR = ARTIFACTS_DIR / "vietocr"
PICK_KIE_ARTIFACTS_DIR = ARTIFACTS_DIR / "pick_kie"
OCR_MANIFEST = MANIFESTS_DIR / "ocr_models.json"

KAGGLE_KERNELS_DIR = OCR_ROOT / "kaggle" / "kernels"

# Kaggle notebook output folder names (inside kernel working dir)
KAGGLE_OUT_VIETOCR = "receipt_ocr_artifacts"
KAGGLE_OUT_PICK_KIE = "pick_kie_artifacts"


def resolve_vietocr_weights_path(explicit: str | Path | None = None) -> Path:
    """Find VietOCR weights on disk; prefer explicit path when valid."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for env_key in ("OCR_WEIGHTS_PATH", "RECEIPT_OCR_WEIGHTS"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            candidates.append(Path(env_val))
    candidates.extend(
        [
            VIETOCR_WEIGHTS,
            VIETOCR_LEGACY_WEIGHTS,
            MODELS_DIR / "vgg_transformer.pth",
            MODELS_DIR / "vietocr_receipt.pth",
        ]
    )
    seen: set[str] = set()
    for raw in candidates:
        p = raw.expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p.resolve()
    return Path(explicit).resolve() if explicit else VIETOCR_WEIGHTS


def resolve_rotation_weights_path(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for env_key in ("ROTATION_MODEL_PATH", "ROTATION_WEIGHTS_PATH"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            candidates.append(Path(env_val))
    candidates.append(ROTATION_WEIGHTS)
    seen: set[str] = set()
    for raw in candidates:
        key = str(raw.expanduser())
        if key in seen:
            continue
        seen.add(key)
        if raw.is_file():
            return raw.resolve()
    return (candidates[0] if candidates else ROTATION_WEIGHTS).resolve()
