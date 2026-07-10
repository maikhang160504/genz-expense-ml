from __future__ import annotations

import os
from pathlib import Path


KAGGLE_DATASET_SLUG = "domixi1989/vietnamese-receipts-mc-ocr-2021"

KAGGLE_INPUT_CANDIDATES = [
    Path(f"/kaggle/input/datasets/{KAGGLE_DATASET_SLUG}"),
    Path(f"/kaggle/input/{KAGGLE_DATASET_SLUG}"),
]

LOCAL_MIRROR = (
    Path(__file__).resolve().parents[2]
    / "kaggle"
    / "input"
    / "datasets"
    / "domixi1989"
    / "vietnamese-receipts-mc-ocr-2021"
)


def resolve_dataset_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        return root

    env_root = os.environ.get("RECEIPT_OCR_DATA_ROOT")
    if env_root:
        root = Path(env_root)
        if root.is_dir():
            return root

    for candidate in KAGGLE_INPUT_CANDIDATES:
        if candidate.is_dir():
            return candidate

    if LOCAL_MIRROR.is_dir():
        return LOCAL_MIRROR

    raise FileNotFoundError(
        "Không tìm thấy dataset MC-OCR. "
        f"Thêm Kaggle dataset `{KAGGLE_DATASET_SLUG}` hoặc đặt RECEIPT_OCR_DATA_ROOT."
    )


def nested_dir(root: Path, name: str) -> Path:
    """Dataset mirror thường có thư mục lồng: train_images/train_images/."""
    direct = root / name
    nested = direct / name
    if nested.is_dir():
        return nested
    if direct.is_dir():
        return direct
    return direct


def recognition_crop_dir(root: Path) -> Path:
    return nested_dir(root, "text_recognition_mcocr_data")
