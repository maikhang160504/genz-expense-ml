"""Chạy pipeline OCR trên tất cả ảnh trong demo/images."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)

DEMO_DIR = Path(__file__).resolve().parent
OCR_ROOT = DEMO_DIR.parent
ROOT = OCR_ROOT.parent
sys.path.insert(0, str(OCR_ROOT / "src"))

from receipt_ocr.paddle_compat import setup_paddle_env  # noqa: E402

setup_paddle_env()

from receipt_ocr.pipeline import ReceiptOCRPipeline  # noqa: E402

IMAGES_DIR = DEMO_DIR / "images"
DEFAULT_WEIGHTS = OCR_ROOT / "models" / "vietocr_receipt.pth"
OUT_DIR = DEMO_DIR / "output"


def main() -> None:
    weights = Path(os.environ.get("RECEIPT_OCR_WEIGHTS", str(DEFAULT_WEIGHTS)))
    if not weights.is_file():
        raise FileNotFoundError(f"Thiếu weights: {weights}")

    images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"Không có ảnh trong {IMAGES_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Weights: {weights}")
    log(f"Images: {len(images)}")

    log("Loading Paddle (detection) + VietOCR (first run may download models)...")
    pipeline = ReceiptOCRPipeline(weights).load()
    log("Pipeline ready.")

    for i, img_path in enumerate(images, 1):
        log(f"\n[{i}/{len(images)}] {img_path.name}")
        result = pipeline.process_image(img_path)
        log("NLU: " + json.dumps(result, ensure_ascii=False, indent=2))

        out_file = OUT_DIR / f"{img_path.stem}.json"
        out_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"Saved: {out_file}")

    log(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
