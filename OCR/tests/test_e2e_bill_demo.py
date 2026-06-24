"""End-to-end hybrid OCR tests on real bill demo images (requires VietOCR weights)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR_ROOT / "src"))

from receipt_ocr.model_paths import PICK_KIE_MODEL_PATH, VIETOCR_WEIGHTS  # noqa: E402

BILL_DEMO = OCR_ROOT / "tests" / "bill-demo"
WEIGHTS_READY = VIETOCR_WEIGHTS.is_file()


@pytest.mark.skipif(not WEIGHTS_READY, reason=f"VietOCR weights missing at {VIETOCR_WEIGHTS}")
@pytest.mark.skipif(
    os.environ.get("USE_REAL_OCR", "true").lower() not in ("1", "true", "yes"),
    reason="Set USE_REAL_OCR=true to run bill-demo e2e",
)
@pytest.mark.parametrize("image_path", sorted(BILL_DEMO.glob("*.jpg")), ids=lambda p: p.name)
def test_bill_demo_hybrid_pipeline(image_path: Path):
    from receipt_ocr.hybrid_pipeline import HybridReceiptOCRPipeline

    pipeline = HybridReceiptOCRPipeline(
        vietocr_weights=VIETOCR_WEIGHTS,
        device=os.environ.get("DEVICE", "cpu"),
        pick_kie_model=PICK_KIE_MODEL_PATH,
        paddle_use_gpu=False,
    ).load()

    result = pipeline.process_image(image_path, split_mode=False)

    assert result.get("amount") is not None and result["amount"] > 0, f"No amount for {image_path.name}"
    assert result.get("category"), f"No category for {image_path.name}"
    assert isinstance(result.get("lines"), list) and len(result["lines"]) > 0
    assert isinstance(result.get("boxes"), list) and len(result["boxes"]) > 0
    assert result.get("kie_backend") in ("pick", "heuristic")


def test_bill_demo_images_exist():
    images = list(BILL_DEMO.glob("*.jpg"))
    assert len(images) >= 3, f"Expected >=3 bill-demo images, found {len(images)}"
