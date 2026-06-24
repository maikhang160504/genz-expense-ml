"""Quick local benchmark for MC-OCR pipeline."""
import os
import sys
import time
from pathlib import Path

import psutil

OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR_ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from receipt_ocr.hybrid_pipeline import HybridReceiptOCRPipeline
from receipt_ocr.model_paths import PICK_KIE_MODEL_PATH, VIETOCR_WEIGHTS

img = next((OCR_ROOT / "tests" / "bill-demo").glob("*.jpg"))
print("Image:", img.name, f"({img.stat().st_size / 1024:.0f} KB)")

proc = psutil.Process()
mem_before = proc.memory_info().rss / 1e6

t0 = time.perf_counter()
pipe = HybridReceiptOCRPipeline(
    vietocr_weights=VIETOCR_WEIGHTS,
    device="cpu",
    pick_kie_model=PICK_KIE_MODEL_PATH,
    paddle_use_gpu=False,
).load()
t_load = time.perf_counter() - t0
mem_after_load = proc.memory_info().rss / 1e6

t1 = time.perf_counter()
result = pipe.process_image(img, split_mode=False)
t_infer = time.perf_counter() - t1
mem_after = proc.memory_info().rss / 1e6

print(f"Load pipeline: {t_load:.1f}s")
print(f"1 image infer: {t_infer:.1f}s")
print(f"KIE backend: {result.get('kie_backend')}")
print(f"Amount: {result.get('amount')}, Category: {result.get('category')}")
print(f"Boxes: {len(result.get('boxes', []))}, Lines: {len(result.get('lines', []))}")
print(
    f"Process RSS MB: before={mem_before:.0f} load={mem_after_load:.0f} "
    f"after={mem_after:.0f} (+{mem_after - mem_before:.0f})"
)
