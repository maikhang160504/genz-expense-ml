"""Smoke-test PICK KIE with pretrained MC_OCR weights."""
import os
import sys
import time
from pathlib import Path

OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR_ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import pandas as pd

from receipt_ocr.model_paths import PICK_KIE_MODEL_PATH, VIETOCR_WEIGHTS
from receipt_ocr.pick_kie import PickKIEEngine, pick_kie_weights_status
from receipt_ocr.hybrid_pipeline import HybridReceiptOCRPipeline
from receipt_ocr.receipt_fusion import run_ocr_stage, run_kie_branch

img_path = next((OCR_ROOT / "tests" / "bill-demo").glob("*.jpg"))
print("Weights status:", pick_kie_weights_status(PICK_KIE_MODEL_PATH))

t0 = time.perf_counter()
pipe = HybridReceiptOCRPipeline(
    vietocr_weights=VIETOCR_WEIGHTS,
    device="cpu",
    pick_kie_model=PICK_KIE_MODEL_PATH,
    paddle_use_gpu=False,
).load()
print(f"Pipeline load: {time.perf_counter() - t0:.1f}s")

image_rgb = pipe._read_rgb(img_path)
ocr = run_ocr_stage(pipe, image_rgb)
kie_engine = pipe._get_kie()
print("KIE backend after load:", kie_engine.backend)
if kie_engine._load_error:
    print("Load note:", kie_engine._load_error)

t1 = time.perf_counter()
kie = run_kie_branch(kie_engine, ocr.df_boxes, image_rgb)
print(f"KIE branch: {time.perf_counter() - t1:.1f}s")
print("KIE backend used:", kie.kie_backend)
print("Fields:", kie.kie_fields)

labeled = kie.labeled_boxes
entities = pd.Series([b.get("entity") for b in labeled]).value_counts().to_dict()
print("Entity counts:", entities)
