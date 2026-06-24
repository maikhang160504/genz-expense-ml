"""Tests for heuristic KIE, brand routing, pick export, kaggle runner, golden eval."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR_ROOT / "src"))

from receipt_ocr.brand_routing import route_brand_category  # noqa: E402
from receipt_ocr.golden_eval import eval_summary_against_golden, load_golden_fixtures, run_golden_eval  # noqa: E402
from receipt_ocr.kaggle_runner import build_retrain_plan, find_kaggle_credentials  # noqa: E402
from receipt_ocr.model_paths import PICK_KIE_MODEL_PATH, VIETOCR_WEIGHTS  # noqa: E402
from receipt_ocr.pick_kie import PickKIEEngine, extract_kie_fields, heuristic_label_boxes, pick_kie_weights_status  # noqa: E402
from receipt_ocr.pick_export import export_verified_samples, sample_to_pick_tsv  # noqa: E402
from receipt_ocr.receipt_fusion import (
    fuse_receipt_summary,
    run_kie_branch,
    run_prep_branch,
)  # noqa: E402
from receipt_ocr.receipt_nlu import extract_receipt_summary  # noqa: E402


def test_paddle_poly_to_coors():
    from receipt_ocr.rotation_corrector import paddle_poly_to_coors, rotation_weights_status

    poly = [[10, 20], [100, 22], [98, 40], [12, 38]]
    coors = paddle_poly_to_coors(poly)
    assert coors == [10, 20, 100, 22, 98, 40, 12, 38]
    status = rotation_weights_status()
    assert "ready" in status
    assert "model_path" in status


def test_fusion_branch_helpers_match_summary():
    lines = pd.DataFrame([
        {"line_text": "HIGHLANDS COFFEE", "bbox": [0, 0, 100, 20]},
        {"line_text": "Tong cong 45000", "bbox": [0, 60, 100, 80]},
    ])
    boxes = pd.DataFrame([
        {"x1": 0, "y1": 0, "x2": 100, "y2": 20, "text": "HIGHLANDS COFFEE"},
        {"x1": 0, "y1": 60, "x2": 140, "y2": 80, "text": "Tong cong 45000"},
    ])
    labeled = heuristic_label_boxes(boxes)
    kie_fields = extract_kie_fields(labeled)
    prep = run_prep_branch(boxes, lines)
    assert prep.lines
    fused = fuse_receipt_summary(lines, boxes, kie_fields, prep=prep)
    direct = extract_receipt_summary(lines, df_boxes=boxes, kie_fields=kie_fields, split_mode=False)
    assert fused["category"] == direct["category"]
    assert fused["amount"] == direct["amount"]


def test_kie_branch_from_heuristic_boxes():
    df = pd.DataFrame([
        {"x1": 10, "y1": 5, "x2": 200, "y2": 30, "text": "WINMART"},
        {"x1": 10, "y1": 80, "x2": 240, "y2": 100, "text": "Tong cong 120000"},
    ])
    from receipt_ocr.pick_kie import PickKIEEngine

    engine = PickKIEEngine().load()
    kie = run_kie_branch(engine, df, image_rgb=None)
    assert kie.kie_backend in ("heuristic", "pick")
    assert kie.kie_fields.get("TOTAL_COST_VALUE") == 120000


def test_brand_routing():
    assert route_brand_category("HIGHLANDS COFFEE") == "Food"
    assert route_brand_category("WINMART PLUS") == "Essentials"
    assert route_brand_category("Tap hoa nho") is None


def test_heuristic_kie_labels():
    df = pd.DataFrame([
        {"x1": 10, "y1": 5, "x2": 200, "y2": 30, "text": "HIGHLANDS COFFEE"},
        {"x1": 10, "y1": 80, "x2": 240, "y2": 100, "text": "Tong cong 45000"},
    ])
    labeled = heuristic_label_boxes(df)
    assert labeled[0]["entity"] == "SELLER"
    assert labeled[1]["entity"] == "TOTAL_COST"
    fields = extract_kie_fields(labeled)
    assert fields["TOTAL_COST_VALUE"] == 45000


def test_extract_receipt_with_kie_and_brand():
    lines = pd.DataFrame([
        {"line_text": "HIGHLANDS COFFEE", "bbox": [0, 0, 100, 20]},
        {"line_text": "Ca phe 45000", "bbox": [0, 30, 100, 50]},
        {"line_text": "Tong cong 45000", "bbox": [0, 60, 100, 80]},
    ])
    boxes = pd.DataFrame([
        {"x1": 0, "y1": 0, "x2": 100, "y2": 20, "text": "HIGHLANDS COFFEE"},
        {"x1": 0, "y1": 30, "x2": 80, "y2": 50, "text": "Ca phe"},
        {"x1": 90, "y1": 30, "x2": 140, "y2": 50, "text": "45000"},
        {"x1": 0, "y1": 60, "x2": 140, "y2": 80, "text": "Tong cong 45000"},
    ])
    kie = {"SELLER": "HIGHLANDS COFFEE", "TOTAL_COST": "Tong cong 45000", "TOTAL_COST_VALUE": 45000}
    out = extract_receipt_summary(lines, df_boxes=boxes, kie_fields=kie, split_mode=False)
    assert out["category"] == "Food"
    assert out["brand_routed"] is True
    assert out["amount"] == 45000


def test_pick_export_tsv():
    boxes = [
        {"text": "WINMART", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "entity": "SELLER"},
    ]
    tsv = sample_to_pick_tsv("sample_1", boxes)
    assert "WINMART" in tsv
    assert "SELLER" in tsv


def test_export_verified_samples(tmp_path):
    samples = [{"id": "abc123", "admin_labels": [
        {"text": "A", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "entity": "SELLER"},
    ]}]
    result = export_verified_samples(samples, tmp_path)
    assert result["manifest"]["count"] == 1
    assert (tmp_path / "boxes_and_transcripts" / "abc123.tsv").is_file()


def test_model_paths_resolve():
    assert VIETOCR_WEIGHTS.parent.name == "vietocr"
    assert VIETOCR_WEIGHTS.name == "vgg_transformer.pth"
    assert PICK_KIE_MODEL_PATH.name == "model_best.pth"
    assert PICK_KIE_MODEL_PATH.parent.name == "pick_kie"


def test_pick_kie_weights_status():
    status = pick_kie_weights_status()
    assert status["model_path"].endswith("model_best.pth")
    assert "weights_found" in status
    assert "pick_kie" in status["model_path"]
    plan = build_retrain_plan("pick_retrain", "/tmp/verified")
    assert "vietnamese-receipts-mc-ocr-2021" in plan["base_dataset"]
    assert plan["job_type"] == "pick_retrain"
    assert isinstance(plan["kaggle_configured"], bool)


def test_pick_tsv_from_df_boxes():
    df = pd.DataFrame([
        {"x1": 10, "y1": 5, "x2": 200, "y2": 30, "text": "WINMART"},
    ])
    from receipt_ocr.pick_kie_inference import df_boxes_to_pick_tsv

    tsv = df_boxes_to_pick_tsv(df)
    assert "WINMART" in tsv
    assert tsv.startswith("1,")


def test_build_training_pack(tmp_path):
    from receipt_ocr.kaggle_dataset_builder import build_training_pack

    img = tmp_path / "img.jpg"
    img.write_bytes(b"jpeg")
    samples = [{
        "id": "abc123",
        "admin_labels": [{"text": "WIN", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "entity": "SELLER"}],
        "image_path": str(img),
        "image_ext": ".jpg",
    }]
    result = build_training_pack(samples, tmp_path / "verified")
    assert result["manifest"]["count"] == 1
    assert result["images_copied"] == 1
    assert (tmp_path / "verified" / "kaggle_upload" / "dataset-metadata.json").is_file()
    assert (tmp_path / "verified" / "incremental" / "training_pack.json").is_file()


def test_golden_fixtures_eval():
    fixtures = load_golden_fixtures()
    assert len(fixtures) >= 2
    results = []
    for fx in fixtures:
        lines = pd.DataFrame([{"line_text": ln, "bbox": [0, 0, 1, 1]} for ln in fx["lines"]])
        boxes = pd.DataFrame(fx["boxes"])
        kie = extract_kie_fields(fx["boxes"])
        pred = extract_receipt_summary(lines, df_boxes=boxes, kie_fields=kie, split_mode=False)
        metrics = eval_summary_against_golden(pred, fx["expected"])
        results.append({"fixture_id": fx["fixture_id"], "predicted": pred, "expected": fx["expected"], "metrics": metrics})
    report = run_golden_eval(results)
    assert report["n"] >= 2
    assert report["category_acc"] >= 0.5


if __name__ == "__main__":
    test_brand_routing()
    test_heuristic_kie_labels()
    test_extract_receipt_with_kie_and_brand()
    test_pick_export_tsv()
    test_export_verified_samples(Path("/tmp/test_pick_export_manual"))
    test_golden_fixtures_eval()
    print("All hybrid OCR tests passed.")
