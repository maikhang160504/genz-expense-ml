"""Tests for artifact deploy (uses temp dirs, no production overwrite)."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR_ROOT / "src"))

from receipt_ocr import artifact_deploy as ad  # noqa: E402
from receipt_ocr import model_paths as mp  # noqa: E402


def test_deploy_pick_kie_from_folder(tmp_path, monkeypatch):
    fake_root = tmp_path / "ocr"
    models = fake_root / "models" / "pick_kie"
    models.mkdir(parents=True)

    artifact = tmp_path / "out" / "pick_kie_artifacts"
    artifact.mkdir(parents=True)
    (artifact / "model_best.pth").write_bytes(b"pick-weights")
    (artifact / "meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mp, "PICK_KIE_DIR", fake_root / "models" / "pick_kie")
    monkeypatch.setattr(mp, "PICK_KIE_MODEL_PATH", fake_root / "models" / "pick_kie" / "model_best.pth")
    monkeypatch.setattr(mp, "PICK_KIE_ARTIFACTS_DIR", fake_root / "artifacts" / "pick_kie")
    monkeypatch.setattr(mp, "OCR_MANIFEST", fake_root / "manifests" / "ocr_models.json")
    monkeypatch.setattr(ad, "PICK_KIE_DIR", mp.PICK_KIE_DIR)
    monkeypatch.setattr(ad, "PICK_KIE_MODEL_PATH", mp.PICK_KIE_MODEL_PATH)
    monkeypatch.setattr(ad, "PICK_KIE_ARTIFACTS_DIR", mp.PICK_KIE_ARTIFACTS_DIR)
    monkeypatch.setattr(ad, "OCR_MANIFEST", mp.OCR_MANIFEST)

    report = ad.deploy_pick_kie(tmp_path / "out")
    assert report["job_type"] == "pick_kie"
    deployed = mp.PICK_KIE_MODEL_PATH
    assert deployed.is_file()
    assert deployed.read_bytes() == b"pick-weights"


def test_deploy_vietocr_from_zip(tmp_path, monkeypatch):
    fake_root = tmp_path / "ocr"
    vietocr_dir = fake_root / "models" / "vietocr"
    vietocr_dir.mkdir(parents=True)
    (vietocr_dir / "vgg_transformer.pth").write_bytes(b"old")

    staging = tmp_path / "zip_src"
    inner = staging / "receipt_ocr_artifacts"
    inner.mkdir(parents=True)
    (inner / "vietocr_receipt.pth").write_bytes(b"new-weights")
    zip_path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(inner / "vietocr_receipt.pth", "receipt_ocr_artifacts/vietocr_receipt.pth")

    monkeypatch.setattr(mp, "VIETOCR_DIR", vietocr_dir)
    monkeypatch.setattr(mp, "VIETOCR_WEIGHTS", vietocr_dir / "vgg_transformer.pth")
    monkeypatch.setattr(mp, "VIETOCR_ARTIFACTS_DIR", fake_root / "artifacts" / "vietocr")
    monkeypatch.setattr(mp, "OCR_MANIFEST", fake_root / "manifests" / "ocr_models.json")
    monkeypatch.setattr(ad, "VIETOCR_DIR", mp.VIETOCR_DIR)
    monkeypatch.setattr(ad, "VIETOCR_WEIGHTS", mp.VIETOCR_WEIGHTS)
    monkeypatch.setattr(ad, "VIETOCR_ARTIFACTS_DIR", mp.VIETOCR_ARTIFACTS_DIR)
    monkeypatch.setattr(ad, "OCR_MANIFEST", mp.OCR_MANIFEST)

    report = ad.deploy_from_source(zip_path, "vietocr", "test_batch")
    assert report["ok"] is True
    assert mp.VIETOCR_WEIGHTS.read_bytes() == b"new-weights"
