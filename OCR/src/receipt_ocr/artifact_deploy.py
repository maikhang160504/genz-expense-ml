"""Deploy trained OCR artifacts from Kaggle output or cloud URL into production models/."""
from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_paths import (
    KAGGLE_OUT_PICK_KIE,
    KAGGLE_OUT_VIETOCR,
    OCR_MANIFEST,
    PICK_KIE_ARTIFACTS_DIR,
    PICK_KIE_CONFIG,
    PICK_KIE_DIR,
    PICK_KIE_META,
    PICK_KIE_MODEL_PATH,
    VIETOCR_ARTIFACTS_DIR,
    VIETOCR_DIR,
    VIETOCR_WEIGHTS,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _backup_dir(src: Path, backup_root: Path) -> Path | None:
    if not src.exists():
        return None
    dest = backup_root / _timestamp()
    shutil.copytree(src, dest)
    return dest


def _find_artifact_root(extracted: Path, job_type: str) -> Path:
    """Locate Kaggle working folder inside extracted zip/dir."""
    if job_type in ("pick_kie", "pick_retrain", "pick_train"):
        expected = KAGGLE_OUT_PICK_KIE
    else:
        expected = KAGGLE_OUT_VIETOCR
    candidates = [
        extracted / expected,
        extracted / expected / expected,
        extracted,
    ]
    for c in candidates:
        if job_type in ("pick_kie", "pick_retrain", "pick_train"):
            if (c / "model_best.pth").is_file():
                return c
        elif (c / "vgg_transformer.pth").is_file() or (c / "vietocr_receipt.pth").is_file():
            return c
    raise FileNotFoundError(f"No {job_type} artifacts found under {extracted}")


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return dest


def _download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310
        data = resp.read()
    dest.write_bytes(data)
    return dest


def prepare_source(source: str | Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return extracted directory from local path, zip, or http(s) URL."""
    p = Path(source) if not isinstance(source, Path) else source
    if str(source).startswith(("http://", "https://")):
        tmp = tempfile.TemporaryDirectory()
        zip_path = Path(tmp.name) / "artifact.zip"
        _download_url(str(source), zip_path)
        extracted = _extract_zip(zip_path, Path(tmp.name) / "unpacked")
        return extracted, tmp
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        extracted = _extract_zip(p, Path(tmp.name) / "unpacked")
        return extracted, tmp
    if p.is_dir():
        return p, None
    raise FileNotFoundError(f"Artifact source not found: {source}")


def deploy_pick_kie(artifact_root: Path) -> dict[str, Any]:
    """Replace models/pick_kie/model_best.pth from Kaggle output tree."""
    root = _find_artifact_root(artifact_root, "pick_kie")
    weight_src = root / "model_best.pth"
    if not weight_src.is_file():
        raise FileNotFoundError("model_best.pth missing in PICK artifacts")

    backup_root = PICK_KIE_ARTIFACTS_DIR / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = _backup_dir(PICK_KIE_DIR, backup_root)

    PICK_KIE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weight_src, PICK_KIE_MODEL_PATH)
    for extra in ("config.json", "meta.json"):
        src = root / extra
        if src.is_file():
            shutil.copy2(src, PICK_KIE_DIR / extra)

    return {
        "job_type": "pick_kie",
        "deployed_to": str(PICK_KIE_MODEL_PATH),
        "backup": str(backup) if backup else None,
    }


def deploy_vietocr(artifact_root: Path) -> dict[str, Any]:
    """Replace models/vietocr/vgg_transformer.pth from Kaggle output tree (legacy zip names accepted)."""
    root = _find_artifact_root(artifact_root, "vietocr")
    weight_src = root / "vgg_transformer.pth"
    if not weight_src.is_file():
        weight_src = root / "vietocr_receipt.pth"
    if not weight_src.is_file():
        raise FileNotFoundError("vgg_transformer.pth missing in artifacts")

    backup_root = VIETOCR_ARTIFACTS_DIR / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = _backup_dir(VIETOCR_DIR, backup_root)

    VIETOCR_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weight_src, VIETOCR_WEIGHTS)
    for extra in ("config.yml", "meta.json"):
        src = root / extra
        if src.is_file():
            shutil.copy2(src, VIETOCR_DIR / extra)

    return {
        "job_type": "vietocr",
        "deployed_to": str(VIETOCR_WEIGHTS),
        "backup": str(backup) if backup else None,
    }


def update_manifest(deploy_info: dict[str, Any], batch_id: str | None = None) -> dict[str, Any]:
    OCR_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    if OCR_MANIFEST.is_file():
        manifest = json.loads(OCR_MANIFEST.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    job = deploy_info["job_type"]
    if job in ("pick_kie", "pick_retrain", "pick_train"):
        manifest["pick_kie"] = "models/pick_kie/model_best.pth"
        manifest["kie_backend"] = "pick"
    else:
        manifest["vietocr_weights"] = "models/vietocr/vgg_transformer.pth"
    manifest["updated_at"] = now
    if batch_id:
        manifest["trained_on_batch"] = batch_id
    OCR_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def deploy_from_source(
    source: str | Path,
    job_type: str,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Deploy from local dir, zip, or cloud URL. Returns deploy report."""
    extracted, tmp = prepare_source(source)
    try:
        if job_type in ("pick_kie", "pick_retrain", "pick_train"):
            deploy_info = deploy_pick_kie(extracted)
        elif job_type == "vietocr":
            deploy_info = deploy_vietocr(extracted)
        else:
            raise ValueError(f"Unknown job_type: {job_type}")
        manifest = update_manifest(deploy_info, batch_id)
        return {"ok": True, "deploy": deploy_info, "manifest": manifest}
    finally:
        if tmp is not None:
            tmp.cleanup()


def rollback(job_type: str, backup_path: str | Path) -> dict[str, Any]:
    """Restore models from a backup folder created during deploy."""
    backup = Path(backup_path)
    if not backup.is_dir():
        raise FileNotFoundError(f"Backup not found: {backup}")
    target = PICK_KIE_DIR if job_type in ("pick_kie", "pick_retrain", "pick_train") else VIETOCR_DIR
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(backup, target)
    return {"ok": True, "restored_from": str(backup), "target": str(target)}
