"""Build Kaggle upload package: WebAdmin verified labels for PICK retrain."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pick_export import export_verified_samples

DEFAULT_BASE_DATASET = "domixi1989/vietnamese-receipts-mc-ocr-2021"
INCREMENTAL_DIR = "incremental"
KAGGLE_UPLOAD_DIR = "kaggle_upload"


def build_training_pack(
    samples: list[dict[str, Any]],
    verified_root: str | Path,
    base_dataset: str = DEFAULT_BASE_DATASET,
) -> dict[str, Any]:
    """Export WebAdmin-approved samples for PICK KIE retrain on Kaggle."""
    root = Path(verified_root)
    incremental = root / INCREMENTAL_DIR
    if incremental.exists():
        shutil.rmtree(incremental)
    incremental.mkdir(parents=True, exist_ok=True)

    export_result = export_verified_samples(samples, incremental)

    images_dir = incremental / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, str]] = []
    copied = 0

    for sample in samples:
        sid = str(sample.get("id", "")).strip()
        src = sample.get("image_path") or sample.get("imagePath")
        src_path = Path(src) if src else None
        ext = sample.get("image_ext") or sample.get("imageExt") or (src_path.suffix if src_path else None) or ".jpg"
        dest_name = f"{sid}{ext if str(ext).startswith('.') else '.' + ext}"
        dest = images_dir / dest_name

        if src_path and src_path.is_file():
            shutil.copy2(src_path, dest)
            image_rows.append({"image": f"images/{dest_name}", "sample_id": sid, "split": "train"})
            copied += 1
        else:
            url = sample.get("image_url") or sample.get("imageUrl")
            if url:
                try:
                    import urllib.request as urllib_request
                    import os
                    from urllib.parse import urljoin
                    
                    # Resolve relative URLs (e.g. from local storage fallback)
                    if url.startswith("/"):
                        base_url = os.environ.get("BILL_RETRAIN_WEBHOOK_URL") or "http://127.0.0.1:4000"
                        url = urljoin(base_url, url)
                    
                    # Tải từ Cloudflare R2 / Internet / Local Backend proxy
                    urllib_request.urlretrieve(url, str(dest))
                    image_rows.append({"image": f"images/{dest_name}", "sample_id": sid, "split": "train"})
                    copied += 1
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to download image for sample {sid} from {url}: {e}")

    image_list_path = incremental / "image_list.csv"
    lines = ["image,sample_id,split"]
    for row in image_rows:
        lines.append(f"{row['image']},{row['sample_id']},{row['split']}")
    image_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pack_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_dataset": base_dataset,
        "base_dataset_url": f"https://www.kaggle.com/datasets/{base_dataset}",
        "incremental_samples": export_result["manifest"]["count"],
        "incremental_images": copied,
        "merge_strategy": "kernel_attaches_base_plus_incremental_dataset",
        "notes": (
            "PICK KIE retrain: incremental/boxes_and_transcripts/*.tsv + images/. "
            "Kaggle kernel Add Data: domixi1989/vietnamese-receipts-mc-ocr-2021 "
            "+ webadmin-verified-receipts."
        ),
    }
    pack_path = incremental / "training_pack.json"
    pack_path.write_text(json.dumps(pack_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    upload_dir = root / KAGGLE_UPLOAD_DIR
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    shutil.copytree(incremental, upload_dir)

    slug = "webadmin-verified-receipts"
    user_note = "Set YOUR_USERNAME in dataset-metadata.json before first kaggle datasets create"
    dataset_meta = {
        "title": "WebAdmin Verified Receipt Labels",
        "id": f"YOUR_USERNAME/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["ocr", "receipt", "vietnamese", "pick"],
        "description": (
            f"Incremental admin-verified receipt labels for PICK KIE. "
            f"Merge with base dataset {base_dataset}. {user_note}"
        ),
    }
    (upload_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta, indent=2), encoding="utf-8"
    )

    return {
        **export_result,
        "incremental_dir": str(incremental),
        "kaggle_upload_dir": str(upload_dir),
        "training_pack": pack_meta,
        "training_pack_path": str(pack_path),
        "images_copied": copied,
        "image_list_path": str(image_list_path),
    }
