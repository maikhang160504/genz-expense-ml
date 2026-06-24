"""Regenerate PICK Kaggle notebooks — readable cells + pick-train-code dataset."""
from __future__ import annotations

import json
import re
from pathlib import Path

KERNELS = Path(__file__).resolve().parent
COMMON_SRC = KERNELS.parent / "pick_kaggle_common.py"
PICK_CODE_DATASET = "mainhatkhangb2205881/pick-train-code"
MCOCR_DATASET = "domixi1989/vietnamese-receipts-mc-ocr-2021"

CELL_MARKER = re.compile(r"^# === KAGGLE_CELL:\s*(\w+)\s*===$")

BOOTSTRAP = """
from __future__ import annotations

import ast
import json
import os
import random
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import torch

WORK = Path("/kaggle/working")
""".strip()

CELL_RUN_DEPS = """
pip_install_pick_deps()
install_allennlp_shim()
print("Deps OK")
""".strip()

CELL_RUN_DATASET = """
KAGGLE_ROOT = resolve_kaggle_mcocr_root()
TRAIN_CSV = resolve_mcocr_train_csv(KAGGLE_ROOT)
TRAIN_IMG = resolve_mcocr_train_images(KAGGLE_ROOT)
print("MC-OCR root:", KAGGLE_ROOT)
print("CSV:", TRAIN_CSV)
print("Images:", TRAIN_IMG, "jpg:", len(list(TRAIN_IMG.glob("*.jpg"))))
""".strip()

CELL_RUN_BUILD = """
stats = build_pick_dataset_from_mcocr_csv(
    TRAIN_CSV, TRAIN_IMG, WORK, train_ratio=0.92,
)
print("PICK dataset:", stats)
assert stats["train_samples"] > 0, "No train samples — check Add Data"
train_list = Path(stats["pick_root"]) / "train" / "train_list.csv"
val_list = Path(stats["pick_root"]) / "val" / "val_list.csv"
""".strip()

CELL_RUN_MERGE = """
wa = resolve_webadmin_root()
if wa:
    print("WebAdmin merged:", merge_webadmin_into_pick(Path(stats["pick_root"]), wa))
else:
    print("No WebAdmin dataset — MC-OCR base only")
""".strip()

CELL_RUN_PICK = """
pick_dir = setup_pick_train_dir(WORK)
print("PICK train code:", pick_dir)
assert (pick_dir / "train.py").is_file(), "Add Data: mainhatkhangb2205881/pick-train-code"
""".strip()

CELL_RUN_TRAIN = """
cfg = write_pick_config(
    pick_dir, train_list, val_list,
    epochs=int(os.environ.get("PICK_EPOCHS", "{epochs}")),
    batch_size=2,
    num_workers=2,
    run_id="{run_id}",
)
print("Config:", cfg)
run_pick_training(pick_dir, cfg, work_dir=WORK)
""".strip()

CELL_RUN_EXPORT = """
best = find_model_best(pick_dir)
assert best, "model_best.pth not found"
out = export_pick_artifacts(best, WORK / "pick_kie_artifacts")
print("Exported:", out)
""".strip()


def _split_common_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "_preamble"
    sections[current] = []
    for line in text.splitlines():
        match = CELL_MARKER.match(line.strip())
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        if current != "_preamble":
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if k != "_preamble" and v}


def _nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": cells,
    }


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def _code(text: str) -> dict:
    lines = [line + "\n" for line in text.splitlines()]
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": lines}


def _train_md(retrain: bool) -> str:
    if retrain:
        return (
            "# Retrain PICK KIE\n\n"
            "Notebook **self-contained** — mỗi cell là code đọc được (không embed file ẩn).\n\n"
            f"**Add Data:** `{MCOCR_DATASET}` · `{PICK_CODE_DATASET}` · "
            "`webadmin-verified-receipts` (optional)\n\n"
            "PICK upstream code từ Kaggle Dataset `pick-train-code` (= `OCR/vendor/pick`)."
        )
    return (
        "# Train PICK KIE — MC-OCR baseline\n\n"
        "Huấn luyện **bước 5 PICK** từ GT CSV. Bước 1–4 không chạy trên Kaggle.\n\n"
        f"**Add Data:** `{MCOCR_DATASET}` · `{PICK_CODE_DATASET}`\n\n"
        "Logic trong các cell bên dưới; code PICK train lấy từ dataset (không zip, không clone MC_OCR)."
    )


def _build_notebook(*, retrain: bool, epochs: int, run_id: str) -> list[dict]:
    sections = _split_common_sections(COMMON_SRC.read_text(encoding="utf-8"))
    cells: list[dict] = [_md(_train_md(retrain))]
    
    cells.append(_md("# 0. Bootstrap\nKhởi tạo môi trường, import các thư viện cơ bản."))
    cells.append(_code(BOOTSTRAP))
    
    section_titles = {
        "deps": "# 1. Dependency Helpers\nCác hàm cài đặt thư viện cần thiết (AllenNLP, v.v.).",
        "dataset": "# 2. Dataset Resolvers\nCác hàm tìm cấu trúc thư mục của dataset trên Kaggle.",
        "build_data": "# 3. Data Formatting Helpers\nCác hàm hỗ trợ convert MC-OCR dữ liệu sang PICK input format.",
        "pick_setup": "# 4. Code & Vendor Setup Helpers\nCác hàm copy và giải nén mã nguồn PICK.",
        "train": "# 5. Training Loop Config & Runners\nCác hàm cấu hình config.json và chạy train.py.",
        "export": "# 6. Export & Compression\nCác hàm nén kết quả sau train (model_best.pth) thành zip."
    }
    
    for key in ("deps", "dataset", "build_data", "pick_setup", "train", "export"):
        if key in sections:
            if key in section_titles:
                cells.append(_md(section_titles[key]))
            cells.append(_code(sections[key]))
            
    cells.append(_md("# --- Execution Steps ---"))
    
    cells.append(_md("## Step 1: Install Dependencies\nTự động cài đặt thư viện cần thiết."))
    cells.append(_code(CELL_RUN_DEPS))
    
    cells.append(_md("## Step 2: Resolve Datasets\nXác định đường dẫn của các dataset đầu vào trên Kaggle."))
    cells.append(_code(CELL_RUN_DATASET))
    
    cells.append(_md("## Step 3: Parse and Build PICK Dataset\nChuyển đổi dữ liệu MC-OCR sang định dạng PICK dataset."))
    cells.append(_code(CELL_RUN_BUILD))
    
    if retrain:
        cells.append(_md("## Step 3b: Merge WebAdmin Verified Data (Retrain only)\nTích hợp dữ liệu xác thực bổ sung từ WebAdmin."))
        cells.append(_code(CELL_RUN_MERGE))
        
    cells.append(_md("## Step 4: Setup PICK Code Source\nThiết lập thư mục code train PICK."))
    cells.append(_code(CELL_RUN_PICK))
    
    cells.append(_md("## Step 5: Start Model Training\nGhi file cấu hình config.json và chạy huấn luyện model PICK KIE."))
    cells.append(_code(CELL_RUN_TRAIN.format(epochs=epochs, run_id=run_id)))
    
    cells.append(_md("## Step 6: Export Artifacts\nĐóng gói trọng số tốt nhất cùng cấu hình thành file zip để download/webhook."))
    cells.append(_code(CELL_RUN_EXPORT))
    
    return cells


def _write_kernel_metadata(kernel_dir: Path, *, retrain: bool) -> None:
    datasets = [MCOCR_DATASET, PICK_CODE_DATASET]
    if retrain:
        datasets.append("mainhatkhangb2205881/webadmin-verified-receipts")
    meta = {
        "id": f"mainhatkhangb2205881/{'retrain-pick-kie' if retrain else 'train-pick-kie'}",
        "title": "retrain-pick-kie" if retrain else "train-pick-kie",
        "code_file": f"vietnamese-receipts-{'retrain' if retrain else 'train'}-pick-kie.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def main() -> None:
    specs = (
        ("train-pick-kie", "vietnamese-receipts-train-pick-kie.ipynb", False, 40, "kaggle"),
        ("retrain-pick-kie", "vietnamese-receipts-retrain-pick-kie.ipynb", True, 25, "retrain"),
    )
    for kernel_name, nb_name, retrain, epochs, run_id in specs:
        kernel_dir = KERNELS / kernel_name
        kernel_dir.mkdir(parents=True, exist_ok=True)
        nb_path = kernel_dir / nb_name
        cells = _build_notebook(retrain=retrain, epochs=epochs, run_id=run_id)
        nb_path.write_text(json.dumps(_nb(cells), indent=1), encoding="utf-8")
        _write_kernel_metadata(kernel_dir, retrain=retrain)
        print(f"Wrote {nb_path.name} ({len(cells)} cells)")
    print("Done — notebooks use readable cells + pick-train-code dataset")


if __name__ == "__main__":
    main()
