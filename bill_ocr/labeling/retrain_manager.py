"""
Retrain manager module.
Integrates verified labels into the dataset, runs the train data pre-processing steps,
updates PICK config paths to be local/relative, and prepares the training scripts.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import sys
BILL_OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BILL_OCR_ROOT))
from mc_ocr.config import (
    dataset as CONFIG_DATASET,
    kie_out_txt_dir as CONFIG_KIE_OUT_TXT_DIR,
    rot_out_img_dir as CONFIG_ROT_OUT_IMG_DIR,
    kie_boxes_transcripts as CONFIG_KIE_BOXES_TRANSCRIPTS
)

EXPORTED_DIR = BILL_OCR_ROOT / "exported"
PICK_DIR = BILL_OCR_ROOT / "mc_ocr" / "key_info_extraction" / "PICK"


def export_verified_to_dataset(image_name: str, verified_data: dict[str, Any]) -> None:
    """
    Export verified data to the format and directory expected by PICK training pipeline.
    Writes ICDAR .txt annotation file and copies image to the input image directory.
    """
    # 1. Copy image to rot_out_img_dir (where the training process expects it)
    src_img_path = EXPORTED_DIR / image_name
    dest_img_path = Path(CONFIG_ROT_OUT_IMG_DIR) / image_name
    
    if src_img_path.is_file():
        Path(CONFIG_ROT_OUT_IMG_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img_path, dest_img_path)
        print(f"[RETRAIN-MANAGER] Copied image to: {dest_img_path}")
    else:
        print(f"[RETRAIN-MANAGER] Warning: Source image {src_img_path} not found.")

    # 2. Format verified items/fields into ICDAR line strings
    # format: x1,y1,x2,y2,x3,y3,x4,y4,transcription,entity_name
    lines = []
    
    # We retrieve the raw bounding boxes with their updated entities from verified_data
    # If the user edited items/total cost/category, those are represented in boxes
    boxes = verified_data.get("boxes", [])
    
    for box in boxes:
        # Convert quad/coordinates to comma separated string
        # Bbox is typically list of coords: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # or flat list: [x1, y1, x2, y2, x3, y3, x4, y4]
        coords = box.get("bbox", [])
        if not coords:
            continue
        
        flat_coords = []
        if isinstance(coords[0], list):
            for pt in coords[:4]:
                flat_coords.extend([int(pt[0]), int(pt[1])])
        else:
            flat_coords = [int(c) for c in coords[:8]]
            
        while len(flat_coords) < 8:
            flat_coords.append(0)
            
        coord_str = ",".join(str(c) for c in flat_coords)
        text = box.get("text", "").strip()
        entity = box.get("entity", "OTHER").upper()
        
        # ICDAR line format
        lines.append(f"{coord_str},{text},{entity}")
        
    # Write to target annotation directory
    txt_name = Path(image_name).with_suffix(".txt").name
    dest_txt_path = Path(CONFIG_KIE_OUT_TXT_DIR) / txt_name
    Path(CONFIG_KIE_OUT_TXT_DIR).mkdir(parents=True, exist_ok=True)
    
    with open(dest_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"[RETRAIN-MANAGER] Exported ICDAR annotation file to: {dest_txt_path}")


def prepare_pick_config_for_training() -> None:
    """
    Reads PICK's config.json, replaces absolute path dataset settings with relative/local paths,
    and saves the file so it is ready for training.
    """
    config_path = PICK_DIR / "config.json"
    if not config_path.is_file():
        print(f"[RETRAIN-MANAGER] Config file not found: {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    kie_train_dir = os.path.dirname(CONFIG_KIE_OUT_TXT_DIR)
    
    # Update train_dataset args files_name & folders
    config["train_dataset"]["args"]["files_name"] = os.path.abspath(
        os.path.join(kie_train_dir, "train_list.csv")
    ).replace("\\", "/")
    config["train_dataset"]["args"]["boxes_and_transcripts_folder"] = os.path.abspath(
        CONFIG_KIE_BOXES_TRANSCRIPTS
    ).replace("\\", "/")
    config["train_dataset"]["args"]["images_folder"] = os.path.abspath(
        CONFIG_ROT_OUT_IMG_DIR
    ).replace("\\", "/")
    
    # Update validation_dataset args files_name & folders
    config["validation_dataset"]["args"]["files_name"] = os.path.abspath(
        os.path.join(kie_train_dir, "val_list.csv")
    ).replace("\\", "/")
    config["validation_dataset"]["args"]["boxes_and_transcripts_folder"] = os.path.abspath(
        CONFIG_KIE_BOXES_TRANSCRIPTS
    ).replace("\\", "/")
    config["validation_dataset"]["args"]["images_folder"] = os.path.abspath(
        CONFIG_ROT_OUT_IMG_DIR
    ).replace("\\", "/")
    
    # Write back config.json
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        
    print(f"[RETRAIN-MANAGER] Updated {config_path} with localized dataset paths.")


def trigger_create_train_data() -> None:
    """
    Executes the create_train_data.py script to split dataset, format boxes and transcripts.
    """
    create_data_script = BILL_OCR_ROOT / "mc_ocr" / "key_info_extraction" / "create_train_data.py"
    print(f"[RETRAIN-MANAGER] Running training data builder: {create_data_script}")
    
    # Use python executable or direct command depending on environment
    cmd = ["python", str(create_data_script)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(create_data_script.parent))
    if res.returncode == 0:
        print("[RETRAIN-MANAGER] Data building completed successfully.")
        print(res.stdout)
    else:
        print(f"[RETRAIN-MANAGER] Error building training data (code {res.returncode}):")
        print(res.stderr)


def reload_model_weights() -> None:
    """
    Reloads the latest model_best.pth weights to the production models directory.
    """
    src_weights = PICK_DIR / "saved" / "models" / "PICK_Default"
    # Find newest model_best.pth recursively inside PICK save directory
    best_pth = None
    max_mtime = 0.0
    for path in src_weights.glob("**/model_best.pth"):
        mtime = path.stat().st_mtime
        if mtime > max_mtime:
            max_mtime = mtime
            best_pth = path
            
    if best_pth:
        dest_weights_dir = BILL_OCR_ROOT / "models" / "pick_kie"
        dest_weights_dir.mkdir(parents=True, exist_ok=True)
        dest_weights = dest_weights_dir / "model_best.pth"
        shutil.copy2(best_pth, dest_weights)
        print(f"[RETRAIN-MANAGER] Reloaded new weights: {best_pth} -> {dest_weights}")
    else:
        print("[RETRAIN-MANAGER] No newly trained model_best.pth weights found to reload.")
