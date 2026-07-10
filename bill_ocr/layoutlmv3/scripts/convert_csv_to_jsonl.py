#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import json
import ast
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert MC_OCR CSV annotations to LayoutLMv3 JSONL format")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--out", type=str, required=True, help="Directory to save the output JSONL file")
    return parser.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file name based on input csv name
    out_name = csv_path.stem + ".jsonl"
    out_path = out_dir / out_name
    
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Determine the format of the CSV
    has_raw_columns = all(col in df.columns for col in ["img_id", "anno_polygons", "anno_texts", "anno_labels"])
    
    records = []
    
    if has_raw_columns:
        print("Detected raw MC_OCR CSV format. Converting to word-level IOB format...")
        for _, row in df.iterrows():
            img_id = row["img_id"]
            image_path = str(csv_path.parent / "imgs" / img_id)
            
            # Skip if image_path doesn't exist to prevent crashes (optional, but let's keep it robust)
            # We don't strictly assert existence here as it might be evaluated in container pathing,
            # but we assume the directory structure has imgs/<img_id>
            
            try:
                polygons = ast.literal_eval(row["anno_polygons"])
            except Exception as e:
                print(f"Warning: failed to parse polygons for {img_id}: {e}")
                continue
                
            texts = str(row["anno_texts"]).split("|||")
            labels = str(row["anno_labels"]).split("|||")
            
            # Ensure list lengths align
            n_segments = min(len(polygons), len(texts), len(labels))
            
            words_all = []
            boxes_all = []
            labels_all = []
            
            for i in range(n_segments):
                poly = polygons[i]
                text = texts[i]
                label = labels[i]
                
                # Extract bbox [x, y, w, h] and size
                bbox = poly.get("bbox", [0, 0, 0, 0])
                img_width = poly.get("width", 1)
                img_height = poly.get("height", 1)
                
                x, y, w, h = bbox
                x0, y0, x1, y1 = x, y, x + w, y + h
                
                # Normalize coordinates to 0-1000
                x0_norm = max(0, min(1000, int(1000 * x0 / img_width)))
                y0_norm = max(0, min(1000, int(1000 * y0 / img_height)))
                x1_norm = max(0, min(1000, int(1000 * x1 / img_width)))
                y1_norm = max(0, min(1000, int(1000 * y1 / img_height)))
                
                norm_box = [x0_norm, y0_norm, x1_norm, y1_norm]
                
                # Split segment text into individual words
                seg_words = text.split()
                if not seg_words:
                    continue
                    
                # Format labels to IOB format
                for idx, word in enumerate(seg_words):
                    words_all.append(word)
                    boxes_all.append(norm_box)
                    
                    # Auto-correct common dataset label typos
                    clean_label = str(label).strip()
                    if clean_label == "TIMESTAMPS":
                        clean_label = "TIMESTAMP"
                    elif clean_label == "TOTAL_TOTAL_COST":
                        clean_label = "TOTAL_COST"

                    if clean_label == "O" or not clean_label:
                        labels_all.append("O")
                    else:
                        prefix = "B-" if idx == 0 else "I-"
                        labels_all.append(prefix + clean_label)
            
            if words_all:
                records.append({
                    "image": image_path,
                    "words": words_all,
                    "boxes": boxes_all,
                    "labels": labels_all
                })

def run_ocr_and_match_labels(image_path, csv_path, polygons, texts, labels):
    # Resolve the image file path
    img_file = Path(image_path)
    if not img_file.is_file():
        img_file = csv_path.parent / "imgs" / Path(image_path).name
        if not img_file.is_file():
            img_file = csv_path.parent / Path(image_path).name
            
    if not img_file.is_file():
        return None  # Signal fallback to old method
        
    # Dynamically import and initialize OCR pipeline
    global _ocr_pipeline
    if "_ocr_pipeline" not in globals():
        import sys
        bill_ocr_path = Path(__file__).resolve().parents[2]
        if str(bill_ocr_path) not in sys.path:
            sys.path.insert(0, str(bill_ocr_path))
        from receipt_ocr.pipeline import ReceiptOCRPipeline
        from receipt_ocr.model_paths import resolve_vietocr_weights_path
        
        vietocr_w = resolve_vietocr_weights_path(None)
        print(f"Initializing ReceiptOCRPipeline for training matching with weights: {vietocr_w}...")
        _ocr_pipeline = ReceiptOCRPipeline(vietocr_weights=vietocr_w, paddle_use_gpu=False).load()

    import cv2
    import numpy as np
    try:
        image_rgb = _ocr_pipeline._read_rgb(img_file)
        h_img, w_img = image_rgb.shape[:2]
    except Exception as e:
        print(f"Warning: failed to read image {img_file}: {e}")
        return None

    df_boxes = _ocr_pipeline.ocr_boxes(image_rgb)
    if df_boxes is None or df_boxes.empty:
        return [], [], []
        
    # Prep ground truth boxes
    n_segments = min(len(polygons), len(texts), len(labels))
    gt_segments = []
    for i in range(n_segments):
        poly = polygons[i]
        bbox = poly.get("bbox", [0, 0, 0, 0])
        img_w = poly.get("width", 1)
        img_h = poly.get("height", 1)
        
        # Scale coordinates if ground truth image size differs from OCR image size
        x, y, w, h = bbox
        scale_x = w_img / img_w
        scale_y = h_img / img_h
        
        gx0, gy0, gx1, gy1 = x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y
        
        # Auto-correct common label typos
        clean_label = str(labels[i]).strip()
        if clean_label == "TIMESTAMPS":
            clean_label = "TIMESTAMP"
        elif clean_label == "TOTAL_TOTAL_COST":
            clean_label = "TOTAL_COST"
            
        gt_segments.append({
            "box": [gx0, gy0, gx1, gy1],
            "label": clean_label,
            "text": texts[i],
            "matched_count": 0
        })
        
    words_all = []
    boxes_all = []
    labels_all = []
    
    for _, row in df_boxes.iterrows():
        text = row["text"]
        ox0, oy0, ox1, oy1 = row["x1"], row["y1"], row["x2"], row["y2"]
        
        # Find best matching ground truth segment
        best_gt_idx = -1
        best_overlap_ratio = 0.0
        
        ocr_area = max(1.0, (ox1 - ox0) * (oy1 - oy0))
        for j, gt in enumerate(gt_segments):
            gx0, gy0, gx1, gy1 = gt["box"]
            
            # Intersection area
            ix0 = max(ox0, gx0)
            iy0 = max(oy0, gy0)
            ix1 = min(ox1, gx1)
            iy1 = min(oy1, gy1)
            
            if ix1 > ix0 and iy1 > iy0:
                intersection_area = (ix1 - ix0) * (iy1 - iy0)
                overlap_ratio = intersection_area / ocr_area
                if overlap_ratio > best_overlap_ratio:
                    best_overlap_ratio = overlap_ratio
                    best_gt_idx = j
                    
        # Map label
        final_label = "O"
        if best_gt_idx != -1 and best_overlap_ratio >= 0.4:
            gt = gt_segments[best_gt_idx]
            if gt["label"] != "O" and gt["label"]:
                prefix = "B-" if gt["matched_count"] == 0 else "I-"
                final_label = prefix + gt["label"]
                gt["matched_count"] += 1
                
        # Normalize box
        x0_norm = max(0, min(1000, int(1000 * ox0 / w_img)))
        y0_norm = max(0, min(1000, int(1000 * oy0 / h_img)))
        x1_norm = max(0, min(1000, int(1000 * ox1 / w_img)))
        y1_norm = max(0, min(1000, int(1000 * oy1 / h_img)))
        norm_box = [x0_norm, y0_norm, x1_norm, y1_norm]
        
        seg_words = str(text).split()
        for idx, word in enumerate(seg_words):
            words_all.append(word)
            boxes_all.append(norm_box)
            # For multi-word OCR boxes, prefix matching: first word B-, rest I-
            if final_label.startswith("B-"):
                labels_all.append("B-" + final_label[2:] if idx == 0 else "I-" + final_label[2:])
            else:
                labels_all.append(final_label)
                
    return words_all, boxes_all, labels_all

def run_ocr_for_image(image_path, csv_path):
    # Resolve the image file path
    img_file = Path(image_path)
    if not img_file.is_file():
        img_file = csv_path.parent / "imgs" / image_path
        if not img_file.is_file():
            img_file = csv_path.parent / "test_images" / image_path
            if not img_file.is_file():
                img_file = csv_path.parent / image_path
    
    if not img_file.is_file():
        print(f"Warning: Image file not found: {image_path}")
        return [], [], []

    # Dynamically import and initialize OCR pipeline to avoid importing if not needed
    global _ocr_pipeline
    if "_ocr_pipeline" not in globals():
        import sys
        bill_ocr_path = Path(__file__).resolve().parents[2]
        if str(bill_ocr_path) not in sys.path:
            sys.path.insert(0, str(bill_ocr_path))
        from receipt_ocr.pipeline import ReceiptOCRPipeline
        from receipt_ocr.model_paths import resolve_vietocr_weights_path
        
        vietocr_w = resolve_vietocr_weights_path(None)
        print(f"Initializing ReceiptOCRPipeline with weights: {vietocr_w}...")
        _ocr_pipeline = ReceiptOCRPipeline(vietocr_weights=vietocr_w, paddle_use_gpu=False).load()

    import cv2
    import numpy as np
    image_rgb = _ocr_pipeline._read_rgb(img_file)
    h_img, w_img = image_rgb.shape[:2]
    
    df_boxes = _ocr_pipeline.ocr_boxes(image_rgb)
    
    words_all = []
    boxes_all = []
    labels_all = []
    
    if df_boxes is not None and not df_boxes.empty:
        for _, row in df_boxes.iterrows():
            text = row["text"]
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            
            # Normalize coordinates to 0-1000
            x0_norm = max(0, min(1000, int(1000 * x1 / w_img)))
            y0_norm = max(0, min(1000, int(1000 * y1 / h_img)))
            x1_norm = max(0, min(1000, int(1000 * x2 / w_img)))
            y1_norm = max(0, min(1000, int(1000 * y2 / h_img)))
            norm_box = [x0_norm, y0_norm, x1_norm, y1_norm]
            
            seg_words = str(text).split()
            for word in seg_words:
                words_all.append(word)
                boxes_all.append(norm_box)
                labels_all.append("O")
                
    return words_all, boxes_all, labels_all

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file name based on input csv name
    out_name = csv_path.stem + ".jsonl"
    out_path = out_dir / out_name
    
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Determine the format of the CSV
    has_raw_columns = all(col in df.columns for col in ["img_id", "anno_polygons", "anno_texts", "anno_labels"])
    
    records = []
    
    if has_raw_columns:
        print("Detected raw MC_OCR CSV format. Converting to word-level IOB format...")
        for _, row in df.iterrows():
            img_id = row["img_id"]
            image_path = str(csv_path.parent / "imgs" / img_id)
            
            try:
                polygons = ast.literal_eval(row["anno_polygons"])
            except Exception as e:
                print(f"Warning: failed to parse polygons for {img_id}: {e}")
                continue
                
            texts = str(row["anno_texts"]).split("|||")
            labels = str(row["anno_labels"]).split("|||")
            
            # 1. Try OCR matching first to include background/O tokens
            ocr_res = run_ocr_and_match_labels(image_path, csv_path, polygons, texts, labels)
            if ocr_res is not None:
                words_all, boxes_all, labels_all = ocr_res
            else:
                # Fallback to old behavior if image file is not found
                n_segments = min(len(polygons), len(texts), len(labels))
                words_all = []
                boxes_all = []
                labels_all = []
                
                for i in range(n_segments):
                    poly = polygons[i]
                    text = texts[i]
                    label = labels[i]
                    
                    bbox = poly.get("bbox", [0, 0, 0, 0])
                    img_width = poly.get("width", 1)
                    img_height = poly.get("height", 1)
                    
                    x, y, w, h = bbox
                    x0, y0, x1, y1 = x, y, x + w, y + h
                    
                    x0_norm = max(0, min(1000, int(1000 * x0 / img_width)))
                    y0_norm = max(0, min(1000, int(1000 * y0 / img_height)))
                    x1_norm = max(0, min(1000, int(1000 * x1 / img_width)))
                    y1_norm = max(0, min(1000, int(1000 * y1 / img_height)))
                    norm_box = [x0_norm, y0_norm, x1_norm, y1_norm]
                    
                    seg_words = text.split()
                    for idx, word in enumerate(seg_words):
                        words_all.append(word)
                        boxes_all.append(norm_box)
                        
                        clean_label = str(label).strip()
                        if clean_label == "TIMESTAMPS":
                            clean_label = "TIMESTAMP"
                        elif clean_label == "TOTAL_TOTAL_COST":
                            clean_label = "TOTAL_COST"

                        if clean_label == "O" or not clean_label:
                            labels_all.append("O")
                        else:
                            prefix = "B-" if idx == 0 else "I-"
                            labels_all.append(prefix + clean_label)
            
            if words_all:
                records.append({
                    "image": image_path,
                    "words": words_all,
                    "boxes": boxes_all,
                    "labels": labels_all
                })
                
    else:
        # Fallback to pre-processed format if present
        print("Detected alternative/pre-processed CSV format...")
        
        # Check if words and boxes are present as columns and not empty
        is_labeled = "words" in df.columns and ("bbox" in df.columns or "boxes" in df.columns)
        
        for _, row in df.iterrows():
            # Support columns: image_path, bbox, label
            image_path_raw = row.get("image_path", row.get("image", row.get("img_id", "")))
            if not image_path_raw:
                continue
                
            if not is_labeled:
                # Dynamically run OCR to get words and boxes
                words, boxes, labels = run_ocr_for_image(image_path_raw, csv_path)
                resolved_img_path = image_path_raw
                img_file = Path(image_path_raw)
                if not img_file.is_file():
                    img_file = csv_path.parent / "imgs" / image_path_raw
                    if not img_file.is_file():
                        img_file = csv_path.parent / "test_images" / image_path_raw
                        if not img_file.is_file():
                            img_file = csv_path.parent / image_path_raw
                if img_file.is_file():
                    resolved_img_path = str(img_file.resolve())
            else:
                resolved_img_path = image_path_raw
                # Expecting semicolon-separated or space-separated lists
                words = str(row.get("words", "")).split(";")
                
                boxes_raw = str(row.get("bbox", row.get("boxes", ""))).split(";")
                boxes = []
                for b in boxes_raw:
                    try:
                        boxes.append([int(coord) for coord in b.split(",")])
                    except:
                        boxes.append([0, 0, 0, 0])
                        
                labels = str(row.get("label", row.get("labels", ""))).split(";")
            
            if words:
                records.append({
                    "image": resolved_img_path,
                    "words": words,
                    "boxes": boxes,
                    "labels": labels
                })

    # Write out as JSONL
    print(f"Writing {len(records)} records to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
