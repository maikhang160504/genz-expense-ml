import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForTokenClassification

# Paths
DATA_ROOT = Path("/workspace/data")
VOLUME_CHECKPOINT = Path("/storage/layoutlmv3/model_best.pth")
PRIVATE_TEST_DIR = Path("/storage/mc_ocr_test")
OUTPUT_VIZ_DIR = Path("/storage/visualizations")
OUTPUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Colors mapping for entities
COLOR_MAP = {
    "SELLER": (230, 50, 50),       # Red
    "ADDRESS": (50, 200, 50),     # Green
    "TIMESTAMP": (50, 50, 230),    # Blue
    "TOTAL_COST": (230, 150, 30),  # Orange
}

def clean_single_tag(tag, token_str):
    return tag

def main():
    print("🚀 Initializing LayoutLMv3 visualization pipeline...")
    
    # 1. Use static predefined label list to match train_eval.py
    label_list = sorted(["O", "B-ADDRESS", "I-ADDRESS", "B-SELLER", "I-SELLER", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST"])
    label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    
    # 2. Load fine-tuned LayoutLMv3 Model
    if not VOLUME_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint model_best.pth not found at {VOLUME_CHECKPOINT}")
        
    print(f"📥 Loading model weights from: {VOLUME_CHECKPOINT}")
    checkpoint = torch.load(VOLUME_CHECKPOINT, map_location="cpu")
    
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    # 3. Load Receipt OCR Pipeline for dynamic OCR boxes
    import sys
    sys.path.insert(0, "/workspace/bill_ocr")
    from receipt_ocr.pipeline import ReceiptOCRPipeline
    from receipt_ocr.model_paths import resolve_vietocr_weights_path
    
    vietocr_w = resolve_vietocr_weights_path(None)
    ocr_pipeline = ReceiptOCRPipeline(vietocr_weights=vietocr_w, paddle_use_gpu=False).load()
    print("✅ OCR pipeline initialized!")

    # 4. Get target images to visualize
    test_img_dir = PRIVATE_TEST_DIR / "test_images"
    if not test_img_dir.is_dir():
        # Fall back to root private test images directory
        test_img_dir = PRIVATE_TEST_DIR
        
    csv_path = PRIVATE_TEST_DIR / "mcocr_test_df.csv"
    if not csv_path.is_file():
        # Fall back to listing directory
        img_files = list(test_img_dir.glob("*.jpg"))[:10]
    else:
        import pandas as pd
        df_test = pd.read_csv(csv_path)
        img_ids = df_test["img_id"].head(10).tolist()
        img_files = []
        for iid in img_ids:
            cand = test_img_dir / iid
            if cand.is_file():
                img_files.append(cand)
            else:
                # search recursively
                for p in test_img_dir.glob(f"**/{iid}"):
                    img_files.append(p)
                    break
        img_files = img_files[:10]

    print(f"🖼️ Found {len(img_files)} test images to process.")

    # 5. Process & Visualize
    for idx, img_path in enumerate(img_files):
        print(f"[{idx+1}/{len(img_files)}] Processing {img_path.name}...")
        
        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        w_img, h_img = img_pil.size
        
        # Run OCR
        image_cv = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        df_boxes = ocr_pipeline.ocr_boxes(image_rgb)
        
        if df_boxes is None or df_boxes.empty:
            print(f"⚠️ No text boxes found on image {img_path.name}, skipping.")
            continue
            
        words = []
        boxes = []
        raw_boxes = [] # unnormalized [x1, y1, x2, y2]
        
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
            for w in seg_words:
                words.append(w)
                boxes.append(norm_box)
                raw_boxes.append([x1, y1, x2, y2])
                
        if not words:
            continue
            
        # Encode inputs for LayoutLMv3
        encoding = processor(
            img_pil,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to device
        for k in encoding:
            encoding[k] = encoding[k].to(device)
            
        # Inference
        with torch.no_grad():
            outputs = model(**encoding)
            
        predictions = outputs.logits.argmax(-1).squeeze(0).cpu().numpy()
        input_ids = encoding["input_ids"].squeeze(0).cpu().numpy()
        
        # Map token predictions back to original words using word_ids
        word_predictions = {} # word_idx -> list of tags
        word_ids = encoding.word_ids(batch_index=0)
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(words):
                pred_tag = id2label[predictions[token_idx]]
                token_str = processor.tokenizer.decode([input_ids[token_idx]])
                # Clean tag using rule-based post-processor
                cleaned_tag = clean_single_tag(pred_tag, token_str)
                if word_idx not in word_predictions:
                    word_predictions[word_idx] = []
                word_predictions[word_idx].append(cleaned_tag)
                
        # Resolve single label per word (voting or selecting the B- prefix)
        resolved_word_labels = []
        for word_idx in range(len(words)):
            tags = word_predictions.get(word_idx, ["O"])
            # Filter out "O" if we have other entities predicted
            entity_tags = [t for t in tags if t != "O"]
            if entity_tags:
                # Prefer B- prefix, otherwise select the first non-O tag
                b_tags = [t for t in entity_tags if t.startswith("B-")]
                resolved_word_labels.append(b_tags[0] if b_tags else entity_tags[0])
            else:
                resolved_word_labels.append("O")
                
        # Apply PA2 Rule-based heuristic post-processing
        def post_process_predictions(words, raw_boxes, resolved_word_labels, w_img, h_img):
            refined_labels = list(resolved_word_labels)
            
            # 1. Clean SELLER: Must be in the top 40% of the image
            for idx, label in enumerate(refined_labels):
                if "SELLER" in label:
                    x1, y1, x2, y2 = raw_boxes[idx]
                    if y1 > h_img * 0.4:
                        refined_labels[idx] = "O"
                        
            # 2. Clean ADDRESS: Avoid table rows (usually middle 35%-75% height)
            for idx, label in enumerate(refined_labels):
                if "ADDRESS" in label:
                    x1, y1, x2, y2 = raw_boxes[idx]
                    word_lower = words[idx].lower()
                    if y1 > h_img * 0.35 and y1 < h_img * 0.75:
                        if word_lower in ["hộp", "gói", "sữa", "chai", "lon", "cái", "thịt", "cá", "rau", "vnd", "vnd:", "đ", "d"]:
                            refined_labels[idx] = "O"
                            
            # 3. Clean TIMESTAMP: Keep if it contains digits or is very close to a digit
            for idx, label in enumerate(refined_labels):
                if "TIMESTAMP" in label:
                    word_lower = words[idx].lower()
                    has_digit = any(c.isdigit() for c in word_lower)
                    if not has_digit:
                        nearby_has_digit = False
                        for offset in [-2, -1, 1, 2]:
                            n_idx = idx + offset
                            if 0 <= n_idx < len(words):
                                if any(c.isdigit() for c in words[n_idx]):
                                    nearby_has_digit = True
                                    break
                        if not nearby_has_digit:
                            refined_labels[idx] = "O"

            # 4. Clean TOTAL_COST:
            # Group TOTAL_COST candidates into rows to check context (e.g., exclude cash paid / change returned)
            total_cost_indices = [i for i, l in enumerate(refined_labels) if "TOTAL_COST" in l]
            if total_cost_indices:
                import re
                
                # Exclude lines containing cash/change keywords
                re_exclude = re.compile(
                    r"tien\s*mat|khach\s*dua|tien\s*khach|cash|received|tra\s*lai|tien\s*thua|thoi\s*lai|change|thoi|tra",
                    re.I
                )
                re_total = re.compile(
                    r"tong\s*thanh\s*toan|tien\s*thanh\s*toan|thuc\s*thu|thuc\s*tra|phai\s*thanh\s*toan|tong\s*cong|thanh\s*tien|tong\s*tien|total|cong|cộng",
                    re.I
                )
                re_discount = re.compile(
                    r"da\s*giam|đã\s*giảm|giam\s*gia|giảm\s*giá|discount|chiet\s*khau|chiết\s*khấu|km",
                    re.I
                )
                
                numeric_candidates = []
                for idx in total_cost_indices:
                    word = words[idx]
                    clean_word = word.replace(".", "").replace(",", "").replace("đ", "").replace("d", "").replace("vndi", "").strip()
                    if clean_word.isdigit():
                        y1 = raw_boxes[idx][1]
                        h_box = raw_boxes[idx][3] - raw_boxes[idx][1]
                        row_tol = max(15.0, h_box * 1.5)
                        
                        # Look for same-row keywords in all words of the receipt
                        same_row_words = []
                        for other_idx, w in enumerate(words):
                            if abs(raw_boxes[other_idx][1] - y1) <= row_tol:
                                same_row_words.append(w)
                        row_text = " ".join(same_row_words).lower()
                        
                        score = 100
                        # Add points for total labels
                        if re_total.search(row_text):
                            score += 500
                        # Penalize for cash/change/discount keywords
                        if re_exclude.search(row_text):
                            score -= 1000
                        if re_discount.search(row_text):
                            score -= 300
                            
                        # Lower position on the receipt is preferred
                        score += int(100 * y1 / h_img)
                        numeric_candidates.append((idx, score, y1, row_tol))
                
                if numeric_candidates:
                    # Select the highest-scoring candidate
                    numeric_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_numeric_idx = numeric_candidates[0][0]
                    y_total = raw_boxes[best_numeric_idx][1]
                    row_tol = numeric_candidates[0][3]
                    
                    # Keep only TOTAL_COST elements on the same row as the best candidate
                    for idx in total_cost_indices:
                        y_curr = raw_boxes[idx][1]
                        if abs(y_curr - y_total) > row_tol:
                            refined_labels[idx] = "O"
                else:
                    # If no digits found, keep the lowest TOTAL_COST word and same-line neighbors
                    total_cost_indices.sort(key=lambda idx: raw_boxes[idx][1], reverse=True)
                    best_word_idx = total_cost_indices[0]
                    y_total = raw_boxes[best_word_idx][1]
                    h_box = raw_boxes[best_word_idx][3] - raw_boxes[best_word_idx][1]
                    row_tol = max(15.0, h_box * 1.5)
                    
                    for idx in total_cost_indices:
                        y_curr = raw_boxes[idx][1]
                        if abs(y_curr - y_total) > row_tol:
                            refined_labels[idx] = "O"
                            
            return refined_labels

        resolved_word_labels = post_process_predictions(words, raw_boxes, resolved_word_labels, w_img, h_img)
                
        # Draw on image
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        for word_idx, label in enumerate(resolved_word_labels):
            if label == "O":
                continue
                
            entity_name = label.split("-")[-1]
            color = COLOR_MAP.get(entity_name, (128, 128, 128))
            
            x1, y1, x2, y2 = raw_boxes[word_idx]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            text_pos = [x1, max(0, y1 - 12)]
            draw.rectangle([text_pos[0], text_pos[1], text_pos[0] + len(entity_name)*7, text_pos[1] + 12], fill=color)
            draw.text(text_pos, entity_name, fill=(255, 255, 255), font=font)
            
        # Save visualized result
        out_file = OUTPUT_VIZ_DIR / f"viz_{img_path.name}"
        img_pil.save(out_file)
        print(f"💾 Visualized result saved to: {out_file}")
        
    print("\n🎉 Done! All visualizations saved successfully to /storage/visualizations.")

if __name__ == "__main__":
    main()
