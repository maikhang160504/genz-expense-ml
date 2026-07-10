"""PICK KIE + heuristic fallback for receipt field labeling (MC_OCR style)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_paths import LAYOUTLMV3_MODEL_PATH, PICK_KIE_MODEL_PATH
from .receipt_nlu import _amounts_in_line, _normalize, _parse_vn_amount

ENTITY_OTHER = "OTHER"
ENTITY_SELLER = "SELLER"
ENTITY_ADDRESS = "ADDRESS"
ENTITY_TIMESTAMP = "TIMESTAMP"
ENTITY_TOTAL = "TOTAL_COST"
KIE_ENTITIES = {ENTITY_SELLER, ENTITY_ADDRESS, ENTITY_TIMESTAMP, ENTITY_TOTAL, ENTITY_OTHER}

_RE_TOTAL = re.compile(
    r"tong\s*thanh\s*toan|tien\s*thanh\s*toan|thuc\s*thu|thuc\s*tra|phai\s*thanh\s*toan|"
    r"tong\s*cong|thanh\s*tien|tong\s*tien|total",
    re.I,
)
_RE_ADDRESS = re.compile(r"duong|phuong|quan|tp\.|thanh\s*pho|so\s*\d", re.I)
_RE_TIMESTAMP = re.compile(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}:\d{2}")
_RE_EXCLUDE = re.compile(
    r"tien\s*mat|khach\s*dua|tien\s*khach|cash|received|tra\s*lai|tien\s*thua|thoi\s*lai|change",
    re.I,
)
_RE_DISCOUNT = re.compile(
    r"da\s*giam|đã\s*giảm|giam\s*gia|giảm\s*giá|discount|chiet\s*khau|chiết\s*khấu|km\s*giam",
    re.I,
)
_RE_TOTAL_LABEL = re.compile(
    r"tong\s*thanh\s*toan|tien\s*thanh\s*toan|thuc\s*thu|thuc\s*tra|phai\s*thanh\s*toan|phai\s*t\.?\s*toan|"
    r"tong\s*cong|thanh\s*tien|tong\s*tien|t\.?\s*tien\b|total",
    re.I,
)


def default_kie_model_path() -> Path:
    return PICK_KIE_MODEL_PATH


def pick_kie_weights_status(model_path: str | Path | None = None) -> dict[str, Any]:
    """Check whether PICK model_best.pth is present on disk."""
    path = Path(model_path) if model_path else default_kie_model_path()
    ready = path.is_file()
    return {
        "model_path": str(path),
        "weights_found": ready,
        "weight_files": [path.name] if ready else [],
        "ready": ready,
    }


def _box_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "x1": int(row["x1"]),
        "y1": int(row["y1"]),
        "x2": int(row["x2"]),
        "y2": int(row["y2"]),
        "text": str(row.get("text", "")).strip(),
        "entity": str(row.get("entity", ENTITY_OTHER)),
        "confidence": float(row.get("confidence", 0.75)),
    }


def heuristic_label_boxes(df_boxes: pd.DataFrame) -> list[dict[str, Any]]:
    """Rule-based KIE when PICK weights are unavailable."""
    if df_boxes is None or df_boxes.empty:
        return []

    rows = df_boxes.sort_values(by=["y1", "x1"]).reset_index(drop=True)
    out: list[dict[str, Any]] = []
    n = len(rows)
    seller_assigned = False

    for idx, row in rows.iterrows():
        text = str(row["text"]).strip()
        low = _normalize(text)
        entity = ENTITY_OTHER
        conf = 0.72

        if not seller_assigned and idx < max(3, n // 8) and len(text) >= 3 and not _RE_TOTAL.search(low):
            if not _RE_TIMESTAMP.search(text) and not _amounts_in_line(text):
                entity = ENTITY_SELLER
                seller_assigned = True
                conf = 0.80
        elif _RE_TIMESTAMP.search(text):
            entity = ENTITY_TIMESTAMP
            conf = 0.78
        elif _RE_ADDRESS.search(low) and len(text) > 8:
            entity = ENTITY_ADDRESS
            conf = 0.70
        elif _RE_TOTAL.search(low) and not _RE_EXCLUDE.search(low) and not _RE_DISCOUNT.search(low):
            entity = ENTITY_TOTAL
            conf = 0.85 if _amounts_in_line(text) else 0.82
        elif (
            idx >= n - 5
            and _amounts_in_line(text)
            and not _RE_EXCLUDE.search(low)
            and not _RE_DISCOUNT.search(low)
            and not _RE_TOTAL_LABEL.search(low)
        ):
            entity = ENTITY_TOTAL
            conf = 0.75

        out.append(_box_row_to_dict({**row.to_dict(), "entity": entity, "confidence": conf}))
    return out


def extract_kie_fields(labeled_boxes: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge labeled boxes into header/footer field values."""
    fields: dict[str, Any] = {
        "SELLER": None,
        "ADDRESS": None,
        "TIMESTAMP": None,
        "TOTAL_COST": None,
    }
    chunks: dict[str, list[str]] = {k: [] for k in fields}

    for box in labeled_boxes:
        ent = box.get("entity", ENTITY_OTHER)
        text = str(box.get("text", "")).strip()
        if not text or ent not in chunks:
            continue
        chunks[ent].append(text)

    for key in fields:
        if chunks[key]:
            fields[key] = " ".join(chunks[key]).strip()

    if fields["TOTAL_COST"]:
        amounts = _amounts_in_line(fields["TOTAL_COST"])
        if amounts:
            fields["TOTAL_COST_VALUE"] = max(amounts)
        else:
            fields["TOTAL_COST_VALUE"] = _parse_vn_amount(fields["TOTAL_COST"])
    else:
        fields["TOTAL_COST_VALUE"] = None

    return fields


def _box_y_center(box: dict[str, Any]) -> float:
    return (float(box["y1"]) + float(box["y2"])) / 2.0


def _box_amount_matches(box: dict[str, Any], final_amount: int) -> bool:
    text = str(box.get("text", "")).strip()
    val = _parse_vn_amount(text)
    amounts = _amounts_in_line(text)
    return val == final_amount or final_amount in amounts


def _is_total_label_text(text: str) -> bool:
    low = _normalize(text)
    if _RE_DISCOUNT.search(low):
        return False
    return bool(_RE_TOTAL_LABEL.search(low)) and not _amounts_in_line(text)


def reconcile_total_cost_boxes(
    labeled_boxes: list[dict[str, Any]],
    final_amount: int | None,
) -> list[dict[str, Any]]:
    """Align TOTAL_COST: keep pay-total label + matching amount (same row); drop line-item/discount noise."""
    if not final_amount or not labeled_boxes:
        return labeled_boxes

    ys = [float(b["y1"]) for b in labeled_boxes] + [float(b["y2"]) for b in labeled_boxes]
    y_min, y_max = min(ys), max(ys)
    img_h = max(y_max - y_min, 1.0)
    row_tol = max(12.0, img_h * 0.035)

    label_boxes = [
        b for b in labeled_boxes if _is_total_label_text(str(b.get("text", "")))
    ]

    def row_alignment_score(box: dict[str, Any]) -> int:
        yc = _box_y_center(box)
        best = 0
        for lb in label_boxes:
            dy = abs(yc - _box_y_center(lb))
            if dy <= row_tol:
                best = max(best, 100)
            elif dy <= row_tol * 2:
                best = max(best, 55)
        return best

    for box in labeled_boxes:
        if box.get("entity") != ENTITY_TOTAL:
            continue
        text = str(box.get("text", "")).strip()
        if _RE_DISCOUNT.search(_normalize(text)):
            box["entity"] = ENTITY_OTHER
            continue
        if _amounts_in_line(text) and not _box_amount_matches(box, final_amount):
            box["entity"] = ENTITY_OTHER

    candidates: list[tuple[int, int]] = []
    for idx, box in enumerate(labeled_boxes):
        text = str(box.get("text", "")).strip()
        if not _amounts_in_line(text) or not _box_amount_matches(box, final_amount):
            continue
        if _RE_DISCOUNT.search(_normalize(text)):
            continue

        score = row_alignment_score(box)
        yc = _box_y_center(box)
        if yc >= y_min + img_h * 0.5:
            score += 40
        if yc >= y_min + img_h * 0.65:
            score += 25
        x_center = (float(box["x1"]) + float(box["x2"])) / 2.0
        x_max = max(float(b["x2"]) for b in labeled_boxes)
        if x_center >= x_max * 0.55:
            score += 15
        if box.get("entity") == ENTITY_TOTAL:
            score += 5
        candidates.append((score, idx))

    if not candidates:
        return labeled_boxes

    candidates.sort(key=lambda x: (-x[0], -x[1]))
    best_idx = candidates[0][1]
    best_yc = _box_y_center(labeled_boxes[best_idx])

    labeled_boxes[best_idx]["entity"] = ENTITY_TOTAL
    for idx, box in enumerate(labeled_boxes):
        if idx == best_idx:
            continue
        text = str(box.get("text", "")).strip()
        if _is_total_label_text(text) and abs(_box_y_center(box) - best_yc) <= row_tol:
            box["entity"] = ENTITY_TOTAL
        elif box.get("entity") == ENTITY_TOTAL:
            box["entity"] = ENTITY_OTHER

    return labeled_boxes


class PickKIEEngine:
    """LayoutLMv3 KIE model; falls back to heuristics until model is deployed."""

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        self.model_path = Path(model_path) if model_path else LAYOUTLMV3_MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._backend = "heuristic"
        self._load_error: str | None = None
        self.label_list = sorted(["O", "B-ADDRESS", "I-ADDRESS", "B-SELLER", "I-SELLER", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST"])
        self.label2id = {lbl: idx for idx, lbl in enumerate(self.label_list)}
        self.id2label = {idx: lbl for lbl, idx in self.label2id.items()}

    @property
    def backend(self) -> str:
        return self._backend

    def load(self) -> PickKIEEngine:
        # Check standard checkpoints path or custom path
        ready = self.model_path.is_file()
        if not ready:
            # Fallback path mapping inside container volume
            volume_path = Path("/storage/layoutlmv3/model_best.pth")
            if volume_path.is_file():
                self.model_path = volume_path
                ready = True
                
        if not ready:
            self._backend = "heuristic"
            self._load_error = f"No LayoutLMv3 weights found at {self.model_path} or /storage."
            return self

        try:
            from transformers import AutoProcessor, AutoModelForTokenClassification
            import torch
            
            self._processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            self._model = AutoModelForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base",
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
            )
            checkpoint = torch.load(self.model_path, map_location="cpu")
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self.device)
            self._model.eval()
            self._backend = "layoutlmv3"
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._processor = None
            self._backend = "heuristic"
            self._load_error = f"LayoutLMv3 loading failed: {exc}"
        return self

    def label_boxes(self, df_boxes: pd.DataFrame, image_rgb: np.ndarray | None = None) -> list[dict[str, Any]]:
        if df_boxes is None or df_boxes.empty:
            return []
        if self._backend == "layoutlmv3" and self._model is not None and image_rgb is not None:
            try:
                return self._predict_layoutlmv3(df_boxes, image_rgb)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: LayoutLMv3 prediction failed, falling back to heuristics: {e}")
                return heuristic_label_boxes(df_boxes)
        return heuristic_label_boxes(df_boxes)

    def _predict_layoutlmv3(self, df_boxes: pd.DataFrame, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        from PIL import Image
        import torch
        img_pil = Image.fromarray(image_rgb).convert("RGB")
        w_img, h_img = img_pil.size
        
        words = []
        boxes = []
        raw_boxes = []
        
        for _, row in df_boxes.iterrows():
            text = row["text"]
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            
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
            return heuristic_label_boxes(df_boxes)
            
        processor = self._processor
        model = self._model
        
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
            encoding[k] = encoding[k].to(self.device)
            
        with torch.no_grad():
            outputs = model(**encoding)
            
        predictions = outputs.logits.argmax(-1).squeeze(0).cpu().numpy()
        word_ids = encoding.word_ids(batch_index=0)
        
        word_predictions = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(words):
                pred_tag = self.id2label[predictions[token_idx]]
                if word_idx not in word_predictions:
                    word_predictions[word_idx] = []
                word_predictions[word_idx].append(pred_tag)
                
        resolved_word_labels = []
        for word_idx in range(len(words)):
            tags = word_predictions.get(word_idx, ["O"])
            entity_tags = [t for t in tags if t != "O"]
            if entity_tags:
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
        
        out: list[dict[str, Any]] = []
        word_pointer = 0
        for _, row in df_boxes.iterrows():
            text = row["text"]
            seg_words = str(text).split()
            if not seg_words:
                out.append(_box_row_to_dict({**row.to_dict(), "entity": "O", "confidence": 0.75}))
                continue
                
            row_labels = []
            for _ in seg_words:
                if word_pointer < len(resolved_word_labels):
                    row_labels.append(resolved_word_labels[word_pointer])
                    word_pointer += 1
            
            clean_row_labels = []
            for rl in row_labels:
                if rl.startswith("B-") or rl.startswith("I-"):
                    clean_row_labels.append(rl[2:])
                else:
                    clean_row_labels.append(rl)
                    
            if clean_row_labels:
                from collections import Counter
                best_entity = Counter(clean_row_labels).most_common(1)[0][0]
            else:
                best_entity = "O"
                
            mapped_entity = ENTITY_OTHER if best_entity == "O" else best_entity
            out.append(_box_row_to_dict({**row.to_dict(), "entity": mapped_entity, "confidence": 0.90}))
            
        return out

import torch

_engine: PickKIEEngine | None = None

def reset_kie_engine() -> None:
    """Drop cached KIE engine so reload picks up new LayoutLMv3 weights."""
    global _engine
    _engine = None

def get_kie_engine(model_path: str | Path | None = None, device: str | None = None) -> PickKIEEngine:
    global _engine
    if _engine is None:
        _engine = PickKIEEngine(model_path=model_path, device=device).load()
    return _engine
