"""Trích xuất danh mục (NLU Record) + số tiền từ dòng OCR hóa đơn."""
from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .brand_routing import route_brand_category

# Nhãn danh mục chuẩn NLU Record (src/nlu/ner.py → CATEGORY_MAP values)
NLU_DEFAULT_CATEGORY = "Others"

# Từ khóa sản phẩm / ngữ cảnh trên hóa đơn (ưu tiên hơn tên cửa minimart/siêu thị)
NLU_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Food": (
        "an uong",
        "cà phê",
        "ca phe",
        "coffee",
        "cafe",
        "the coffee",
        "highlands",
        "phuc long",
        "starbucks",
        "tra ",
        "tra den",
        "keo",
        "kẹo",
        "banh",
        "bánh",
        "snack",
        "chocolate",
        "alpenliebe",
        "dynamite",
        "duong",
        "đường",
        "dau ",
        "dầu",
        "mam ",
        "mắm",
        "tuong",
        "tương",
        "kem ",
        "kem que",
        "sua ",
        "sữa",
        "bot ",
        "bột",
        "knorr",
        "pho",
        "bun",
        "com ",
        "cơm",
        "nuoc",
        "nước",
        "bia ",
        "do uong",
        "đồ uống",
        "thap cam",
        "trang tien",
        "cai lan",
        "meizan",
        "tinh luyen",
        "grabfood",
        "ship do an",
    ),
    "Essentials": (
        "bobby",
        "ta ",
        "tã",
        "giay",
        "giấy",
        "ve sinh",
        "vệ sinh",
        "rau ",
        "thit",
        "thịt",
        "gao",
        "gạo",
        "di cho",
        "đi chợ",
        "xa phong",
        "xà phòng",
        "nuoc giat",
        "nước giặt",
    ),
    "Shopping": (
        "durex",
        "bao cao su",
        "bao cao",
        "dien thoai",
        "điện thoại",
        "tai nghe",
        "may tinh",
        "máy tính",
        "laptop",
        "quần áo",
        "ao ",
        "áo ",
        "giay dep",
        "giày",
    ),
    "Transport": (
        "xang",
        "xăng",
        "grab bike",
        "grabbike",
        "vexere",
        "ve xe",
        "gui xe",
        "gửi xe",
        "bai xe",
    ),
    "Housing": (
        "dien ",
        "điện ",
        "nuoc ",
        "nước ",
        "internet",
        "wifi",
        "thue nha",
        "thuê nhà",
    ),
    "Health": (
        "thuoc",
        "thuốc",
        "nha thuoc",
        "nhà thuoc",
        "yte",
        "y tế",
    ),
    "Beauty": (
        "my pham",
        "mỹ phẩm",
        "son ",
        "kem duong",
        "dau goi",
        "dầu gội",
    ),
    "Entertainment": (
        "netflix",
        "spotify",
        "youtube",
        "xem phim",
        "cgv",
        "game",
    ),
    "Education": (
        "hoc phi",
        "học phí",
        "sach",
        "sách",
        "truong",
        "trường",
    ),
    "Social": (
        "qua tang",
        "quà tặng",
        "sinh nhat",
        "sinh nhật",
    ),
}

_HEADER_SKIP = re.compile(
    r"hoa don|hóa đơn|tel|website|mst|ma so thue|so gd|thu ngan|xin cam",
    re.I,
)

_AMOUNT_TOKEN = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+|\d{4,9})(?![\d.,])"
)

_TOTAL_LINE_RULES: tuple[tuple[int, re.Pattern[str]], ...] = (
    (100, re.compile(r"cộng\s*tiền\s*hàng|cộng\s*tien\s*hang", re.I)),
    (98, re.compile(r"tổng\s*tiền\s*phải|phải\s*t\.?\s*toán|phai\s*t\.?\s*toan", re.I)),
    (95, re.compile(r"tổng\s*cộng|tong\s*cong", re.I)),
    (92, re.compile(r"thành\s*tiền|thanh\s*tien|t\.?\s*ti[eê]n\b", re.I)),
    (88, re.compile(r"tổng\s*tiền|tong\s*tien", re.I)),
    (80, re.compile(r"tiền\s*khách\s*trả|tien\s*khach\s*tra", re.I)),
    (70, re.compile(r"thanh\s*toán|thanh\s*toan", re.I)),
)

_SKIP_LINE = re.compile(
    r"đơn\s*giá|don\s*gia|mã\s*sp|barcode|số\s*gd|so\s*gd|mst|mã\s*số\s*thuế",
    re.I,
)


def _train_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_train_on_path() -> None:
    root = str(_train_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _nlu_category_map() -> dict[str, str]:
    _ensure_train_on_path()
    try:
        from src.nlu.ner import CATEGORY_MAP

        return dict(CATEGORY_MAP)
    except ImportError:
        return {
            "ăn uống": "Food",
            "cà phê": "Food",
            "mua sắm": "Shopping",
            "đi chợ": "Essentials",
            "giải trí": "Entertainment",
            "linh tinh": "Others",
        }


def to_nlu_record_category(label: str | None) -> str:
    """Chuẩn hóa nhãn Việt/Anh → nhãn Record NLU (Food, Shopping, …)."""
    if not label:
        return NLU_DEFAULT_CATEGORY
    cmap = _nlu_category_map()
    valid = {
        "Food", "Essentials", "Social", "Transport", "Shopping", "Housing", 
        "Health", "Beauty", "Education", "Entertainment", "Investment", "Others"
    }
    if label in valid:
        return label
    _ensure_train_on_path()
    try:
        from src.nlu.ner import normalize_text

        key = normalize_text(str(label))
        pred = _predict_category_model([key])
        if pred in valid:
            return pred
        mapped = cmap.get(key)
        if mapped:
            return mapped
    except Exception:
        pass
    key = _normalize(str(label))
    for vi, en in cmap.items():
        if vi in key or key in vi:
            return en
    return NLU_DEFAULT_CATEGORY


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def _fix_ocr_amount_typos(line: str) -> str:
    return re.sub(
        r"(\d{1,7}),\((\d{2})(?!\d)",
        lambda m: f"{m.group(1)},{m.group(2)}0",
        line,
    )


def _parse_vn_amount(token: str) -> int | None:
    token = (token or "").strip()
    if not token or not re.search(r"\d", token):
        return None

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "." in token:
        parts = token.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            token = "".join(parts)
        elif len(parts) == 2 and len(parts[1]) == 3:
            token = parts[0] + parts[1]
    elif "," in token:
        parts = token.split(",")
        if len(parts) > 1 and len(parts[-1]) == 3:
            token = "".join(parts)
        else:
            token = token.replace(",", "")

    if not token.isdigit():
        return None
    return int(token)


def _is_noise_amount(value: int, line: str, raw_token: str) -> bool:
    if value < 1000:
        return True
    if value > 50_000_000:
        return True
    digits = str(value)
    if len(digits) >= 8 and "," not in raw_token and "." not in raw_token:
        return True
    if re.search(r"số\s*h[dđ]|so\s*h[dđ]|số\s*gd|so\s*gd", line, re.I) and len(digits) >= 10:
        return True
    if len(digits) >= 9 and digits.startswith("0"):
        return True
    return False


def _item_line_amount(line: str) -> int | None:
    m = re.search(r"([\d.,]+)\s+([\d.,]+)\s*$", line.strip())
    if not m:
        return None
    a1, a2 = _parse_vn_amount(m.group(1)), _parse_vn_amount(m.group(2))
    if a1 and a2 and a1 == a2 and a1 >= 1000:
        return a1
    return None


def _amounts_in_line(line: str) -> list[int]:
    line = _fix_ocr_amount_typos(line)
    out: list[int] = []
    for m in _AMOUNT_TOKEN.finditer(line):
        raw = m.group(1)
        val = _parse_vn_amount(raw)
        if val is None or _is_noise_amount(val, line, raw):
            continue
        out.append(val)
    return out


def _line_total_score(line: str) -> int:
    if _SKIP_LINE.search(line):
        return 0
    low = _normalize(line)
    if "tien tra lai" in low or "tiền trả lại" in low:
        return 30
    best = 0
    for score, pat in _TOTAL_LINE_RULES:
        if pat.search(line):
            best = max(best, score)
    return best


def _pick_amount_on_total_line(amounts: list[int], line: str) -> int | None:
    if not amounts:
        return None
    low = _normalize(line)
    counts = Counter(amounts)

    repeated = [a for a, n in counts.items() if n >= 2]
    if repeated:
        return max(repeated)

    if ("phai" in low or "phải" in low) and ("toan" in low or "toán" in low):
        mid = sorted(amounts)
        if len(mid) >= 2 and mid[-1] >= mid[-2] * 1.4:
            return mid[-2]
    if "tien khach tra" in low.replace("ệ", "e"):
        without_huge = [a for a in amounts if a <= 500_000]
        if without_huge:
            return max(without_huge)

    return max(amounts)


def extract_total_amount(lines: list[str]) -> int | None:
    candidates: list[tuple[int, int]] = []

    for line in reversed(lines):
        score = _line_total_score(line)
        if score == 0:
            continue
        amounts = _amounts_in_line(line)
        picked = _pick_amount_on_total_line(amounts, line)
        if picked is not None:
            candidates.append((score, picked))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        best_score, best_amt = candidates[0]
        if best_score >= 90:
            return best_amt
        item_vals = [_item_line_amount(ln) for ln in lines]
        item_vals = [v for v in item_vals if v]
        if item_vals and best_score <= 80:
            if best_amt not in item_vals or max(item_vals) < best_amt:
                return max(item_vals)
        return best_amt

    item_totals: list[int] = []
    item_pat = re.compile(r"^\s*\d+\s+[\d.,]+\s+([\d.,]+)\s*$")
    for line in lines:
        val = _item_line_amount(line)
        if val:
            item_totals.append(val)
            continue
        m = item_pat.match(line.strip())
        if m:
            val = _parse_vn_amount(m.group(1))
            if val and val >= 1000:
                item_totals.append(val)

    if item_totals:
        return sum(item_totals)

    tail = lines[len(lines) // 2 :] if lines else lines
    tail_amounts: list[int] = []
    for line in tail:
        tail_amounts.extend(_amounts_in_line(line))
    if tail_amounts:
        return max(tail_amounts)

    return None


def _score_categories_from_lines(lines: list[str]) -> dict[str, int]:
    scores: dict[str, int] = {cat: 0 for cat in NLU_CATEGORY_KEYWORDS}
    blob = _normalize(" ".join(lines))

    for cat, keywords in NLU_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in blob:
                scores[cat] += 1

    for line in lines:
        if _HEADER_SKIP.search(line):
            continue
        low = _normalize(line)
        for cat, keywords in NLU_CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in low:
                    scores[cat] += 3

    return scores


def _predict_category_model(lines: list[str]) -> str | None:
    """Gọi category model NLU (nếu có) trên mô tả hóa đơn."""
    try:
        _ensure_train_on_path()
        from pipeline.text_preprocessing import clean_category_text
        from src.nlu.encoder_runtime import predict_category_encoder
        from src.nlu.models import load_category_model

        text = clean_category_text(" ".join(lines)[:600])
        if not text.strip():
            return None
        model = load_category_model()
        if model.get("backend") == "encoder" and model.get("bundle"):
            raw = predict_category_encoder(model["bundle"], text)
            return to_nlu_record_category(raw)
        if model.get("backend") == "tfidf" and model.get("vectorizer"):
            vec = model["vectorizer"].transform([text])
            raw = model["model"].predict(vec)[0]
            return to_nlu_record_category(str(raw))
    except Exception:
        return None
    return None


def classify_category(lines: list[str]) -> str:
    """
    Trả nhãn Record NLU: Food, Shopping, Essentials, …
    Ưu tiên từ khóa sản phẩm (kẹo, đường, …) hơn tên cửa (minimart, siêu thị).
    """
    scores = _score_categories_from_lines(lines)
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score >= 3:
        return best

    model_cat = _predict_category_model(lines)
    if model_cat and model_cat != NLU_DEFAULT_CATEGORY:
        if best_score >= 2 and scores[best] > scores.get(model_cat, 0):
            return best
        if best_score < 2:
            return model_cat

    if best_score > 0:
        return best

    blob = _normalize(" ".join(lines[:6]))
    if any(x in blob for x in ("sieu thi", "bach hoa", "vinmart", "mart")):
        return "Essentials"

    return model_cat or NLU_DEFAULT_CATEGORY


def merge_horizontal_fragmented_boxes(df_boxes: pd.DataFrame, font_height_scale: float = 1.8) -> pd.DataFrame:
    """
    Hợp nhất các bounding box bị phân mảnh theo chiều ngang trên cùng một dòng Y.
    """
    if df_boxes is None or df_boxes.empty:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2", "text", "conf_paddle"])

    required_cols = ["x1", "y1", "x2", "y2", "text"]
    for col in required_cols:
        if col not in df_boxes.columns:
            return df_boxes

    boxes = df_boxes.to_dict(orient="records")
    boxes.sort(key=lambda b: b["y1"])
    
    grouped_rows: list[list[dict]] = []
    
    for box in boxes:
        y1, y2 = box["y1"], box["y2"]
        h = y2 - y1
        if h <= 0:
            continue
        y_center = (y1 + y2) / 2.0
        
        placed = False
        for row in grouped_rows:
            ref_box = row[0]
            ref_y1, ref_y2 = ref_box["y1"], ref_box["y2"]
            ref_h = ref_y2 - ref_y1
            ref_y_center = (ref_y1 + ref_y2) / 2.0
            
            overlap = max(0, min(y2, ref_y2) - max(y1, ref_y1))
            min_h = min(h, ref_h)
            
            if min_h > 0 and (overlap / min_h >= 0.5 or abs(y_center - ref_y_center) < (min_h * 0.4)):
                row.append(box)
                placed = True
                break
        if not placed:
            grouped_rows.append([box])
            
    merged_boxes = []
    
    for row in grouped_rows:
        row.sort(key=lambda b: b["x1"])
        
        current_merged = row[0]
        for next_box in row[1:]:
            curr_x1, curr_y1, curr_x2, curr_y2 = current_merged["x1"], current_merged["y1"], current_merged["x2"], current_merged["y2"]
            next_x1, next_y1, next_x2, next_y2 = next_box["x1"], next_box["y1"], next_box["x2"], next_box["y2"]
            
            curr_h = curr_y2 - curr_y1
            x_distance = next_x1 - curr_x2
            
            if x_distance <= (curr_h * font_height_scale):
                current_merged["text"] = str(current_merged["text"]) + " " + str(next_box["text"])
                current_merged["x1"] = min(curr_x1, next_x1)
                current_merged["y1"] = min(curr_y1, next_y1)
                current_merged["x2"] = max(curr_x2, next_x2)
                current_merged["y2"] = max(curr_y2, next_y2)
                if "conf_paddle" in current_merged and "conf_paddle" in next_box:
                    current_merged["conf_paddle"] = (current_merged["conf_paddle"] + next_box["conf_paddle"]) / 2.0
            else:
                merged_boxes.append(current_merged)
                current_merged = next_box
        merged_boxes.append(current_merged)
        
    return pd.DataFrame(merged_boxes)


def align_skewed_items_and_prices(
    df_boxes: pd.DataFrame,
    img_width: int | None = None,
    skew_angle_rad: float = 0.0
) -> list[tuple[str, int]]:
    """
    Ghép cặp tên mặt hàng và giá tiền tương ứng dựa trên Baseline hình học, có hỗ trợ góc nghiêng.
    """
    if df_boxes is None or df_boxes.empty:
        return []
        
    required_cols = ["x1", "y1", "x2", "y2", "text"]
    for col in required_cols:
        if col not in df_boxes.columns:
            return []
            
    if img_width is None:
        img_width = int(df_boxes["x2"].max())
        
    boundary_x = img_width * 0.55
    names = []
    prices = []
    
    amount_pattern = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+\b|\b\d{4,9}\b")
    
    for _, row in df_boxes.iterrows():
        text = str(row["text"]).strip()
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        
        amounts = amount_pattern.findall(text)
        if amounts and x1 >= boundary_x:
            val = _parse_vn_amount(amounts[-1])
            if val is not None and val >= 1000:
                prices.append({"amount": val, "bbox": [x1, y1, x2, y2]})
        else:
            cleaned_text = re.sub(r"^\s*\d+[\s.-]+", "", text)
            if len(cleaned_text) > 1:
                names.append({"text": cleaned_text, "bbox": [x1, y1, x2, y2]})
                
    names.sort(key=lambda x: x["bbox"][1])
    
    matched_results = []
    used_price_indices = set()
    
    for name_box in names:
        n_bbox = name_box["bbox"]
        name_x = (n_bbox[0] + n_bbox[2]) / 2.0
        name_y = (n_bbox[1] + n_bbox[3]) / 2.0
        font_h = n_bbox[3] - n_bbox[1]
        
        best_price_idx = -1
        min_distance = float("inf")
        
        for idx, price_box in enumerate(prices):
            if idx in used_price_indices:
                continue
                
            p_bbox = price_box["bbox"]
            price_x = (p_bbox[0] + p_bbox[2]) / 2.0
            price_y = (p_bbox[1] + p_bbox[3]) / 2.0
            
            dx = price_x - name_x
            if dx <= 0:
                continue
                
            expected_price_y = name_y + dx * math.tan(skew_angle_rad)
            vertical_deviation = abs(price_y - expected_price_y)
            
            if vertical_deviation < (font_h * 1.5):
                dist = math.sqrt(dx**2 + (price_y - name_y)**2)
                if dist < min_distance:
                    min_distance = dist
                    best_price_idx = idx
                    
        if best_price_idx != -1:
            matched_results.append((name_box["text"], prices[best_price_idx]["amount"]))
            used_price_indices.add(best_price_idx)
        else:
            matched_results.append((name_box["text"], 0))
            
    return matched_results


def validate_and_correct_total_amount(
    ocr_total: int | None,
    items: list[tuple[str, int]],
    tolerance_ratio: float = 0.05
) -> tuple[int | None, list[str]]:
    """
    Kiểm chéo số tiền tổng cộng với tổng các mặt hàng để phát hiện lỗi mất nét số 0 (Digit Drop).
    """
    warnings = []
    if not items:
        return ocr_total, warnings
        
    sum_items = sum(price for _, price in items if price)
    if sum_items == 0:
        return ocr_total, warnings
        
    if ocr_total is None:
        return sum_items, ["AMOUNT_SUMMED_FROM_ITEMS"]
        
    if abs(sum_items - ocr_total * 10) / sum_items <= tolerance_ratio:
        warnings.append("AMOUNT_CORRECTED_SCALE_10X")
        return sum_items, warnings
        
    if abs(sum_items - ocr_total * 100) / sum_items <= tolerance_ratio:
        warnings.append("AMOUNT_CORRECTED_SCALE_100X")
        return sum_items, warnings
        
    if abs(sum_items - ocr_total) / sum_items <= tolerance_ratio:
        return sum_items, warnings
        
    if abs(sum_items - ocr_total * 1000) / sum_items <= tolerance_ratio:
        warnings.append("AMOUNT_CORRECTED_SCALE_1000X")
        return sum_items, warnings
        
    return ocr_total, warnings


def predict_item_category(item_name: str) -> tuple[str, float]:
    """Dự đoán danh mục cho một tên sản phẩm riêng lẻ."""
    model_cat = _predict_category_model([item_name])
    if model_cat:
        return model_cat, 0.9
    scores = _score_categories_from_lines([item_name])
    best = max(scores, key=lambda k: scores[k])
    if scores[best] > 0:
        return best, 0.8
    return NLU_DEFAULT_CATEGORY, 0.5


def resolve_mixed_receipt_categories(
    items: list[tuple[str, int]],
    split_mode: bool = False,
    entropy_threshold: float = 0.65,
    confidence_threshold: float = 0.60,
) -> list[dict[str, Any]]:
    """
    Phân loại hóa đơn hỗn hợp.

    - ``split_mode=False`` (mặc định, bill-only app): Weighted Voting by Value
      Score(C) = sum(Price_i * Confidence_i) → một danh mục đại diện.
    - ``split_mode=True``: gom tiền thực theo danh mục → nhiều giao dịch con.
    """
    if not items:
        return [{"category": NLU_DEFAULT_CATEGORY, "amount": 0}]

    category_scores: dict[str, float] = {}
    category_amounts: dict[str, int] = {}

    for name, price in items:
        price = price or 0
        category, confidence = predict_item_category(name)
        cat_label = category if confidence >= confidence_threshold else NLU_DEFAULT_CATEGORY
        vote_weight = price * confidence

        category_scores[cat_label] = category_scores.get(cat_label, 0.0) + vote_weight
        category_amounts[cat_label] = category_amounts.get(cat_label, 0) + price

    total_bill_amount = sum(price for _, price in items if price)
    if total_bill_amount == 0:
        first_cat = predict_item_category(items[0][0])[0] if items else NLU_DEFAULT_CATEGORY
        return [{"category": first_cat, "amount": 0}]

    total_weighted = sum(category_scores.values()) or 1.0

    if split_mode:
        transactions = [
            {
                "category": cat,
                "amount": amt,
                "is_split": True,
                "vote_score": category_scores.get(cat, 0.0),
            }
            for cat, amt in category_amounts.items()
        ]
        transactions.sort(key=lambda x: (-x.get("vote_score", 0.0), -x["amount"]))
        return transactions

    primary_category = max(category_scores, key=category_scores.get)
    primary_score = category_scores[primary_category]
    dominance_ratio = primary_score / total_weighted

    if dominance_ratio >= entropy_threshold:
        return [{
            "category": primary_category,
            "amount": total_bill_amount,
            "vote_score": primary_score,
        }]

    return [{
        "category": "Essentials",
        "amount": total_bill_amount,
        "note": "Hóa đơn siêu thị hỗn hợp nhiều mặt hàng",
    }]


def extract_receipt_summary(
    df_lines: pd.DataFrame,
    df_boxes: pd.DataFrame | None = None,
    split_mode: bool = False,
    kie_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse OCR lines/boxes with KIE fields → category, amount, transactions.

    Responsibility split (logical, not parallel):
    - **Category**: item alignment + NLU category model + brand routing (SELLER from KIE when set)
    - **Amount**: PICK ``TOTAL_COST_VALUE`` when set, else heuristic OCR + item-sum validation
    """
    if df_lines is None or df_lines.empty:
        return {
            "category": NLU_DEFAULT_CATEGORY,
            "amount": None,
            "currency": "VND",
        }

    lines = [str(x).strip() for x in df_lines["line_text"].tolist() if str(x).strip()]
    seller_text = (kie_fields or {}).get("SELLER") or " ".join(lines[:2])
    brand_category = route_brand_category(seller_text)

    if df_boxes is not None and not df_boxes.empty:
        df_boxes_merged = merge_horizontal_fragmented_boxes(df_boxes)
        matched_items = align_skewed_items_and_prices(df_boxes_merged)
        ocr_total = extract_total_amount(lines)
        kie_total = (kie_fields or {}).get("TOTAL_COST_VALUE")
        if kie_total:
            raw_amount = int(kie_total)
            if ocr_total is not None and ocr_total != raw_amount:
                math_preface = [f"AMOUNT_FROM_KIE ({raw_amount}); OCR heuristic had {ocr_total}"]
            else:
                math_preface = []
        else:
            raw_amount = ocr_total
            math_preface = []
        final_amount, math_warnings = validate_and_correct_total_amount(raw_amount, matched_items)
        math_warnings = math_preface + math_warnings

        if brand_category:
            transactions = [{"category": brand_category, "amount": final_amount, "brand_routed": True}]
            category = brand_category
        else:
            transactions = resolve_mixed_receipt_categories(matched_items, split_mode=split_mode)
            primary_tx = transactions[0] if transactions else {"category": NLU_DEFAULT_CATEGORY}
            category = primary_tx.get("category", NLU_DEFAULT_CATEGORY)

        return {
            "category": category,
            "amount": final_amount,
            "currency": "VND",
            "transactions": transactions,
            "warnings": math_warnings,
            "items_count": len(matched_items),
            "seller": seller_text,
            "kie_fields": kie_fields or {},
            "brand_routed": bool(brand_category),
        }

    category = brand_category or classify_category(lines)
    amount = extract_total_amount(lines)
    if kie_fields and kie_fields.get("TOTAL_COST_VALUE"):
        amount = kie_fields["TOTAL_COST_VALUE"] or amount
    return {
        "category": category,
        "amount": amount,
        "currency": "VND",
        "seller": seller_text,
        "kie_fields": kie_fields or {},
        "brand_routed": bool(brand_category),
    }
