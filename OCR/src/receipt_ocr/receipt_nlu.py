"""Trích xuất danh mục (NLU Record) + số tiền từ dòng OCR hóa đơn."""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

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
    valid = set(cmap.values())
    if label in valid:
        return label
    _ensure_train_on_path()
    try:
        from src.nlu.ner import map_category_to_label, normalize_text

        key = normalize_text(str(label))
        mapped = map_category_to_label(key) or cmap.get(key)
        if mapped:
            return mapped
    except ImportError:
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


def extract_receipt_summary(df_lines: pd.DataFrame) -> dict[str, Any]:
    """Kết quả demo: category (NLU Record) + amount."""
    if df_lines is None or df_lines.empty:
        return {
            "category": NLU_DEFAULT_CATEGORY,
            "amount": None,
            "currency": "VND",
        }

    lines = [str(x).strip() for x in df_lines["line_text"].tolist() if str(x).strip()]
    return {
        "category": classify_category(lines),
        "amount": extract_total_amount(lines),
        "currency": "VND",
    }
