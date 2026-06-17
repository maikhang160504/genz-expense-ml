"""Kiểm tra NLU hóa đơn: nhãn Record + số tiền."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR_ROOT / "src"))

from receipt_ocr.receipt_nlu import extract_receipt_summary  # noqa: E402

# raw_lines từ OCR/demo/output (phiên bản đầy đủ) hoặc lines trong JSON cũ
FIXTURE_LINES: dict[str, list[str]] = {
    "mcocr_public_145013aukcu": [
        "CỬA HÀNG NĂM OÁNH",
        "Cộng tiền hàng 63,000",
        "1 14,000 14,000",
        "1 49,000 49,000",
        "Đường tinh luyện xuất khầu lkg",
        "Dầu Cái Lân 2L",
    ],
    "mcocr_public_145013cnknh": [
        "SIEU THI BACH HOA TONG HOP",
        "HOP 34,000 34,000",
        "Kem que thập cẩm Tràng Tiền",
        "Tổng số Tổng cộng (đã gồm VAT) 1.00 34,(00",
    ],
    "mcocr_public_145013tvyce": [
        "THE COFFEE HQUSE",
        "Cà Phề Đen Đá",
        "Thành tiền: 32.000",
        "T.Tiên 32.000",
    ],
    "mcocr_public_145014irkqn": [
        "MINIMART ANAN",
        "knor nấm",
        "bobby s46",
        "Tổng cộng Tiền khách trả: 2 VND 198,000 198,000",
    ],
    "mcocr_public_145014klyxi": [
        "MINIMART ANAN",
        "Keo Dynamite",
        "keo alpenliebe",
        "Tổng cộng: 50,000 50,000",
    ],
    "mcocr_public_145014nasdz": [
        "MINIMART ANAN",
        "Đường trắng 1kg",
        "Tương",
        "Măm nam ngư",
        "Tổng cộng Tiền khách trả: 110,000 110,000",
    ],
    "mcocr_public_145014omzxy": [
        "MINIMART ANAN",
        "Sữa bột",
        "Tổng cộng 195,000",
    ],
    "mcocr_public_145014oxghl": [
        "VinCommerce",
        "DUREx Bao cao su",
        "TỔNG TIỀN PHẢI T.TOÁN 100.000 58.500 58.500",
    ],
}

EXPECTED = {
    "mcocr_public_145013aukcu": (63000, "Food"),
    "mcocr_public_145013cnknh": (34000, "Food"),
    "mcocr_public_145013tvyce": (32000, "Food"),
    "mcocr_public_145014irkqn": (198000, "Essentials"),
    "mcocr_public_145014klyxi": (50000, "Food"),
    "mcocr_public_145014nasdz": (110000, "Food"),
    "mcocr_public_145014omzxy": (195000, "Food"),
    "mcocr_public_145014oxghl": (58500, "Shopping"),
}


def _lines_for(stem: str) -> list[str]:
    path = OCR_ROOT / "demo" / "output" / f"{stem}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        old = data.get("fields", {}).get("raw_lines")
        if old:
            return old
        if data.get("lines"):
            return [ln["line_text"] for ln in data["lines"]]
    return FIXTURE_LINES[stem]


def test_receipt_nlu_categories_and_amounts():
    for stem, (exp_amount, exp_cat) in EXPECTED.items():
        lines = _lines_for(stem)
        got = extract_receipt_summary(pd.DataFrame({"line_text": lines}))
        assert got["category"] == exp_cat, f"{stem}: {got['category']} != {exp_cat}"
        assert got["amount"] == exp_amount, f"{stem}: {got['amount']} != {exp_amount}"


def test_horizontal_merging():
    from receipt_ocr.receipt_nlu import merge_horizontal_fragmented_boxes
    df = pd.DataFrame([
        {"x1": 10, "y1": 100, "x2": 50, "y2": 120, "text": "Trà"},
        {"x1": 55, "y1": 100, "x2": 100, "y2": 120, "text": "Sữa"},
        {"x1": 110, "y1": 100, "x2": 160, "y2": 120, "text": "Matcha"},
    ])
    merged = merge_horizontal_fragmented_boxes(df)
    assert not merged.empty
    assert merged.iloc[0]["text"] == "Trà Sữa Matcha"
    assert merged.iloc[0]["x1"] == 10
    assert merged.iloc[0]["x2"] == 160


def test_skewed_alignment():
    from receipt_ocr.receipt_nlu import align_skewed_items_and_prices
    df = pd.DataFrame([
        {"x1": 10, "y1": 100, "x2": 150, "y2": 120, "text": "Cà phê sữa"},
        {"x1": 600, "y1": 102, "x2": 680, "y2": 122, "text": "29.000"},
        {"x1": 10, "y1": 130, "x2": 150, "y2": 150, "text": "Bánh ngọt"},
        {"x1": 600, "y1": 132, "x2": 680, "y2": 152, "text": "45.000"},
    ])
    items = align_skewed_items_and_prices(df, img_width=1000)
    assert len(items) == 2
    assert ("Cà phê sữa", 29000) in items
    assert ("Bánh ngọt", 45000) in items


def test_amount_math_validation():
    from receipt_ocr.receipt_nlu import validate_and_correct_total_amount
    items = [("Cà phê sữa", 29000), ("Bánh ngọt", 45000)]
    
    # 1. Khớp hoàn toàn
    val, w = validate_and_correct_total_amount(74000, items)
    assert val == 74000
    assert not w
    
    # 2. Lỗi mất số 0 cuối (đọc là 7400)
    val, w = validate_and_correct_total_amount(7400, items)
    assert val == 74000
    assert "AMOUNT_CORRECTED_SCALE_10X" in w

    # 3. Lỗi mất hai số 0 cuối (đọc là 740)
    val, w = validate_and_correct_total_amount(740, items)
    assert val == 74000
    assert "AMOUNT_CORRECTED_SCALE_100X" in w


def test_resolve_mixed_categories():
    from receipt_ocr.receipt_nlu import resolve_mixed_receipt_categories
    items = [("Trà Sữa Matcha", 35000), ("Cà phê sữa", 29000), ("Khẩu trang y tế", 50000)]
    
    txs_split = resolve_mixed_receipt_categories(items, split_mode=True)
    assert len(txs_split) >= 2
    
    txs_single = resolve_mixed_receipt_categories(items, split_mode=False)
    assert len(txs_single) == 1


if __name__ == "__main__":
    test_receipt_nlu_categories_and_amounts()
    test_horizontal_merging()
    test_skewed_alignment()
    test_amount_math_validation()
    test_resolve_mixed_categories()
    print("All NLU receipt tests passed successfully.")
