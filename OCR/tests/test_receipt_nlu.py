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


if __name__ == "__main__":
    test_receipt_nlu_categories_and_amounts()
    print("All NLU receipt tests passed.")
