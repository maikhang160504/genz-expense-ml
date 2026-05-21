from __future__ import annotations

from typing import Any

import pandas as pd

from .receipt_nlu import extract_receipt_summary

MCOCR_LABELS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "TAX_ID", "PRODUCT")


def extract_receipt_fields(df_lines: pd.DataFrame) -> dict[str, Any]:
    """Giữ tương thích nội bộ; demo dùng extract_receipt_summary."""
    return extract_receipt_summary(df_lines)
