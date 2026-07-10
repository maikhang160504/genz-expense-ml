"""Hybrid receipt pipeline: OCR → parallel(KIE, CPU prep) → fusion.

Architecture
------------
1. **OCR stage** (sequential): Paddle detect → rotation (MobileNetV3) → VietOCR read → df_boxes + df_lines
2. **Parallel branches** (same OCR output):
   - **KIE branch**: PICK (MC_OCR) or heuristic → entity tags → kie_fields (SELLER, TOTAL_COST, …)
   - **Prep branch** (CPU): merge/align boxes, line text — while KIE runs on GPU
3. **Fusion**: ``extract_receipt_summary`` merges OCR + kie_fields → category + amount

Category and amount are NOT independent outputs — fusion cross-checks KIE total vs
heuristic OCR lines and uses SELLER from KIE for brand routing when available.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .pick_kie import PickKIEEngine, extract_kie_fields
from .receipt_nlu import (
    align_skewed_items_and_prices,
    extract_receipt_summary,
    merge_horizontal_fragmented_boxes,
)


@dataclass
class OcrStageResult:
    df_boxes: pd.DataFrame
    df_lines: pd.DataFrame


@dataclass
class KieBranchResult:
    labeled_boxes: list[dict[str, Any]]
    kie_fields: dict[str, Any]
    kie_backend: str


@dataclass
class PrepBranchResult:
    lines: list[str]
    matched_items: list[tuple[str, int]]


@dataclass
class FusedReceiptResult:
    summary: dict[str, Any]
    labeled_boxes: list[dict[str, Any]] = field(default_factory=list)
    kie_fields: dict[str, Any] = field(default_factory=dict)
    kie_backend: str = "heuristic"
    prep: PrepBranchResult | None = None


def run_ocr_stage(pipeline: Any, image_rgb: np.ndarray) -> OcrStageResult:
    """Paddle detect + VietOCR read + line grouping."""
    df_boxes = pipeline.ocr_boxes(image_rgb)
    df_lines = pipeline.group_lines(df_boxes)
    return OcrStageResult(df_boxes=df_boxes, df_lines=df_lines)


def run_kie_branch(
    kie_engine: PickKIEEngine,
    df_boxes: pd.DataFrame,
    image_rgb: np.ndarray,
) -> KieBranchResult:
    """PICK KIE (or heuristic) entity labeling."""
    labeled_boxes = kie_engine.label_boxes(df_boxes, image_rgb)
    kie_fields = extract_kie_fields(labeled_boxes)
    return KieBranchResult(
        labeled_boxes=labeled_boxes,
        kie_fields=kie_fields,
        kie_backend=kie_engine.backend,
    )


def run_prep_branch(df_boxes: pd.DataFrame, df_lines: pd.DataFrame) -> PrepBranchResult:
    """CPU-side OCR structure prep (can overlap with KIE inference)."""
    lines: list[str] = []
    if df_lines is not None and not df_lines.empty:
        lines = [str(x).strip() for x in df_lines["line_text"].tolist() if str(x).strip()]

    matched_items: list[tuple[str, int]] = []
    if df_boxes is not None and not df_boxes.empty:
        merged = merge_horizontal_fragmented_boxes(df_boxes)
        matched_items = align_skewed_items_and_prices(merged)

    return PrepBranchResult(lines=lines, matched_items=matched_items)


def fuse_receipt_summary(
    df_lines: pd.DataFrame,
    df_boxes: pd.DataFrame,
    kie_fields: dict[str, Any],
    *,
    split_mode: bool = False,
    prep: PrepBranchResult | None = None,
) -> dict[str, Any]:
    """Merge KIE + OCR into final category, amount, transactions."""
    summary = extract_receipt_summary(
        df_lines,
        df_boxes=df_boxes,
        split_mode=split_mode,
        kie_fields=kie_fields,
    )
    if prep is not None:
        summary["_prep_items_count"] = len(prep.matched_items)
    return summary


def _lines_for_output(df_lines: pd.DataFrame) -> list[dict[str, Any]]:
    lines_out: list[dict[str, Any]] = []
    if df_lines is None or df_lines.empty:
        return lines_out
    for _, row in df_lines.iterrows():
        text = str(row["line_text"]).strip()
        if not text:
            continue
        bbox = row.get("bbox")
        lines_out.append({
            "text": text,
            "bbox": [float(x) for x in bbox] if bbox is not None else None,
        })
    return lines_out


def run_mcocr_pipeline(
    pipeline: Any,
    kie_engine: PickKIEEngine,
    image_rgb: np.ndarray,
    *,
    split_mode: bool = False,
    parallel: bool = True,
) -> FusedReceiptResult:
    """
    Full MC_OCR-style flow: OCR → parallel(PICK KIE, prep) → fusion.

    Set ``parallel=False`` for deterministic single-thread debugging.
    """
    ocr = run_ocr_stage(pipeline, image_rgb)

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            kie_future = pool.submit(run_kie_branch, kie_engine, ocr.df_boxes, image_rgb)
            prep_future = pool.submit(run_prep_branch, ocr.df_boxes, ocr.df_lines)
            kie = kie_future.result()
            prep = prep_future.result()
    else:
        kie = run_kie_branch(kie_engine, ocr.df_boxes, image_rgb)
        prep = run_prep_branch(ocr.df_boxes, ocr.df_lines)

    summary = fuse_receipt_summary(
        ocr.df_lines,
        ocr.df_boxes,
        kie.kie_fields,
        split_mode=split_mode,
        prep=prep,
    )

    # Auto-correct TOTAL_COST: value box must match final_amount and align with total label row
    final_amount = summary.get("amount")
    if not final_amount:
        from .receipt_nlu import _amounts_in_line

        candidates: list[int] = []
        for box in kie.labeled_boxes:
            candidates.extend(_amounts_in_line(str(box.get("text", ""))))
        if candidates:
            final_amount = max(candidates)
    if final_amount:
        from .pick_kie import extract_kie_fields, reconcile_total_cost_boxes

        kie.labeled_boxes = reconcile_total_cost_boxes(kie.labeled_boxes, final_amount)
        kie.kie_fields = extract_kie_fields(kie.labeled_boxes)
        summary["kie_fields"] = kie.kie_fields

    summary["lines"] = _lines_for_output(ocr.df_lines)
    summary["boxes"] = kie.labeled_boxes
    summary["kie_fields"] = kie.kie_fields
    summary["kie_backend"] = kie.kie_backend

    return FusedReceiptResult(
        summary=summary,
        labeled_boxes=kie.labeled_boxes,
        kie_fields=kie.kie_fields,
        kie_backend=kie.kie_backend,
        prep=prep,
    )


run_hybrid_pipeline = run_mcocr_pipeline
