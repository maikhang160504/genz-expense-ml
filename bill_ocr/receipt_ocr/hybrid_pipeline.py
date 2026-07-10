"""Backward-compatible alias — use mcocr_pipeline.McOcrReceiptPipeline."""
from __future__ import annotations

from .mcocr_pipeline import McOcrReceiptPipeline

HybridReceiptOCRPipeline = McOcrReceiptPipeline

__all__ = ["HybridReceiptOCRPipeline", "McOcrReceiptPipeline"]
