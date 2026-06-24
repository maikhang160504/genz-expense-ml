"""Vendored MC_OCR rotation corrector (MobileNetV3) — no external MC_OCR repo at runtime."""
from .predictor import PageRotationModel, load_rotation_model

__all__ = ["PageRotationModel", "load_rotation_model"]
