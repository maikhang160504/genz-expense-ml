"""MC_OCR-style receipt pipeline: PaddleOCR + VietOCR + PICK KIE + NLU fusion."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .pick_kie import get_kie_engine
from .pipeline import ReceiptOCRPipeline
from .receipt_fusion import run_mcocr_pipeline
from .rotation_corrector import RotationCorrector, get_rotation_corrector


class McOcrReceiptPipeline(ReceiptOCRPipeline):
    """Paddle → rotation → VietOCR → parallel(PICK KIE, CPU prep) → fusion."""

    def __init__(
        self,
        vietocr_weights: str | Path,
        device: str | None = None,
        pick_kie_model: str | Path | None = None,
        rotation_weights: str | Path | None = None,
        use_rotation: bool = True,
        parallel_branches: bool = True,
        **kwargs: Any,
    ):
        super().__init__(vietocr_weights=vietocr_weights, device=device, **kwargs)
        self.pick_kie_model = pick_kie_model
        self.use_rotation = use_rotation
        self.rotation_weights = rotation_weights
        self.parallel_branches = parallel_branches
        self._kie = None
        self._rotator: RotationCorrector | None = None

    def _get_kie(self):
        if self._kie is None:
            self._kie = get_kie_engine(model_path=self.pick_kie_model, device=self.device)
        return self._kie

    def _get_rotator(self) -> RotationCorrector:
        if self._rotator is None:
            self._rotator = get_rotation_corrector(
                self.rotation_weights,
                enabled=self.use_rotation,
            )
        return self._rotator

    def load(self):
        super().load()
        if self.use_rotation:
            self._get_rotator()
        return self

    def ocr_boxes(self, image_rgb: np.ndarray) -> pd.DataFrame:
        rotator = self._get_rotator()
        if self.use_rotation and rotator.available:
            polys = self.detect_polys(image_rgb)
            if polys:
                image_rgb, _meta = rotator.correct(image_rgb, polys)
        return super().ocr_boxes(image_rgb)

    def process_image_rgb(
        self,
        image_rgb: np.ndarray,
        split_mode: bool = False,
    ) -> dict[str, Any]:
        fused = run_mcocr_pipeline(
            self,
            self._get_kie(),
            image_rgb,
            split_mode=split_mode,
            parallel=self.parallel_branches,
        )
        return fused.summary

    def process_image(self, image_path: str | Path, split_mode: bool = False) -> dict[str, Any]:
        image_rgb = self._read_rgb(image_path)
        return self.process_image_rgb(image_rgb, split_mode=split_mode)

    def prelabel_for_admin(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """Auto-label for WebAdmin: OCR → PICK entities → fusion summary."""
        result = self.process_image_rgb(image_rgb, split_mode=False)
        kie = self._get_kie()
        warnings = list(result.get("warnings") or [])
        if kie.backend != "pick":
            msg = getattr(kie, "_load_error", None) or "PICK weights not loaded — using heuristic KIE."
            warnings.append(msg)
        return {
            "boxes": result.get("boxes", []),
            "kie_fields": result.get("kie_fields", {}),
            "amount": result.get("amount"),
            "category": result.get("category"),
            "lines": result.get("lines", []),
            "items_count": result.get("items_count", 0),
            "warnings": warnings,
            "kie_backend": result.get("kie_backend", "heuristic"),
            "auto_label_engine": "paddle+rotation+vietocr+pick"
            if result.get("kie_backend") == "pick"
            else "paddle+rotation+vietocr+heuristic",
            "rotation_enabled": self.use_rotation and self._get_rotator().available,
        }
