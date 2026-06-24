"""MC_OCR rotation corrector (MobileNetV3, step 2) — bundled weights + vendored code."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .model_paths import ROTATION_WEIGHTS, resolve_rotation_weights_path
from .mcocr_rotation.geometry import (
    drop_box,
    filter_90_box,
    get_mean_horizontal_angle,
    rotate_image_bbox_angle,
    vote_page_flip,
)
from .mcocr_rotation.predictor import PageRotationModel, load_rotation_model

DEFAULT_WEIGHT_NAME = ROTATION_WEIGHTS.name
ROT_DROP_THRESH = (0.5, 2.0)


def rotation_weights_status(explicit: str | Path | None = None) -> dict[str, Any]:
    path = resolve_rotation_weights_path(explicit)
    return {
        "ready": path.is_file(),
        "weights_found": path.is_file(),
        "model_path": str(path),
    }


def paddle_poly_to_coors(poly: list[Any]) -> list[int]:
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    flat: list[int] = []
    for x, y in pts[:4]:
        flat.extend([int(round(float(x))), int(round(float(y)))])
    while len(flat) < 8:
        flat.extend(flat[-2:])
    return flat[:8]


def polys_to_box_dicts(polys: list[Any]) -> list[dict[str, Any]]:
    return [{"coors": paddle_poly_to_coors(poly), "data": ""} for poly in polys]


class RotationCorrector:
    def __init__(self, weight_path: str | Path | None = None, *, enabled: bool = True):
        self.weight_path = resolve_rotation_weights_path(weight_path)
        self.enabled = enabled
        self._model: PageRotationModel | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def load(self) -> RotationCorrector:
        if not self.enabled:
            return self
        if not self.weight_path.is_file():
            self._load_error = f"Rotation weights missing: {self.weight_path}"
            return self
        try:
            self._model = load_rotation_model(self.weight_path)
            self._load_error = None
        except Exception as exc:
            self._model = None
            self._load_error = str(exc)
        return self

    def correct(self, image_rgb: np.ndarray, polys: list[Any]) -> tuple[np.ndarray, dict[str, Any]]:
        meta: dict[str, Any] = {
            "applied": False,
            "skew_deg": 0.0,
            "flip_deg": 0,
            "backend": "none",
        }
        if not self.enabled or not polys or self._model is None:
            if self._load_error:
                meta["error"] = self._load_error
            return image_rgb, meta

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        boxes_list = drop_box(polys_to_box_dicts(polys), drop_gap=ROT_DROP_THRESH)
        if not boxes_list:
            meta["error"] = "no boxes after drop_box filter"
            return image_rgb, meta

        skew = float(get_mean_horizontal_angle(boxes_list, True))
        img_rotated, boxes_list = rotate_image_bbox_angle(image_bgr, boxes_list, skew)
        flip = int(vote_page_flip(self._model, img_rotated, boxes_list))
        if flip:
            img_rotated, boxes_list = rotate_image_bbox_angle(img_rotated, boxes_list, flip)
        filter_90_box(boxes_list)

        meta.update(
            {
                "applied": abs(skew) > 0.5 or flip != 0,
                "skew_deg": skew,
                "flip_deg": flip,
                "backend": "mobilenetv3",
            }
        )
        return cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB), meta


def get_rotation_corrector(weight_path: str | Path | None = None, *, enabled: bool = True) -> RotationCorrector:
    return RotationCorrector(weight_path=weight_path, enabled=enabled).load()
