"""PICK KIE + heuristic fallback for receipt field labeling (MC_OCR style)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_paths import PICK_KIE_MODEL_PATH
from .receipt_nlu import _amounts_in_line, _normalize, _parse_vn_amount

ENTITY_OTHER = "OTHER"
ENTITY_SELLER = "SELLER"
ENTITY_ADDRESS = "ADDRESS"
ENTITY_TIMESTAMP = "TIMESTAMP"
ENTITY_TOTAL = "TOTAL_COST"
KIE_ENTITIES = {ENTITY_SELLER, ENTITY_ADDRESS, ENTITY_TIMESTAMP, ENTITY_TOTAL, ENTITY_OTHER}

_RE_TOTAL = re.compile(
    r"tong\s*thanh\s*toan|thuc\s*thu|thuc\s*tra|phai\s*thanh\s*toan|"
    r"tong\s*cong|thanh\s*tien|tong\s*tien|total",
    re.I,
)
_RE_ADDRESS = re.compile(r"duong|phuong|quan|tp\.|thanh\s*pho|so\s*\d", re.I)
_RE_TIMESTAMP = re.compile(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}:\d{2}")
_RE_EXCLUDE = re.compile(
    r"tien\s*mat|khach\s*dua|tien\s*khach|cash|received|tra\s*lai|tien\s*thua|thoi\s*lai|change",
    re.I,
)


def default_kie_model_path() -> Path:
    return PICK_KIE_MODEL_PATH


def pick_kie_weights_status(model_path: str | Path | None = None) -> dict[str, Any]:
    """Check whether PICK model_best.pth is present on disk."""
    path = Path(model_path) if model_path else default_kie_model_path()
    ready = path.is_file()
    return {
        "model_path": str(path),
        "weights_found": ready,
        "weight_files": [path.name] if ready else [],
        "ready": ready,
    }


def _box_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "x1": int(row["x1"]),
        "y1": int(row["y1"]),
        "x2": int(row["x2"]),
        "y2": int(row["y2"]),
        "text": str(row.get("text", "")).strip(),
        "entity": str(row.get("entity", ENTITY_OTHER)),
        "confidence": float(row.get("confidence", 0.75)),
    }


def heuristic_label_boxes(df_boxes: pd.DataFrame) -> list[dict[str, Any]]:
    """Rule-based KIE when PICK weights are unavailable."""
    if df_boxes is None or df_boxes.empty:
        return []

    rows = df_boxes.sort_values(by=["y1", "x1"]).reset_index(drop=True)
    out: list[dict[str, Any]] = []
    n = len(rows)
    seller_assigned = False

    for idx, row in rows.iterrows():
        text = str(row["text"]).strip()
        low = _normalize(text)
        entity = ENTITY_OTHER
        conf = 0.72

        if not seller_assigned and idx < max(3, n // 8) and len(text) >= 3 and not _RE_TOTAL.search(low):
            if not _RE_TIMESTAMP.search(text) and not _amounts_in_line(text):
                entity = ENTITY_SELLER
                seller_assigned = True
                conf = 0.80
        elif _RE_TIMESTAMP.search(text):
            entity = ENTITY_TIMESTAMP
            conf = 0.78
        elif _RE_ADDRESS.search(low) and len(text) > 8:
            entity = ENTITY_ADDRESS
            conf = 0.70
        elif _RE_TOTAL.search(low) and not _RE_EXCLUDE.search(low):
            entity = ENTITY_TOTAL
            conf = 0.85
        elif idx >= n - 5 and _amounts_in_line(text) and not _RE_EXCLUDE.search(low):
            entity = ENTITY_TOTAL
            conf = 0.75

        out.append(_box_row_to_dict({**row.to_dict(), "entity": entity, "confidence": conf}))
    return out


def extract_kie_fields(labeled_boxes: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge labeled boxes into header/footer field values."""
    fields: dict[str, Any] = {
        "SELLER": None,
        "ADDRESS": None,
        "TIMESTAMP": None,
        "TOTAL_COST": None,
    }
    chunks: dict[str, list[str]] = {k: [] for k in fields}

    for box in labeled_boxes:
        ent = box.get("entity", ENTITY_OTHER)
        text = str(box.get("text", "")).strip()
        if not text or ent not in chunks:
            continue
        chunks[ent].append(text)

    for key in fields:
        if chunks[key]:
            fields[key] = " ".join(chunks[key]).strip()

    if fields["TOTAL_COST"]:
        amounts = _amounts_in_line(fields["TOTAL_COST"])
        if amounts:
            fields["TOTAL_COST_VALUE"] = max(amounts)
        else:
            fields["TOTAL_COST_VALUE"] = _parse_vn_amount(fields["TOTAL_COST"])
    else:
        fields["TOTAL_COST_VALUE"] = None

    return fields


class PickKIEEngine:
    """PICK graph KIE model (MC_OCR); falls back to heuristics until model is deployed."""

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        self.model_path = Path(model_path) if model_path else default_kie_model_path()
        self.device = device or "cpu"
        self._model = None
        self._backend = "heuristic"
        self._load_error: str | None = None

    @property
    def backend(self) -> str:
        return self._backend

    def load(self) -> PickKIEEngine:
        status = pick_kie_weights_status(self.model_path)
        if not status["weights_found"]:
            self._backend = "heuristic"
            self._load_error = (
                f"No PICK weights at {self.model_path}. "
                "Run Kaggle train-pick-kie and copy model_best.pth into models/pick_kie/."
            )
            return self

        try:
            from .pick_kie_inference import PickRuntime

            runtime = PickRuntime(self.model_path, device=self.device).load()
            self._model = runtime
            self._backend = "pick"
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._backend = "heuristic"
            self._load_error = (
                f"PICK weights found at {self.model_path} but inference failed: {exc}. "
                "Install torchtext==0.6.0 overrides; MC_OCR PICK sources must be available."
            )
        return self

    def label_boxes(self, df_boxes: pd.DataFrame, image_rgb: np.ndarray | None = None) -> list[dict[str, Any]]:
        if df_boxes is None or df_boxes.empty:
            return []
        if self._backend == "pick" and self._model is not None and image_rgb is not None:
            try:
                return self._predict_pick(df_boxes, image_rgb)
            except Exception:  # noqa: BLE001
                return heuristic_label_boxes(df_boxes)
        return heuristic_label_boxes(df_boxes)

    def _predict_pick(self, df_boxes: pd.DataFrame, image_rgb: np.ndarray | None) -> list[dict[str, Any]]:
        if image_rgb is None:
            return heuristic_label_boxes(df_boxes)
        runtime = self._model
        return runtime.predict(df_boxes, image_rgb)


_engine: PickKIEEngine | None = None


def reset_kie_engine() -> None:
    """Drop cached KIE engine so reload picks up new PICK weights."""
    global _engine
    _engine = None


def get_kie_engine(model_path: str | Path | None = None, device: str | None = None) -> PickKIEEngine:
    global _engine
    if _engine is None:
        _engine = PickKIEEngine(model_path=model_path, device=device).load()
    return _engine
