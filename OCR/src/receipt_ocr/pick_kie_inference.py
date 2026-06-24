"""Runtime PICK KIE inference using vendored PICK code under OCR/vendor/pick.

Requires:
  - ``model_best.pth`` at ``models/pick_kie/``
  - PICK sources at ``OCR/vendor/pick`` (copied from MC_OCR at build time)
  - Extra packages: ``torchtext==0.6.0``, ``overrides``, ``prefetch-generator``

Set ``MC_OCR_PICK_DIR`` to override PICK source path.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from .pick_kie import ENTITY_OTHER, KIE_ENTITIES, _box_row_to_dict

RESIZED_IMAGE_SIZE = (560, 784)
MAX_BOXES_NUM = 130
MAX_TRANSCRIPT_LEN = 70


def resolve_mc_ocr_pick_dir() -> Path | None:
    env = os.environ.get("MC_OCR_PICK_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "model" / "pick.py").is_file():
            return p
    ocr_root = Path(__file__).resolve().parents[2]
    vendor = ocr_root / "vendor" / "pick"
    if (vendor / "model" / "pick.py").is_file():
        return vendor
    return None


def _ensure_pick_on_path(pick_dir: Path) -> None:
    pick_str = str(pick_dir)
    if pick_str not in sys.path:
        sys.path.insert(0, pick_str)


def _axis_aligned_to_quad(x1: int, y1: int, x2: int, y2: int) -> list[float]:
    return [float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]


def df_boxes_to_pick_tsv(df_boxes: pd.DataFrame) -> str:
    """Convert OCR boxes to MC_OCR PICK inference TSV (8-point + transcript)."""
    rows = df_boxes.sort_values(by=["y1", "x1"]).reset_index(drop=True)
    lines: list[str] = []
    for idx, row in rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        text = str(row.get("text", "")).replace("\n", " ").strip() or " "
        pts = _axis_aligned_to_quad(x1, y1, x2, y2)
        coord = ",".join(str(int(v)) for v in pts)
        lines.append(f"{idx + 1},{coord},{text}")
    return "\n".join(lines) + "\n"


def _entity_from_box_tags(box_tags: list[str]) -> str:
    for tag in box_tags:
        if tag.startswith("B-"):
            name = tag[2:]
            return name if name in KIE_ENTITIES else ENTITY_OTHER
    for tag in box_tags:
        if tag.startswith("I-"):
            name = tag[2:]
            return name if name in KIE_ENTITIES else ENTITY_OTHER
    return ENTITY_OTHER


def _per_box_entities(decoded_tags: list[str], text_lengths: list[int]) -> list[str]:
    entities: list[str] = []
    pos = 0
    for box_len in text_lengths:
        box_len = int(box_len)
        box_tags = decoded_tags[pos : pos + box_len]
        pos += box_len
        entities.append(_entity_from_box_tags(box_tags))
    return entities


class PickRuntime:
    """Loads MC_OCR PICK checkpoint and runs single-image KIE."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self._pick_dir: Path | None = None
        self._model = None

    def load(self) -> PickRuntime:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"PICK weights not found: {self.model_path}")

        pick_dir = resolve_mc_ocr_pick_dir()
        if pick_dir is None:
            raise FileNotFoundError(
                "MC_OCR PICK sources not found. Clone MC_OCR or set MC_OCR_PICK_DIR to "
                "mc_ocr/key_info_extraction/PICK"
            )
        self._pick_dir = pick_dir
        _ensure_pick_on_path(pick_dir)

        from .pick_allennlp_shim import install_allennlp_shim

        install_allennlp_shim()

        import model.pick as pick_arch_module  # noqa: WPS433
        from data_utils.documents import Document  # noqa: WPS433
        from data_utils.pick_dataset import BatchCollateFn  # noqa: WPS433
        from utils.util import iob_index_to_str  # noqa: WPS433

        self._Document = Document
        self._BatchCollateFn = BatchCollateFn
        self._iob_index_to_str = iob_index_to_str

        load_kw: dict[str, Any] = {"map_location": self.device}
        # Checkpoint may contain PosixPath from Linux training.
        import pathlib

        _orig_posix = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
            try:
                checkpoint = torch.load(self.model_path, weights_only=False, **load_kw)
            except TypeError:
                checkpoint = torch.load(self.model_path, **load_kw)
        finally:
            pathlib.PosixPath = _orig_posix

        config = checkpoint["config"]
        pick_model = config.init_obj("model_arch", pick_arch_module)
        pick_model = pick_model.to(self.device)
        pick_model.load_state_dict(checkpoint["state_dict"])
        pick_model.eval()
        self._model = pick_model
        return self

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, df_boxes: pd.DataFrame, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("PickRuntime not loaded")
        if df_boxes is None or df_boxes.empty:
            return []

        tmp = tempfile.TemporaryDirectory(prefix="pick_kie_")
        root = Path(tmp.name)
        img_dir = root / "images"
        tsv_dir = root / "boxes_and_transcripts"
        img_dir.mkdir()
        tsv_dir.mkdir()

        img_path = img_dir / "receipt.jpg"
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(img_path), bgr):
            raise RuntimeError("Failed to write temp image for PICK inference")

        tsv_path = tsv_dir / "receipt.tsv"
        work_df = df_boxes.sort_values(by=["y1", "x1"]).reset_index(drop=True).head(MAX_BOXES_NUM)
        tsv_path.write_text(df_boxes_to_pick_tsv(work_df), encoding="utf-8")

        document = self._Document(
            tsv_path,
            img_path,
            resized_image_size=RESIZED_IMAGE_SIZE,
            image_index=0,
            training=False,
            max_boxes_num=MAX_BOXES_NUM,
            max_transcript_len=MAX_TRANSCRIPT_LEN,
        )
        batch = self._BatchCollateFn(training=False)([document])
        with torch.no_grad():
            for key, val in batch.items():
                if val is not None:
                    batch[key] = val.to(self.device)
            output = self._model(**batch)
            logits = output["logits"]
            new_mask = output["new_mask"]
            mask = batch["mask"]
            text_segments = batch["text_segments"]
            text_length = batch["text_length"]

            best_paths = self._model.decoder.crf_layer.viterbi_tags(
                logits, mask=new_mask, logits_batch_first=True
            )
            predicted_tags = [path for path, _score in best_paths]
            decoded_tags_list = self._iob_index_to_str(predicted_tags)

        decoded_tags = decoded_tags_list[0]
        box_lengths = text_length.cpu().numpy()[0].tolist()
        n_boxes = min(len(box_lengths), int(document.boxes_num))
        box_lengths = box_lengths[:n_boxes]
        entities = _per_box_entities(decoded_tags, box_lengths)

        rows = work_df
        out: list[dict[str, Any]] = []
        for i in range(min(len(rows), len(entities))):
            row = rows.iloc[i]
            ent = entities[i]
            conf = 0.88 if ent != ENTITY_OTHER else 0.55
            out.append(
                _box_row_to_dict(
                    {
                        "x1": row["x1"],
                        "y1": row["y1"],
                        "x2": row["x2"],
                        "y2": row["y2"],
                        "text": row.get("text", ""),
                        "entity": ent,
                        "confidence": conf,
                    }
                )
            )
        tmp.cleanup()
        return out
