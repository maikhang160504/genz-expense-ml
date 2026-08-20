from __future__ import annotations



from pathlib import Path

from typing import Any



import cv2

import numpy as np

import pandas as pd

import torch

from PIL import Image



from .receipt_nlu import extract_receipt_summary

from .paddle_compat import init_paddleocr, paddle_lines, run_paddle_ocr, setup_paddle_env



setup_paddle_env()





class ReceiptOCRPipeline:

    """PaddleOCR + VietOCR + rule extraction. Use McOcrReceiptPipeline for PICK KIE."""



    def __init__(

        self,

        vietocr_weights: str | Path,

        device: str | None = None,

        use_paddle_cls: bool = True,  # kept for API compat; PaddleOCR 3.x ignores cls=

        paddle_use_gpu: bool | None = None,

    ):

        self.vietocr_weights = str(vietocr_weights)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._paddle = None

        self._vietocr = None

        self._use_paddle_cls = use_paddle_cls

        if paddle_use_gpu is None:

            paddle_use_gpu = not Path("/kaggle").is_dir()

        self._paddle_use_gpu = paddle_use_gpu



    def _init_paddle(self, use_gpu: bool | None = None):

        if self._paddle is not None and use_gpu is None:

            return

        self._paddle = init_paddleocr(

            self._paddle_use_gpu if use_gpu is None else use_gpu

        )



    def _init_vietocr(self):

        if self._vietocr is not None:

            return

        from vietocr.tool.config import Cfg

        from vietocr.tool.predictor import Predictor



        config = Cfg.load_config_from_name("vgg_transformer")

        config["weights"] = self.vietocr_weights

        config["device"] = self.device

        config["cnn"]["pretrained"] = False

        self._vietocr = Predictor(config)



    def load(self):

        self._init_paddle()

        self._init_vietocr()

        return self



    @staticmethod

    def decode_rgb_bytes(image_bytes: bytes) -> np.ndarray:

        arr = np.frombuffer(image_bytes, dtype=np.uint8)

        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if bgr is None:

            raise ValueError("Cannot decode image bytes")

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



    @staticmethod

    def _read_rgb(image_path: str | Path) -> np.ndarray:

        image = cv2.imread(str(image_path))

        if image is None:

            raise FileNotFoundError(f"Cannot read image: {image_path}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    def _vietocr_text(self, crop_rgb: np.ndarray) -> str:

        if crop_rgb.size == 0:

            return ""

        pil_img = Image.fromarray(crop_rgb)

        try:

            return (self._vietocr.predict(pil_img) or "").strip()

        except Exception:

            return ""



    def detect_polys(self, image_rgb: np.ndarray) -> list[list[Any]]:
        """Paddle detection only — quads for rotation corrector (no VietOCR)."""
        result = self._run_paddle(image_rgb)
        return [line[0] for line in paddle_lines(result)]

    def _run_paddle(self, image_rgb: np.ndarray):
        self._init_paddle()
        try:
            return run_paddle_ocr(self._paddle, image_rgb)
        except Exception as exc:
            self._paddle = None
            self._init_paddle(use_gpu=False)
            return run_paddle_ocr(self._paddle, image_rgb)



    def _boxes_from_paddle_result(self, image_rgb: np.ndarray, result) -> pd.DataFrame:

        self._init_vietocr()

        rows: list[dict[str, Any]] = []

        for line in paddle_lines(result):

            box, rec = line

            text_paddle, conf = rec if isinstance(rec, (list, tuple)) else (str(rec), 0.0)

            xs = [pt[0] for pt in box]

            ys = [pt[1] for pt in box]

            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            crop = image_rgb[y1:y2, x1:x2]

            text = self._vietocr_text(crop) or text_paddle

            rows.append(

                {

                    "x1": x1,

                    "y1": y1,

                    "x2": x2,

                    "y2": y2,

                    "text": text,

                    "conf_paddle": float(conf),

                }

            )

        return pd.DataFrame(rows)



    def ocr_boxes(self, image_rgb: np.ndarray) -> pd.DataFrame:

        result = self._run_paddle(image_rgb)

        return self._boxes_from_paddle_result(image_rgb, result)



    @staticmethod

    def group_lines(df: pd.DataFrame, line_threshold: int = 30) -> pd.DataFrame:

        if df.empty:

            return pd.DataFrame(columns=["line_text", "bbox"])



        df_sorted = df.sort_values(by="y1").reset_index(drop=True)

        lines: list[list[dict]] = []

        cur: list[dict] = []

        last_y = -10_000.0



        for _, row in df_sorted.iterrows():

            y_center = (row["y1"] + row["y2"]) / 2

            if abs(y_center - last_y) > line_threshold and cur:

                lines.append(cur)

                cur = []

            cur.append(row.to_dict())

            last_y = y_center

        if cur:

            lines.append(cur)



        out = []

        for line in lines:

            line_sorted = sorted(line, key=lambda r: r["x1"])

            line_text = " ".join(r["text"] for r in line_sorted if r.get("text"))

            x1 = min(r["x1"] for r in line)

            y1 = min(r["y1"] for r in line)

            x2 = max(r["x2"] for r in line)

            y2 = max(r["y2"] for r in line)

            out.append({"line_text": line_text, "bbox": [x1, y1, x2, y2]})

        return pd.DataFrame(out)



    def process_image(self, image_path: str | Path, split_mode: bool = False) -> dict[str, Any]:

        image_rgb = self._read_rgb(image_path)

        df_boxes = self.ocr_boxes(image_rgb)

        df_lines = self.group_lines(df_boxes)

        return extract_receipt_summary(df_lines, df_boxes=df_boxes, split_mode=split_mode)


