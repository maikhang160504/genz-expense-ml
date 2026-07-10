"""PaddleOCR 2.x / 3.x compatibility helpers (CPU, no MKLDNN on Py3.12+)."""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

PY312 = sys.version_info >= (3, 12)


def setup_paddle_env() -> None:
    """Call before `import paddle` / PaddleOCR (PIR+oneDNN bug on Paddle 3.3+ CPU)."""
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _try_text_detection(use_gpu: bool):
    """PaddleOCR 3.x: chỉ detection — VietOCR đọc chữ (tránh tải rec/doc/unwarp)."""
    try:
        from paddleocr import TextDetection
    except ImportError:
        return None

    attempts = [
        dict(model_name="PP-OCRv5_mobile_det", enable_mkldnn=False),
        dict(enable_mkldnn=False),
        {},
    ]
    for kwargs in attempts:
        try:
            det = TextDetection(**kwargs)
            det._receipt_det_only = True  # noqa: SLF001
            return det
        except Exception:
            continue
    return None


def paddle_ocr_init_kwargs(use_gpu: bool) -> dict[str, Any]:
    kw: dict[str, Any] = dict(lang="vi", use_gpu=use_gpu, enable_mkldnn=False)
    if PY312:
        kw.update(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    else:
        kw["use_angle_cls"] = True
    return kw


def init_paddleocr(use_gpu: bool | None = None):
    if use_gpu is None:
        use_gpu = not os.path.isdir("/kaggle")

    # Demo/local PaddleOCR 3.x: detection-only nhanh hơn full pipeline
    det = _try_text_detection(use_gpu)
    if det is not None:
        return det

    from paddleocr import PaddleOCR

    base = paddle_ocr_init_kwargs(use_gpu)
    attempts = [
        base,
        dict(lang="vi", use_gpu=use_gpu, enable_mkldnn=False),
        dict(lang="vi", enable_mkldnn=False),
        dict(lang="vi"),
    ]
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as exc:
            last_err = exc
    assert last_err is not None
    raise last_err


def run_paddle_ocr(engine, image: np.ndarray):
    if hasattr(engine, "predict"):
        raw = engine.predict(image)
        return list(raw) if raw is not None else []
    return engine.ocr(image)


def _res_dict(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item.get("res", item)
    for attr in ("json", "res"):
        if not hasattr(item, attr):
            continue
        val = getattr(item, attr)
        if callable(val):
            val = val()
        if isinstance(val, dict):
            return val.get("res", val)
    return None


def paddle_lines(ocr_result: Any) -> list[list[Any]]:
    if not ocr_result:
        return []
    # PaddleOCR 2.x: [page] where page = [[box, (text, conf)], ...]
    if isinstance(ocr_result[0], (list, tuple)):
        page = ocr_result[0]
        if page:
            first = page[0]
            if (
                isinstance(first, (list, tuple))
                and len(first) >= 2
                and isinstance(first[0], (list, tuple))
                and len(first[0]) >= 2
            ):
                return list(page)
    lines: list[list[Any]] = []
    for item in ocr_result:
        data = _res_dict(item)
        if not data:
            continue
        polys = data.get("dt_polys")
        if polys is None or len(polys) == 0:
            polys = data.get("rec_polys")
        if polys is None:
            polys = []
        texts = list(data.get("rec_texts") or [])
        scores = list(data.get("rec_scores") or [])
        for i, poly in enumerate(polys):
            box = np.asarray(poly).tolist()
            text = texts[i] if i < len(texts) else ""
            conf = float(scores[i]) if i < len(scores) else 0.0
            lines.append([box, (text, conf)])
    return lines
